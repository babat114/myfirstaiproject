"""
清理过拟合模型 — 升级版 (v2)

支持两种模式:
  1. 传统模式 (默认): 直接扫描 DB，accuracy >= THRESHOLD 视为过拟合
  2. 报告模式: 读取 diagnose_model_quality.py 生成的 JSON 报告，按 Tier 分级处理

升级要点:
  - 使用 ModelService.delete_model() 替代原始 SQL (正确处理 FK 和缓存)
  - 支持 --dry-run 预览
  - 支持 --tier {1,2,3} 分级处理
  - 批次删除，避免长事务

Usage:
  python scripts/clean_overfit_models.py                              # 传统模式
  python scripts/clean_overfit_models.py --report-json REPORT.json    # 基于诊断报告
  python scripts/clean_overfit_models.py --report-json R.json --dry-run  # 预览
  python scripts/clean_overfit_models.py --report-json R.json --tier 1 --force  # 自动删Tier1
"""
import sys
import os
import json
import shutil
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import create_app, db
from app.models.model_record import ModelRecord
from app.models.training_job import TrainingJob
from app.services.model_service import ModelService

THRESHOLD = 0.999  # 传统模式阈值: 准确率 >= 99.9% 视为过拟合


def legacy_scan(app):
    """传统模式: 直接扫描 DB 查找过拟合模型"""
    with app.app_context():
        models = ModelRecord.query.all()
        overfit = [m for m in models if m.accuracy is not None and m.accuracy >= THRESHOLD]

        # 按数据集分组
        by_dataset = {}
        for m in overfit:
            ds_name = "unknown"
            if m.training_dataset_id:
                from app.models.dataset import Dataset
                ds = Dataset.query.get(m.training_dataset_id)
                ds_name = ds.name if ds else f"dataset_id={m.training_dataset_id}"
            by_dataset.setdefault(ds_name, []).append(m)

        print(f"{'=' * 70}")
        print(f"  过拟合模型清理工具 v2 (传统模式)")
        print(f"  阈值: accuracy >= {THRESHOLD} ({THRESHOLD * 100:.1f}%)")
        print(f"  待删除: {len(overfit)} 个模型 (共 {len(models)} 个)")
        print(f"{'=' * 70}")

        if not overfit:
            print("\n没有找到过拟合模型，退出。")
            return [], []

        print("\n按数据集分组:")
        for ds_name, model_list in sorted(by_dataset.items(), key=lambda x: -len(x[1])):
            print(f"\n  [{len(model_list)}个] {ds_name}:")
            for m in model_list:
                print(f"    ID={m.id:>4} | acc={m.accuracy:.4f} | f1={m.f1_score or 0:.4f} | "
                      f"name={m.name[:50]} | uuid={m.uuid}")

        # 构建旧的统一列表 (不分Tier)
        to_delete = [{
            'model_id': m.id,
            'model_uuid': m.uuid,
            'model_name': m.name,
            'accuracy': m.accuracy,
            'dataset_name': ds_name,
            'tier': 'legacy',
        } for m in overfit]
        return to_delete, []


def find_duplicates(app):
    """查找所有重复模型: 同数据集+同算法+同准确率 → 保留最佳1个。

    去重规则:
      - 训练模型: 按 (dataset_id, algorithm) 分组, 同组内 accuracy 差异 < 0.0001
        视为重复, 保留 accuracy 最高 + f1 最高的 1 个
      - 草稿模型: 保留最新创建的 1 个
      - perfect_accuracy (acc >= 0.999): 标记为 definite_delete
    """
    import json
    from collections import Counter
    from app.models.dataset import Dataset

    with app.app_context():
        all_models = ModelRecord.query.order_by(ModelRecord.id).all()

        # ── Build model info list ──
        model_infos = []
        for m in all_models:
            algo = 'unknown'
            if m.hyperparameters_json:
                try:
                    hp = json.loads(m.hyperparameters_json) if isinstance(m.hyperparameters_json, str) else m.hyperparameters_json
                    algo = hp.get('algorithm', hp.get('algo', 'unknown'))
                except Exception:
                    pass

            ds_id = m.training_dataset_id or 0
            ds_name = 'unknown'
            if m.training_dataset_id:
                ds = Dataset.query.get(m.training_dataset_id)
                ds_name = ds.name if ds else f'ds_{m.training_dataset_id}'

            model_infos.append({
                'm': m, 'algo': algo, 'ds_id': ds_id,
                'ds_name': ds_name, 'acc': m.accuracy,
            })

        # ── Group by (dataset_id, algorithm) ──
        groups = {}
        for info in model_infos:
            key = (info['ds_id'], info['algo'])
            groups.setdefault(key, []).append(info)

        to_delete = []
        keep_count = 0

        for key, items in groups.items():
            if len(items) <= 1:
                keep_count += len(items)
                continue

            ds_name = items[0]['ds_name']
            algo = items[0]['algo']

            # Separated trained vs draft
            trained = [x for x in items if x['m'].status == 'trained']
            drafts = [x for x in items if x['m'].status == 'draft']

            # ── Drafts: keep newest ──
            if len(drafts) > 1:
                drafts_sorted = sorted(drafts, key=lambda x: x['m'].created_at or datetime.min, reverse=True)
                for info in drafts_sorted[1:]:
                    to_delete.append({
                        'model_id': info['m'].id, 'model_uuid': info['m'].uuid,
                        'model_name': info['m'].name, 'accuracy': info['acc'],
                        'dataset_name': ds_name, 'algorithm': algo,
                        'tier': 'draft_duplicate',
                        'reasons': [f'草稿重复: {len(drafts)}个中保留1个'],
                    })
                keep_count += 1
            elif len(drafts) == 1:
                keep_count += 1

            # ── Trained: cluster by accuracy (4dp tolerance), keep best per cluster ──
            if len(trained) <= 1:
                keep_count += len(trained)
                continue

            trained_sorted = sorted(trained, key=lambda x: (
                x['acc'] if x['acc'] is not None else -1,
                x['m'].f1_score or 0,
            ), reverse=True)

            # Cluster by accuracy tolerance
            bands = []
            current_band = [trained_sorted[0]]
            for info in trained_sorted[1:]:
                prev_acc = current_band[0]['acc']
                curr_acc = info['acc']
                if prev_acc is not None and curr_acc is not None and abs(prev_acc - curr_acc) < 0.0001:
                    current_band.append(info)
                else:
                    bands.append(current_band)
                    current_band = [info]
            bands.append(current_band)

            # Check if any non-perfect band exists (for perfect_accuracy cleanup)
            has_good_band = any(
                b[0]['acc'] is not None and b[0]['acc'] < 0.999
                for b in bands
            )

            for band in bands:
                best = band[0]
                is_perfect = best['acc'] is not None and best['acc'] >= 0.999

                # If there's a good model with acc < 0.999, delete ALL perfect_accuracy models
                if is_perfect and has_good_band:
                    # Delete ALL models in this perfect_accuracy band
                    for info in band:
                        to_delete.append({
                            'model_id': info['m'].id, 'model_uuid': info['m'].uuid,
                            'model_name': info['m'].name, 'accuracy': info['acc'],
                            'dataset_name': ds_name, 'algorithm': algo,
                            'tier': 'definite_delete',
                            'reasons': ['perfect_accuracy (存在更好模型 acc<0.999)'],
                        })
                    # Don't keep any — find the best acc < 0.999 model from other bands
                    continue

                keep_count += 1
                for info in band[1:]:
                    reasons = [f'{len(band)}个重复模型保留1个 (acc={best["acc"]:.4f})']
                    tier = 'duplicate'
                    if is_perfect:
                        reasons.append('perfect_accuracy')
                        tier = 'definite_delete'
                    to_delete.append({
                        'model_id': info['m'].id, 'model_uuid': info['m'].uuid,
                        'model_name': info['m'].name, 'accuracy': info['acc'],
                        'dataset_name': ds_name, 'algorithm': algo,
                        'tier': tier, 'reasons': reasons,
                    })

        # ── Print summary ──
        total = len(model_infos)
        dup_count = len(to_delete)
        print(f"{'=' * 70}")
        print(f"  模型去重扫描")
        print(f"{'=' * 70}")
        print(f"  总模型数: {total}")
        print(f"  唯一模型 (保留): {keep_count}")
        print(f"  重复模型 (待删除): {dup_count}")
        print(f"  去重分组数: {len(groups)}")
        print(f"{'=' * 70}")

        tier_counts = Counter(d['tier'] for d in to_delete)
        print(f"\n  按类型分布:")
        labels = {
            'definite_delete': 'perfect_accuracy (acc=1.0)  ',
            'duplicate': '训练模型重复',
            'draft_duplicate': '草稿模型重复',
        }
        for tier, count in sorted(tier_counts.items()):
            print(f"    {labels.get(tier, tier)}: {count} 个")

        if not to_delete:
            print("\n  没有重复模型！")

        return to_delete, []


def load_from_report(report_path, tier_filter):
    """从诊断报告加载待删除模型列表"""
    with open(report_path, 'r', encoding='utf-8') as f:
        report = json.load(f)

    to_delete = []

    tiers_to_load = []
    if tier_filter is None:
        tiers_to_load = [1, 2]  # 默认处理 Tier 1 + 2
    else:
        tiers_to_load = tier_filter

    if 1 in tiers_to_load:
        for item in report.get('definite_delete', []):
            item['tier'] = 'definite_delete'
            to_delete.append(item)
    if 2 in tiers_to_load:
        for item in report.get('high_risk_delete', []):
            item['tier'] = 'high_risk_delete'
            to_delete.append(item)
    if 3 in tiers_to_load:
        for item in report.get('warning_only', []):
            item['tier'] = 'warning_only'
            to_delete.append(item)

    summary = report.get('summary', {})
    print(f"{'=' * 70}")
    print(f"  过拟合模型清理工具 v2 (报告模式)")
    print(f"  报告: {report_path}")
    print(f"  生成时间: {report.get('generated_at', 'unknown')}")
    print(f"{'=' * 70}")
    print(f"  报告中: Tier1={summary.get('definite_delete', 0)}, "
          f"Tier2={summary.get('high_risk_delete', 0)}, "
          f"Tier3={summary.get('warning_only', 0)}")
    print(f"  本次处理 Tier: {tiers_to_load} → 共 {len(to_delete)} 个模型")
    print(f"{'=' * 70}")

    if not to_delete:
        print("\n没有需要删除的模型，退出。")
        return to_delete, []

    return to_delete, []


def dry_run_preview(to_delete):
    """预览模式: 列出所有待删除模型"""
    print(f"\n>>> DRY RUN 预览 — 不会实际删除 <<<\n")
    print(f"待删除模型: {len(to_delete)} 个\n")

    # 按 tier 分组显示
    for tier_name in ['definite_delete', 'high_risk_delete', 'warning_only', 'duplicate', 'draft_duplicate', 'legacy']:
        tier_models = [m for m in to_delete if m.get('tier') == tier_name]
        if not tier_models:
            continue
        labels = {
            'definite_delete': '[Tier 1] 必定删除',
            'high_risk_delete': '[Tier 2] 高风险',
            'warning_only': '[Tier 3] 仅警告',
            'duplicate': '[去重] 训练模型重复',
            'draft_duplicate': '[去重] 草稿模型重复',
            'legacy': '[Legacy] 传统模式',
        }
        print(f"  {labels.get(tier_name, tier_name)}: {len(tier_models)} 个")
        for m in tier_models[:5]:
            acc = m.get('accuracy', 'N/A')
            reasons = ', '.join(m.get('reasons', ['legacy_threshold']))
            print(f"    - {m.get('model_name', '?')[:50]:50s}  acc={acc}  "
                  f"dataset={m.get('dataset_name', 'N/A'):20s}  reasons=[{reasons}]")
        if len(tier_models) > 5:
            print(f"    ... 还有 {len(tier_models) - 5} 个")
        print()

    print(f"实际删除请去掉 --dry-run 参数")
    print(f"仅删除 Tier 1: 加上 --tier 1")
    print(f"自动确认: 加上 --force 或 -f")


def delete_models_safely(to_delete, app, batch_size=10):
    """使用 ModelService.delete_model() 安全删除模型，分批提交"""
    experiments_dir = Path(__file__).resolve().parent.parent / 'experiments'

    deleted_count = 0
    skipped_count = 0
    errors = []
    log_entries = []

    total = len(to_delete)
    start_time = datetime.now()

    print(f"\n开始删除 {total} 个模型 (批次大小: {batch_size})...")
    print(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    for batch_start in range(0, total, batch_size):
        batch = to_delete[batch_start:batch_start + batch_size]
        batch_num = batch_start // batch_size + 1
        total_batches = (total + batch_size - 1) // batch_size

        print(f"\n  批次 {batch_num}/{total_batches} ({len(batch)} 个模型)...")

        for item in batch:
            model_id = item.get('model_id')
            model_uuid = item.get('model_uuid')
            model_name = item.get('model_name', 'unknown')

            with app.app_context():
                # 重新获取模型 (前一批次可能已提交删除)
                model = db.session.get(ModelRecord, model_id)
                if model is None:
                    skipped_count += 1
                    print(f"    [SKIP] ID={model_id} {model_name} — 已不存在")
                    log_entries.append({
                        'model_id': model_id, 'model_name': model_name,
                        'action': 'skip', 'reason': 'already_deleted',
                    })
                    continue

                try:
                    # 使用 ModelService 安全删除
                    success, error = ModelService.delete_model(model)
                    if success:
                        # ModelService 删了 model 文件，还需清理实验目录
                        exp_dir = experiments_dir / (model_uuid or '')
                        if exp_dir.exists():
                            try:
                                shutil.rmtree(exp_dir)
                            except Exception as e:
                                print(f"    [WARN] 无法删除实验目录 {model_uuid}: {e}")

                        deleted_count += 1
                        print(f"    [OK] ID={model_id} {model_name[:40]:40s} "
                              f"acc={item.get('accuracy', 'N/A')} "
                              f"reasons={item.get('reasons', [])}")
                        log_entries.append({
                            'model_id': model_id, 'model_uuid': model_uuid,
                            'model_name': model_name, 'action': 'deleted',
                            'accuracy': item.get('accuracy'),
                            'dataset': item.get('dataset_name'),
                            'reasons': item.get('reasons', []),
                        })
                    else:
                        skipped_count += 1
                        errors.append(f"删除失败 {model_name}: {error}")
                        print(f"    [FAIL] ID={model_id} {model_name}: {error}")
                        log_entries.append({
                            'model_id': model_id, 'model_name': model_name,
                            'action': 'fail', 'error': error,
                        })
                except Exception as e:
                    skipped_count += 1
                    errors.append(f"异常 {model_name}: {e}")
                    print(f"    [ERR] ID={model_id} {model_name}: {e}")
                    log_entries.append({
                        'model_id': model_id, 'model_name': model_name,
                        'action': 'error', 'error': str(e),
                    })
                    try:
                        db.session.rollback()
                    except Exception:
                        pass

        # 批次间简短报告
        progress = min(batch_start + batch_size, total)
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"  批次 {batch_num} 完成: 累计 {deleted_count} 删除, {skipped_count} 跳过, "
              f"进度 {progress}/{total} ({elapsed:.0f}s)")

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n{'=' * 70}")
    print(f"  删除完成!")
    print(f"  总耗时: {elapsed:.0f}s")
    print(f"  成功删除: {deleted_count}")
    print(f"  跳过: {skipped_count}")
    if errors:
        print(f"  错误: {len(errors)}")
        for e in errors[:5]:
            print(f"    - {e}")
    print(f"{'=' * 70}")

    # 保存删除日志
    log_path = experiments_dir / f'cleanup_log_{start_time.strftime("%Y%m%d_%H%M%S")}.json'
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump({
            'start_time': start_time.isoformat(),
            'elapsed_seconds': elapsed,
            'deleted_count': deleted_count,
            'skipped_count': skipped_count,
            'errors': errors,
            'entries': log_entries,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n删除日志已保存: {log_path}")

    return deleted_count, skipped_count, errors


def main():
    parser = argparse.ArgumentParser(description='清理过拟合模型 v2')
    parser.add_argument('--report-json', type=str, default=None,
                        help='诊断报告 JSON 路径 (默认: 传统 DB 扫描模式)')
    parser.add_argument('--dedup', action='store_true',
                        help='去重模式: 查找并删除所有重复模型 (同数据集+同算法+同准确率)')
    parser.add_argument('--dry-run', action='store_true',
                        help='预览模式，不实际删除')
    parser.add_argument('--tier', type=str, default=None,
                        help='处理的 Tier，逗号分隔 (1,2,3)，默认: 1,2。仅报告模式有效')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='每批删除数量 (默认: 10)')
    parser.add_argument('--force', '-f', action='store_true',
                        help='跳过确认提示，直接删除')
    args = parser.parse_args()

    app = create_app()

    # ── 加载待删除列表 ──
    to_delete = []
    errors = []

    if args.dedup:
        to_delete, errors = find_duplicates(app)
    elif args.report_json:
        if not os.path.exists(args.report_json):
            print(f"[ERROR] 报告文件不存在: {args.report_json}")
            print(f"  请先运行: python scripts/diagnose_model_quality.py")
            sys.exit(1)

        tier_filter = None
        if args.tier:
            try:
                tier_filter = [int(t.strip()) for t in args.tier.split(',')]
            except ValueError:
                print(f"[ERROR] Invalid --tier value: {args.tier}")
                sys.exit(1)

        to_delete, errors = load_from_report(args.report_json, tier_filter)
    else:
        to_delete, errors = legacy_scan(app)

    if not to_delete:
        return

    # ── 按 tier 统计 ──
    tiers = {}
    for m in to_delete:
        t = m.get('tier', 'legacy')
        tiers[t] = tiers.get(t, 0) + 1
    print(f"\n待删除分布:")
    for t, c in sorted(tiers.items()):
        labels = {
            'definite_delete': 'Tier 1 (必定删除)',
            'high_risk_delete': 'Tier 2 (高风险)',
            'warning_only': 'Tier 3 (仅警告)',
            'duplicate': '去重 (训练重复)',
            'draft_duplicate': '去重 (草稿重复)',
            'legacy': '传统模式',
        }
        print(f"  {labels.get(t, t)}: {c} 个")

    # ── Dry run 预览 ──
    if args.dry_run:
        dry_run_preview(to_delete)
        return

    # ── 确认 ──
    if args.tier and set(args.tier.split(',')) == {'3'}:
        print("\n[!] 警告: 你正在处理 Tier 3 (仅警告) 模型。这些模型并非明确过拟合，仅健康评分偏低。")

    if args.force:
        print("\n[!] --force 模式: 跳过确认，直接删除！")
    else:
        print(f"\n[!] 此操作将从数据库和磁盘永久删除以上 {len(to_delete)} 个模型！")
        confirm = input("输入 'yes' 确认删除，其他任意键取消: ").strip()
        if confirm.lower() != 'yes':
            print("已取消。")
            return

    # ── 执行删除 ──
    deleted, skipped, del_errors = delete_models_safely(to_delete, app, args.batch_size)
    errors.extend(del_errors)

    # ── 验证 ──
    with app.app_context():
        remaining = ModelRecord.query.count()
        still_overfit = ModelRecord.query.filter(
            ModelRecord.accuracy >= THRESHOLD
        ).count()
        print(f"\n验证:")
        print(f"  剩余模型总数: {remaining}")
        if args.dedup:
            # Count remaining duplicates
            from collections import Counter
            import json as _json
            all_m = ModelRecord.query.all()
            groups = {}
            for m in all_m:
                algo = '?'
                if m.hyperparameters_json:
                    try:
                        hp = _json.loads(m.hyperparameters_json) if isinstance(m.hyperparameters_json, str) else m.hyperparameters_json
                        algo = hp.get('algorithm', '?')
                    except: pass
                key = (m.training_dataset_id or 0, algo)
                groups.setdefault(key, []).append(m)
            dup_remaining = sum(len(v) - 1 for v in groups.values() if len(v) > 1)
            print(f"  剩余重复模型: {dup_remaining}")
        else:
            print(f"  剩余 acc>={THRESHOLD}: {still_overfit}")

    # ── 去重后刷新 watchdog 报告 ──
    if args.dedup and not args.dry_run and deleted > 0:
        print(f"\n刷新 watchdog 报告...")
        try:
            from scripts.quality_watchdog import WatchdogEngine
            engine = WatchdogEngine(app)
            report = engine.scan_all_models()
            report_path = Path(__file__).resolve().parent.parent / 'experiments' / 'watchdog_report.json'
            with open(report_path, 'w', encoding='utf-8') as f:
                import json as _json2
                _json2.dump(report, f, ensure_ascii=False, indent=2)
            summary = report.get('summary', {})
            print(f"  watchdog_report.json 已更新: "
                  f"Tier1={summary.get('definite_delete',0)} "
                  f"Tier2={summary.get('high_risk_delete',0)} "
                  f"Healthy={summary.get('healthy',0)}")
        except Exception as e:
            print(f"  [WARN] 无法刷新 watchdog_report: {e}")


if __name__ == '__main__':
    main()
