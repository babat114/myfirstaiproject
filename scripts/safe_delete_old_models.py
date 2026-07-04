"""
安全清理旧模型 — 分级删除候选

识别并安全清理以下四类模型:
  1. 已归档/失败模型: status='archived' 或 'failed'
  2. 旧格式模型: hyperparameters_json 只有 4-5 个基础键 (缺少 actual_params)
  3. 重复模型: 同 (dataset_id, algorithm) 组内保留最佳+最新
  4. 低质量模型: classification accuracy<0.5 / regression r2<0 / clustering silhouette<0.3

Usage:
  python scripts/safe_delete_old_models.py --dry-run --all
  python scripts/safe_delete_old_models.py --archived --dry-run
  python scripts/safe_delete_old_models.py --pre-backfill --dry-run
  python scripts/safe_delete_old_models.py --duplicates --dry-run
  python scripts/safe_delete_old_models.py --low-quality --dry-run
  python scripts/safe_delete_old_models.py --archived --pre-backfill --duplicates --force
"""

import sys
import os
import json
import shutil
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import create_app, db
from app.models.model_record import ModelRecord
from app.models.training_job import TrainingJob
from app.services.model_service import ModelService

# 旧格式判定: 仅包含这些基础键 (缺少 actual_params / param_source 之外的训练参数)
PRE_BACKFILL_KEYS = {'task_type', 'algorithm', 'target_column', 'test_size', 'param_source'}


# ═══════════════════════════════════════════════════════════════
# 扫描函数
# ═══════════════════════════════════════════════════════════════

def scan_archived(app):
    """扫描已归档/失败的模型。"""
    with app.app_context():
        models = ModelRecord.query.filter(
            ModelRecord.status.in_(['archived', 'failed'])
        ).all()
        results = []
        for m in models:
            results.append({
                'model_id': m.id, 'model_uuid': m.uuid,
                'model_name': m.name, 'status': m.status,
                'accuracy': m.accuracy, 'r2': m.r2,
                'category': 'archived_failed',
                'reasons': [f'status={m.status}'],
            })
        return results


def scan_pre_backfill(app):
    """扫描旧格式 (pre-backfill) 模型 — 仅包含基础键, 无 actual_params。"""
    with app.app_context():
        all_models = ModelRecord.query.filter_by(framework='sklearn').all()
        results = []
        for m in all_models:
            hp = m.hyperparameters_dict
            hp_keys = set(hp.keys())
            # Pre-backfill: 只有 4-5 个基础键, 无 learned params
            if hp_keys and hp_keys.issubset(PRE_BACKFILL_KEYS):
                results.append({
                    'model_id': m.id, 'model_uuid': m.uuid,
                    'model_name': m.name, 'status': m.status,
                    'accuracy': m.accuracy, 'r2': m.r2,
                    'category': 'pre_backfill',
                    'reasons': [f'旧格式: 仅{len(hp_keys)}个基础键, 无actual_params'],
                    'hp_keys': sorted(hp_keys),
                })
        return results


def scan_duplicates(app):
    """扫描重复模型 — 同 (dataset_id, algorithm) 组内保留 best+latest。"""
    with app.app_context():
        all_models = ModelRecord.query.order_by(ModelRecord.id).all()

        groups = defaultdict(list)
        for m in all_models:
            hp = m.hyperparameters_dict
            algo = hp.get('algorithm', 'unknown')
            ds_id = m.training_dataset_id or 0
            groups[(ds_id, algo)].append(m)

        results = []
        for (ds_id, algo), models in groups.items():
            if len(models) <= 1:
                continue

            # 按状态分组
            trained = [m for m in models if m.status == 'trained']
            drafts = [m for m in models if m.status == 'draft']
            others = [m for m in models if m.status not in ('trained', 'draft')]

            # 草稿: 只保留最新创建的一个
            if len(drafts) > 1:
                drafts_sorted = sorted(
                    drafts,
                    key=lambda m: m.created_at or datetime.min,
                    reverse=True,
                )
                for m in drafts_sorted[1:]:
                    results.append({
                        'model_id': m.id, 'model_uuid': m.uuid,
                        'model_name': m.name, 'status': m.status,
                        'accuracy': m.accuracy, 'r2': m.r2,
                        'category': 'duplicate_draft',
                        'reasons': [
                            f'草稿重复: {len(drafts)}个中保留1个 '
                            f'(ds={ds_id}, algo={algo})'
                        ],
                    })

            # 已训练: 保留 best (按 accuracy/r2/silhouette) 和 latest
            if len(trained) > 1:
                def _score(m):
                    hp = m.hyperparameters_dict
                    task = hp.get('task_type', '')
                    if task == 'classification':
                        return m.accuracy or 0
                    elif task == 'regression':
                        return m.r2 if m.r2 is not None else -999
                    elif task == 'clustering':
                        metrics = m.metrics_dict
                        return metrics.get('silhouette_score', 0)
                    return 0

                trained_sorted = sorted(
                    trained,
                    key=lambda m: (_score(m), m.created_at or datetime.min),
                    reverse=True,
                )
                best = trained_sorted[0]
                for m in trained_sorted[1:]:
                    results.append({
                        'model_id': m.id, 'model_uuid': m.uuid,
                        'model_name': m.name, 'status': m.status,
                        'accuracy': m.accuracy, 'r2': m.r2,
                        'category': 'duplicate_trained',
                        'reasons': [
                            f'训练重复: 保留 id={best.id} '
                            f'(ds={ds_id}, algo={algo})'
                        ],
                    })

            # 其他状态 (archived/failed): 全部可删
            for m in others:
                results.append({
                    'model_id': m.id, 'model_uuid': m.uuid,
                    'model_name': m.name, 'status': m.status,
                    'accuracy': m.accuracy, 'r2': m.r2,
                    'category': 'duplicate_other',
                    'reasons': [
                        f'状态={m.status}, 有更好模型 '
                        f'(ds={ds_id}, algo={algo})'
                    ],
                })

        return results


def scan_low_quality(app):
    """扫描低质量模型: classification acc<0.5 / regression r2<0 / clustering sil<0.3。"""
    with app.app_context():
        all_models = ModelRecord.query.filter_by(status='trained').all()
        results = []
        for m in all_models:
            hp = m.hyperparameters_dict
            task = hp.get('task_type', '')

            if task == 'classification':
                if m.accuracy is not None and m.accuracy < 0.5:
                    results.append({
                        'model_id': m.id, 'model_uuid': m.uuid,
                        'model_name': m.name, 'status': m.status,
                        'accuracy': m.accuracy,
                        'category': 'low_quality',
                        'reasons': [f'分类准确率={m.accuracy:.4f} < 0.5'],
                    })
            elif task == 'regression':
                if m.r2 is not None and m.r2 < 0:
                    results.append({
                        'model_id': m.id, 'model_uuid': m.uuid,
                        'model_name': m.name, 'status': m.status,
                        'r2': m.r2,
                        'category': 'low_quality',
                        'reasons': [f'回归R2={m.r2:.4f} < 0'],
                    })
            elif task == 'clustering':
                metrics = m.metrics_dict
                sil = metrics.get('silhouette_score')
                if sil is not None and sil < 0.3:
                    results.append({
                        'model_id': m.id, 'model_uuid': m.uuid,
                        'model_name': m.name, 'status': m.status,
                        'category': 'low_quality',
                        'reasons': [f'聚类Silhouette={sil:.4f} < 0.3'],
                    })
        return results


# ═══════════════════════════════════════════════════════════════
# 汇总与删除
# ═══════════════════════════════════════════════════════════════

def collect_candidates(args, app):
    """根据过滤标志收集删除候选 (自动去重)。"""
    all_candidates = []
    seen_ids = set()

    if args.archived or args.all:
        for item in scan_archived(app):
            if item['model_id'] not in seen_ids:
                seen_ids.add(item['model_id'])
                all_candidates.append(item)

    if args.pre_backfill or args.all:
        for item in scan_pre_backfill(app):
            if item['model_id'] not in seen_ids:
                seen_ids.add(item['model_id'])
                all_candidates.append(item)

    if args.duplicates or args.all:
        for item in scan_duplicates(app):
            if item['model_id'] not in seen_ids:
                seen_ids.add(item['model_id'])
                all_candidates.append(item)

    if args.low_quality or args.all:
        for item in scan_low_quality(app):
            if item['model_id'] not in seen_ids:
                seen_ids.add(item['model_id'])
                all_candidates.append(item)

    return all_candidates


def print_summary(candidates):
    """按类别分组打印汇总。"""
    print(f"\n{'=' * 70}")
    print(f"  安全清理旧模型 — 候选汇总")
    print(f"{'=' * 70}")

    by_category = defaultdict(list)
    for c in candidates:
        by_category[c['category']].append(c)

    labels = {
        'archived_failed': '已归档/失败模型',
        'pre_backfill': '旧格式 (pre-backfill)',
        'duplicate_trained': '训练模型重复',
        'duplicate_draft': '草稿模型重复',
        'duplicate_other': '其他状态重复',
        'low_quality': '低质量模型',
    }

    for cat in ['archived_failed', 'pre_backfill', 'duplicate_trained',
                'duplicate_draft', 'duplicate_other', 'low_quality']:
        items = by_category.get(cat, [])
        if not items:
            continue
        print(f"\n  [{len(items)}个] {labels.get(cat, cat)}:")
        for item in items[:5]:
            reasons = ', '.join(item.get('reasons', []))
            acc = item.get('accuracy')
            r2 = item.get('r2')
            score_str = ''
            if acc is not None:
                score_str = f'acc={acc:.4f}'
            elif r2 is not None:
                score_str = f'R2={r2:.4f}'
            print(f"    - id={item['model_id']:<5} {item['model_name'][:45]:45s} "
                  f"{score_str}  [{reasons}]")
        if len(items) > 5:
            print(f"    ... 还有 {len(items) - 5} 个")

    print(f"\n{'=' * 70}")
    print(f"  总计候选: {len(candidates)} 个模型")
    print(f"{'=' * 70}")


def delete_models_safely(to_delete, app, batch_size=10):
    """使用 ModelService.delete_model() 安全删除模型，分批提交。"""
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
                    success, error = ModelService.delete_model(model)
                    if success:
                        # 清理实验目录
                        exp_dir = experiments_dir / (model_uuid or '')
                        if exp_dir.exists():
                            try:
                                shutil.rmtree(exp_dir)
                            except Exception as e:
                                print(f"    [WARN] 无法删除实验目录 {model_uuid}: {e}")

                        deleted_count += 1
                        reasons = ', '.join(item.get('reasons', []))
                        print(f"    [OK] ID={model_id} {model_name[:40]:40s}  [{reasons}]")
                        log_entries.append({
                            'model_id': model_id, 'model_uuid': model_uuid,
                            'model_name': model_name, 'action': 'deleted',
                            'accuracy': item.get('accuracy'),
                            'r2': item.get('r2'),
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
    experiments_dir.mkdir(parents=True, exist_ok=True)
    log_path = experiments_dir / f'safe_delete_log_{start_time.strftime("%Y%m%d_%H%M%S")}.json'
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


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='安全清理旧模型 — 分级删除候选')
    parser.add_argument('--dry-run', action='store_true',
                        help='预览模式，不实际删除')
    parser.add_argument('--force', '-f', action='store_true',
                        help='跳过确认提示，直接删除')
    parser.add_argument('--archived', action='store_true',
                        help='清理已归档/失败的模型')
    parser.add_argument('--pre-backfill', action='store_true',
                        help='清理旧格式 (pre-backfill) 模型')
    parser.add_argument('--duplicates', action='store_true',
                        help='清理重复模型 (保留 best+latest)')
    parser.add_argument('--low-quality', action='store_true',
                        help='清理低质量模型 (acc<0.5 / r2<0 / sil<0.3)')
    parser.add_argument('--all', action='store_true',
                        help='启用全部四个过滤器')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='每批删除数量 (默认: 10)')
    args = parser.parse_args()

    if not any([args.archived, args.pre_backfill, args.duplicates,
                args.low_quality, args.all]):
        parser.print_help()
        print("\n[!] 请至少指定一个过滤标志 "
              "(--archived, --pre-backfill, --duplicates, --low-quality, --all)")
        return

    app = create_app()

    # 收集候选
    candidates = collect_candidates(args, app)

    if not candidates:
        print("\n没有找到符合条件的模型。")
        return

    # 打印汇总
    print_summary(candidates)

    if args.dry_run:
        print("\n>>> DRY RUN — 不会实际删除 <<<")
        return

    # 确认
    if args.force:
        print("\n[!] --force 模式: 跳过确认，直接删除！")
    else:
        print(f"\n[!] 此操作将从数据库和磁盘永久删除以上 {len(candidates)} 个模型！")
        confirm = input("输入 'yes' 确认删除，其他任意键取消: ").strip()
        if confirm.lower() != 'yes':
            print("已取消。")
            return

    # 执行删除
    deleted, skipped, errors = delete_models_safely(
        candidates, app, args.batch_size)

    # 验证
    with app.app_context():
        remaining = ModelRecord.query.count()
        print(f"\n验证:")
        print(f"  剩余模型总数: {remaining}")

    return deleted, skipped, errors


if __name__ == '__main__':
    main()
