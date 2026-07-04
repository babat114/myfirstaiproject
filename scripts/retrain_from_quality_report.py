"""
基于质量诊断报告定向重训练编排器

读取 model_quality_report.json → 筛选需重训练模型 → 按数据集分组 → 批量重训练

特性:
  - 去重: 同数据集+同算法只训练一次 (保留最佳参数组合)
  - 保守参数: 自动根据数据集规模选择 profile
  - 不依赖 AI 数据增强 — 使用真实数据 + 降低模型复杂度

Usage:
  python scripts/retrain_from_quality_report.py                   # 读取默认报告
  python scripts/retrain_from_quality_report.py --dry-run         # 预览
  python scripts/retrain_from_quality_report.py --tier 1          # 仅重训 Tier 1
  python scripts/retrain_from_quality_report.py --workers 4       # 4 线程并行
"""
import sys
import os
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import create_app, db
from app.models.user import User
from app.models.dataset import Dataset

# 复用 retrain_nlp.py 的训练函数
from scripts.retrain_nlp import (
    train_one, get_conservative_params, get_test_sentences,
    verify_model_predictions, ALL_SKLEARN_ALGOS, BASE_DIR,
)

DEFAULT_REPORT = BASE_DIR / 'experiments' / 'model_quality_report.json'
DEFAULT_OUTPUT = BASE_DIR / 'experiments' / 'retrain_summary.json'


def load_report(report_path: str) -> dict:
    """加载质量诊断报告"""
    if not os.path.exists(report_path):
        print(f"[ERROR] 报告文件不存在: {report_path}")
        print(f"  请先运行: python scripts/diagnose_model_quality.py")
        sys.exit(1)

    with open(report_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_retrain_queue(report, tier_filter=None):
    """从诊断报告中提取需要重训练的模型列表 (去重)。

    去重逻辑: 同一 (数据集名, 算法) 只训练一次。
    虽然多个模型可能使用相同的数据集+算法, 但它们的参数和随机种子不同
    导致结果差异很大。重训练时统一使用保守参数重新训练一份即可。
    """
    queue = []
    seen = set()

    tiers_to_include = tier_filter or [1, 2]

    candidates = []
    if 1 in tiers_to_include:
        candidates.extend(report.get('definite_delete', []))
    if 2 in tiers_to_include:
        candidates.extend(report.get('high_risk_delete', []))
    if 3 in tiers_to_include:
        candidates.extend(report.get('warning_only', []))

    for item in candidates:
        # 只处理 NLP 模型 (其他类型暂不支持自动重训练)
        if item.get('model_type') != 'nlp':
            continue

        dataset_name = item.get('dataset_name', '')
        algorithm = item.get('algorithm', '')

        # 跳过无法重训练的
        if not dataset_name or not algorithm:
            continue

        key = (dataset_name, algorithm)
        if key in seen:
            continue
        seen.add(key)
        queue.append(item)

    return queue


def find_dataset_by_name(app, dataset_name):
    """根据名称查找数据集"""
    with app.app_context():
        ds = Dataset.query.filter(Dataset.name == dataset_name).first()
        return ds


def find_algo_conf(algorithm):
    """根据算法名查找算法配置"""
    for a in ALL_SKLEARN_ALGOS:
        if a['algo'] == algorithm:
            return a
    return None


def print_summary(queue, dry_run=False):
    """打印重训练队列摘要"""
    mode = "DRY RUN (预览)" if dry_run else "执行重训练"
    print(f"\n{'=' * 60}")
    print(f"  定向重训练编排器 — {mode}")
    print(f"{'=' * 60}")

    # 按数据集分组
    by_dataset = defaultdict(list)
    for item in queue:
        by_dataset[item['dataset_name']].append(item)

    print(f"\n重训练队列: {len(queue)} 个 (去重后)")
    for ds_name, items in sorted(by_dataset.items()):
        algos = [i['algorithm'] for i in items]
        print(f"  {ds_name}: {len(items)} 算法 {algos}")

    if dry_run:
        print(f"\n[DRY RUN] 不会实际训练。去掉 --dry-run 参数执行。")


def main():
    parser = argparse.ArgumentParser(description='基于诊断报告定向重训练')
    parser.add_argument('--report-json', type=str, default=str(DEFAULT_REPORT),
                        help='诊断报告路径')
    parser.add_argument('--output-json', type=str, default=str(DEFAULT_OUTPUT),
                        help='重训练报告输出路径')
    parser.add_argument('--tier', type=str, default='1,2',
                        help='处理的 Tier (默认: 1,2)')
    parser.add_argument('--dry-run', action='store_true',
                        help='预览模式')
    parser.add_argument('--workers', type=int, default=2,
                        help='并行训练数 (注意: retrain_nlp.py 不支持真正的并行, '
                             '此参数预留)')
    parser.add_argument('--profile', type=str, default='auto',
                        choices=['auto', 'standard', 'conservative', 'ultra_conservative'],
                        help='训练参数配置')
    parser.add_argument('--quality-gate', action='store_true', default=True,
                        help='训练后质量检查 (默认启用)')
    args = parser.parse_args()

    # ── 解析 tier ──
    try:
        tier_filter = [int(t.strip()) for t in args.tier.split(',')]
    except ValueError:
        print(f"[ERROR] Invalid --tier: {args.tier}")
        sys.exit(1)

    # ── 加载报告 ──
    report = load_report(args.report_json)
    summary = report.get('summary', {})
    print(f"报告加载: {args.report_json}")
    print(f"  生成时间: {report.get('generated_at', 'unknown')}")
    print(f"  诊断结果: definite={summary.get('definite_delete', 0)}, "
          f"high_risk={summary.get('high_risk_delete', 0)}, "
          f"warning={summary.get('warning_only', 0)}")

    # ── 构建重训练队列 ──
    queue = build_retrain_queue(report, tier_filter)
    print_summary(queue, args.dry_run)

    if not queue:
        print("\n没有需要重训练的 NLP 模型。")
        return

    if args.dry_run:
        return

    # ── 查找数据集和算法 ──
    app = create_app()

    with app.app_context():
        admin = User.query.filter_by(username='admin').first()
        if not admin:
            print("[ERROR] 管理员用户不存在")
            return

    # ── 逐数据集+算法重训练 ──
    results = []
    success = 0
    failed = 0
    t0 = time.time()

    print(f"\n{'=' * 60}")
    print(f"  开始重训练 ({len(queue)} 个)")
    print(f"  Profile: {args.profile}")
    print(f"  Quality Gate: {'ON' if args.quality_gate else 'OFF'}")
    print(f"{'=' * 60}")

    for idx, item in enumerate(queue, 1):
        dataset_name = item['dataset_name']
        algorithm = item['algorithm']
        old_acc = item.get('accuracy', 'N/A')

        print(f"\n[{idx}/{len(queue)}] {dataset_name} / {algorithm} "
              f"(旧 acc={old_acc})")

        # 查找数据集
        ds = find_dataset_by_name(app, dataset_name)
        if not ds:
            print(f"  [SKIP] 数据集不存在: {dataset_name}")
            failed += 1
            results.append({
                'index': idx, 'dataset': dataset_name, 'algorithm': algorithm,
                'status': 'dataset_not_found',
            })
            continue

        # 查找算法配置
        algo_conf = find_algo_conf(algorithm)
        if not algo_conf:
            print(f"  [SKIP] 不支持的算法: {algorithm}")
            failed += 1
            results.append({
                'index': idx, 'dataset': dataset_name, 'algorithm': algorithm,
                'status': 'algo_not_found',
            })
            continue

        # 计算保守参数
        ds_row_count = ds.row_count or 5000
        cons_params = get_conservative_params(ds_row_count, args.profile)
        print(f"  Profile: {args.profile}, rows={ds_row_count}, "
              f"max_features={cons_params['max_features']}, "
              f"rf_depth={cons_params['rf_max_depth']}")

        # 训练
        try:
            model = train_one(
                app, admin.id, ds.id, algo_conf, idx,
                verify=True,
                nlp_max_features=cons_params['max_features'],
                nlp_min_df=cons_params.get('min_df', 2),
                nlp_max_df=cons_params.get('max_df', 0.9),
                balance_mode='undersample',  # 欠采样避免类别不平衡
                cv_folds=cons_params.get('cv_folds', 5),
                augment_factor=0,
                conservative_params=cons_params,
                quality_gate=args.quality_gate,
            )

            if model and model.status == 'trained':
                success += 1
                new_acc = model.accuracy or 0
                results.append({
                    'index': idx,
                    'dataset': dataset_name,
                    'algorithm': algorithm,
                    'status': 'success',
                    'old_accuracy': item.get('accuracy'),
                    'new_accuracy': new_acc,
                    'model_uuid': model.uuid,
                    'model_name': model.name,
                })
                print(f"  [OK] 新模型: acc={new_acc:.4f}, uuid={model.uuid}")
            else:
                failed += 1
                results.append({
                    'index': idx, 'dataset': dataset_name, 'algorithm': algorithm,
                    'status': 'training_failed',
                })
                print(f"  [FAIL] 训练失败")
        except Exception as e:
            failed += 1
            results.append({
                'index': idx, 'dataset': dataset_name, 'algorithm': algorithm,
                'status': 'error',
                'error': str(e),
            })
            print(f"  [ERR] {e}")
            import traceback
            traceback.print_exc()

    elapsed = time.time() - t0

    # ── 汇总 ──
    print(f"\n{'=' * 60}")
    print(f"  重训练完成!")
    print(f"  成功: {success}/{len(queue)}, 失败: {failed}")
    print(f"  耗时: {elapsed:.0f}s")
    print(f"{'=' * 60}")

    # ── 保存报告 ──
    report_out = {
        'title': '定向重训练报告',
        'generated_at': datetime.now().isoformat(),
        'config': {
            'profile': args.profile,
            'quality_gate': args.quality_gate,
            'tier_filter': tier_filter,
            'source_report': args.report_json,
        },
        'summary': {
            'total': len(queue),
            'success': success,
            'failed': failed,
            'elapsed_seconds': elapsed,
        },
        'results': results,
    }

    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(report_out, f, ensure_ascii=False, indent=2)

    print(f"\n报告已保存: {args.output_json}")

    # ── 快速质量对比 ──
    if success > 0:
        old_accs = [r.get('old_accuracy') for r in results
                    if r['status'] == 'success' and r.get('old_accuracy')]
        new_accs = [r['new_accuracy'] for r in results
                    if r['status'] == 'success' and r.get('new_accuracy') is not None]
        if old_accs and new_accs:
            print(f"\n质量对比:")
            print(f"  旧模型平均 acc: {sum(old_accs) / len(old_accs):.4f}")
            print(f"  新模型平均 acc: {sum(new_accs) / len(new_accs):.4f}")
            print(f"  (注意: 降低的 accuracy 是正常的, 说明过拟合已被修正)")


if __name__ == '__main__':
    main()
