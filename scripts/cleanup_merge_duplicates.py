"""
综合模型清理脚本:
1. 修复 retrain/anti-overfit 模型的 hyperparameters (从名称提取 algorithm)
2. 修复 v2 模型的空 hyperparameters
3. 回填所有缺失的描述
4. 去重 (同 dataset + 同 algorithm 保留最佳)
5. 删除 draft 模型 (存在 trained 版本时)
"""
import sys, json, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import create_app, db
from app.models.model_record import ModelRecord
from app.models.training_job import TrainingJob
from app.services.model_service import ModelService
from collections import defaultdict

# 算法名称 → 算法 key 映射 (从模型名中提取)
ALGO_NAME_MAP = {
    'random_forest': 'random_forest',
    'logistic_regression': 'logistic_regression',
    'svm': 'svm',
    'knn': 'knn',
    'gradient_boosting': 'gradient_boosting',
    'decision_tree': 'decision_tree',
    'mlp': 'mlp',
    'transformer_bert': 'transformer_bert',
    'linear_regression': 'linear_regression',
    'ridge': 'ridge',
    'svr': 'svr',
    'kmeans': 'kmeans',
    'dbscan': 'dbscan',
    'agglomerative': 'agglomerative',
}

# NLP 前缀 → algorithm key
NLP_PREFIX_MAP = {
    'NLP-RandomForest': 'random_forest',
    'NLP-LogisticReg': 'logistic_regression',
    'NLP-SVM': 'svm',
    'NLP-MLP': 'mlp',
    'NLP-BERT': 'transformer_bert',
}

from app.utils.algorithm_info import ALGORITHM_INFO

app = create_app()


def extract_algo_from_name(name):
    """从模型名中提取算法 key"""
    # "真实数据集: XXX - random_forest (retrain)" 或 "(anti-overfit)"
    m = re.search(r'-\s+(\w+)\s+\((retrain|anti-overfit)\)', name)
    if m:
        algo = m.group(1)
        if algo in ALGO_NAME_MAP:
            return ALGO_NAME_MAP[algo]
    # "NLP-RandomForest-Douban-Movie-Re - 模型"
    for prefix, algo in NLP_PREFIX_MAP.items():
        if prefix in name:
            return algo
    return None


def fix_model_hyperparameters(dry_run=False):
    """修复 retrain/anti-overfit/v2 模型的 hyperparameters"""
    models = ModelRecord.query.all()
    fixed = 0
    skipped = 0

    for model in models:
        hp = model.hyperparameters_dict or {}
        if hp.get('algorithm'):
            skipped += 1
            continue

        algo = extract_algo_from_name(model.name)
        if not algo:
            skipped += 1
            continue

        new_hp = dict(hp)
        new_hp['algorithm'] = algo
        if 'task_type' not in new_hp:
            new_hp['task_type'] = 'classification'
        if 'framework' not in new_hp:
            new_hp['framework'] = 'sklearn'
        if 'target_column' not in new_hp:
            new_hp['target_column'] = 'label'

        if dry_run:
            print(f'[DRY-RUN] id={model.id} "{model.name}" → algo={algo}')
        else:
            model.hyperparameters_json = json.dumps(new_hp, ensure_ascii=False)
            if model.description is None:
                algo_info = ALGORITHM_INFO.get(algo, {})
                algo_cn = algo_info.get('name', algo)
                algo_desc = algo_info.get('description', '')
                model.description = f'使用{algo_cn}算法训练 - {algo_desc}' if algo_desc else f'使用{algo_cn}算法训练'
            fixed += 1

    if fixed > 0 and not dry_run:
        db.session.commit()
        print(f'已修复 {fixed} 个模型的 hyperparameters')
    else:
        print(f'[DRY-RUN] 将修复 {fixed} 个模型 (跳过 {skipped} 个)')
    return fixed


def backfill_descriptions(dry_run=False):
    """回填所有缺失的描述"""
    from app.utils.algorithm_info import ALGORITHM_INFO
    models = ModelRecord.query.all()
    updated = 0
    skipped = 0

    for model in models:
        if model.description is not None:
            skipped += 1
            continue

        hp = model.hyperparameters_dict or {}
        algo = hp.get('algorithm', '')
        if not algo:
            skipped += 1
            continue

        algo_info = ALGORITHM_INFO.get(algo)
        if not algo_info:
            skipped += 1
            continue

        new_desc = f'使用{algo_info["name"]}算法训练 - {algo_info["description"]}'

        if dry_run:
            print(f'[DRY-RUN] id={model.id} "{model.name}" → desc set')
        else:
            model.description = new_desc
            updated += 1

    if updated > 0 and not dry_run:
        db.session.commit()
        print(f'已更新 {updated} 个模型描述 (跳过 {skipped} 个)')
    else:
        print(f'[DRY-RUN] 将更新 {updated} 个模型描述 (跳过 {skipped} 个)')


def find_and_delete_duplicates(dry_run=False, force=False):
    """去重: 同 (dataset_id, algorithm) 保留最佳, 删除其他"""
    models = ModelRecord.query.order_by(ModelRecord.id).all()

    groups = defaultdict(list)
    for m in models:
        hp = m.hyperparameters_dict or {}
        algo = hp.get('algorithm', 'unknown')
        ds_id = m.training_dataset_id or 0
        groups[(ds_id, algo)].append(m)

    to_delete = []
    keep_ids = set()

    for (ds_id, algo), items in groups.items():
        if len(items) <= 1:
            keep_ids.add(items[0].id)
            continue

        trained = [m for m in items if m.status == 'trained']
        drafts = [m for m in items if m.status == 'draft']

        # Draft: keep newest only
        if len(drafts) > 1:
            drafts.sort(key=lambda m: m.created_at or '', reverse=True)
            for m in drafts[1:]:
                to_delete.append((m, 'draft_duplicate'))
            keep_ids.add(drafts[0].id)

        # Trained: keep best accuracy
        if len(trained) > 1:
            trained.sort(key=lambda m: (m.accuracy or 0, m.f1_score or 0), reverse=True)
            keep_ids.add(trained[0].id)
            for m in trained[1:]:
                to_delete.append((m, 'trained_duplicate'))
        elif len(trained) == 1:
            keep_ids.add(trained[0].id)

    # Draft when trained exists
    for (ds_id, algo), items in groups.items():
        trained = [m for m in items if m.status == 'trained']
        drafts = [m for m in items if m.status == 'draft']
        if trained and drafts:
            for m in drafts:
                if all(t.id != m.id for t in trained):
                    if m.id not in [d[0].id for d in to_delete]:
                        to_delete.append((m, 'draft_with_trained_exists'))

    print(f'\n重复清理计划:')
    for m, reason in to_delete:
        print(f'  [DELETE] id={m.id:>4} acc={str(m.accuracy):>6} status={m.status:>7} '
              f'reason={reason:>30} name="{m.name[:50]}"')

    print(f'\n保留: {len(keep_ids)} 个, 删除: {len(to_delete)} 个')

    if dry_run or not to_delete:
        return to_delete

    # 执行删除
    if not force:
        confirm = input(f'\n确认删除 {len(to_delete)} 个模型? (yes/no): ').strip()
        if confirm.lower() != 'yes':
            print('已取消')
            return to_delete

    deleted = 0
    errors = []
    for m, reason in to_delete:
        try:
            success, err = ModelService.delete_model(m)
            if success:
                deleted += 1
                print(f'  [OK] 已删除 id={m.id} "{m.name[:50]}"')
            else:
                errors.append(f'id={m.id}: {err}')
                print(f'  [FAIL] id={m.id}: {err}')
        except Exception as e:
            errors.append(f'id={m.id}: {e}')
            print(f'  [ERR] id={m.id}: {e}')
        try:
            db.session.commit()
        except:
            db.session.rollback()

    print(f'\n删除完成: {deleted}/{len(to_delete)}, 错误: {len(errors)}')
    for e in errors[:5]:
        print(f'  ERROR: {e}')
    return to_delete


def main():
    import argparse
    parser = argparse.ArgumentParser(description='综合模型清理工具')
    parser.add_argument('--fix-hp', action='store_true', help='修复 hyperparameters')
    parser.add_argument('--backfill-desc', action='store_true', help='回填描述')
    parser.add_argument('--dedup', action='store_true', help='去重')
    parser.add_argument('--all', action='store_true', help='执行全部步骤')
    parser.add_argument('--dry-run', action='store_true', help='仅预览')
    parser.add_argument('--force', '-f', action='store_true', help='跳过确认直接删除')
    args = parser.parse_args()

    if not any([args.fix_hp, args.backfill_desc, args.dedup, args.all]):
        parser.print_help()
        return

    with app.app_context():
        if args.all or args.fix_hp:
            print('=== Step 1: 修复 hyperparameters ===')
            fix_model_hyperparameters(dry_run=args.dry_run)

        if args.all or args.backfill_desc:
            print('\n=== Step 2: 回填描述 ===')
            backfill_descriptions(dry_run=args.dry_run)

        if args.all or args.dedup:
            print('\n=== Step 3: 去重 ===')
            find_and_delete_duplicates(dry_run=args.dry_run, force=args.force)

    print('\n完成!')


if __name__ == '__main__':
    main()
