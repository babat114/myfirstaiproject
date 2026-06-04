"""
============================================
多模型类型批量训练脚本
为7种模型类型各训练6个模型 (共42个)
============================================
"""
import os
import sys
import json
import time
from datetime import datetime, timezone

# 将项目根目录添加到 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app, db
from app.models.user import User
from app.models.dataset import Dataset
from app.models.training_job import TrainingJob
from app.models.model_record import ModelRecord
from app.services.training_service import TrainingService


# ===================================================================
# 训练配置: 每种模型类型 → 数据集ID, 目标列, ml_task_type, 算法列表
# ===================================================================
TRAINING_CONFIGS = [
    {
        'model_type': 'regression',
        'dataset_id': 22,  # Regression-Large-50K
        'target_column': 'target',
        'ml_task_type': 'regression',
        'framework': 'sklearn',
        'algorithms': [
            'linear_regression',
            'ridge',
            'random_forest_regressor',
            'svr',
            'gradient_boosting_regressor',
            'knn_regressor',
        ],
        'total_epochs': 5,
    },
    {
        'model_type': 'clustering',
        'dataset_id': 24,  # Clustering-Blobs-40K
        'target_column': 'cluster_label',
        'ml_task_type': 'classification',
        'framework': 'sklearn',
        'algorithms': [
            'random_forest',
            'logistic_regression',
            'svm',
            'knn',
            'gradient_boosting',
            'decision_tree',
        ],
        'total_epochs': 5,
    },
    {
        'model_type': 'nlp',
        'dataset_id': 25,  # NLP-TextFeatures-30K
        'target_column': 'sentiment',
        'ml_task_type': 'classification',
        'framework': 'sklearn',
        'algorithms': [
            'random_forest',
            'logistic_regression',
            'svm',
            'knn',
            'gradient_boosting',
            'decision_tree',
        ],
        'total_epochs': 5,
    },
    {
        'model_type': 'computer_vision',
        'dataset_id': 26,  # CV-ImageFeatures-35K
        'target_column': 'object_class',
        'ml_task_type': 'classification',
        'framework': 'sklearn',
        'algorithms': [
            'random_forest',
            'logistic_regression',
            'svm',
            'knn',
            'gradient_boosting',
            'decision_tree',
        ],
        'total_epochs': 5,
    },
    {
        'model_type': 'reinforcement',
        'dataset_id': 27,  # RL-StateAction-45K
        'target_column': 'optimal_action',
        'ml_task_type': 'classification',
        'framework': 'sklearn',
        'algorithms': [
            'random_forest',
            'logistic_regression',
            'svm',
            'knn',
            'gradient_boosting',
            'decision_tree',
        ],
        'total_epochs': 5,
    },
    {
        'model_type': 'generative',
        'dataset_id': 28,  # Gen-LatentSpace-60K
        'target_column': 'gen_class',
        'ml_task_type': 'classification',
        'framework': 'sklearn',
        'algorithms': [
            'random_forest',
            'logistic_regression',
            'svm',
            'knn',
            'gradient_boosting',
            'decision_tree',
        ],
        'total_epochs': 5,
    },
    {
        'model_type': 'other',
        'dataset_id': 29,  # Other-AnomalyMix-55K
        'target_column': 'anomaly_label',
        'ml_task_type': 'classification',
        'framework': 'sklearn',
        'algorithms': [
            'random_forest',
            'logistic_regression',
            'svm',
            'knn',
            'gradient_boosting',
            'decision_tree',
        ],
        'total_epochs': 5,
    },
]

# 算法中文名称映射
ALGO_NAMES_CN = {
    'random_forest': '随机森林',
    'logistic_regression': '逻辑回归',
    'svm': '支持向量机',
    'knn': 'K近邻',
    'gradient_boosting': '梯度提升',
    'decision_tree': '决策树',
    'linear_regression': '线性回归',
    'ridge': '岭回归',
    'random_forest_regressor': '随机森林回归',
    'svr': '支持向量回归',
    'gradient_boosting_regressor': '梯度提升回归',
    'knn_regressor': 'K近邻回归',
}

# 模型类型中文名称
MODEL_TYPE_NAMES_CN = {
    'regression': '回归',
    'clustering': '聚类',
    'nlp': '自然语言处理',
    'computer_vision': '计算机视觉',
    'reinforcement': '强化学习',
    'generative': '生成式',
    'other': '其他',
}


def find_admin_user():
    """查找管理员用户"""
    admin = User.query.filter_by(username='admin').first()
    if not admin:
        admin = User.query.filter_by(role='admin').first()
    if not admin:
        admin = User.query.first()
    if not admin:
        raise RuntimeError('数据库中没有用户，请先运行 seed_data.py')
    return admin


def train_one_model(app, admin, config, algo, idx):
    """
    训练单个模型: 创建Job → 创建trainer → 运行训练 → 保存结果

    Returns:
        dict with training results or None on failure
    """
    model_type = config['model_type']
    type_name_cn = MODEL_TYPE_NAMES_CN.get(model_type, model_type)
    algo_name_cn = ALGO_NAMES_CN.get(algo, algo)

    job_name = f'{algo_name_cn}-{type_name_cn}-{idx}'

    with app.app_context():
        try:
            # 创建训练任务和模型记录
            job, error = TrainingService.create_job(
                user=admin,
                name=job_name,
                dataset_id=config['dataset_id'],
                description=f'{type_name_cn}模型 — {algo_name_cn}算法 (批量训练 #{idx})',
                task_type='training',
                framework=config['framework'],
                total_epochs=config['total_epochs'],
                cpu_cores=1,
                memory_gb=2.0,
                ml_task_type=config['ml_task_type'],
                algorithm=algo,
                target_column=config['target_column'],
                test_size=0.2,
                model_type=model_type,
            )

            if error:
                print(f'  [FAIL] Create job failed: {error}')
                return None

            # 刷新job以获取完整数据
            job = db.session.get(TrainingJob, job.id)
            dataset = db.session.get(Dataset, config['dataset_id'])

            # 解析超参数
            hyperparams = {}
            if job.model and job.model.hyperparameters_json:
                try:
                    hyperparams = json.loads(job.model.hyperparameters_json)
                except (json.JSONDecodeError, TypeError):
                    pass

            # 创建训练器
            from app.executor.trainers.sklearn_trainer import SklearnTrainer
            trainer = SklearnTrainer(job, dataset, hyperparams)

            # 运行训练
            print(f'  [TRAIN] {job_name} ...', end=' ', flush=True)
            start_time = time.time()
            trainer.run()
            elapsed = time.time() - start_time

            # 获取结果
            job = db.session.get(TrainingJob, job.id)
            model = db.session.get(ModelRecord, job.model_id) if job and job.model_id else None

            if job and job.status == 'completed' and model:
                acc = model.accuracy or 0
                print(f'[OK] accuracy={acc:.4f} ({elapsed:.1f}s)')
                return {
                    'job_id': job.id,
                    'model_id': model.id,
                    'name': job_name,
                    'model_type': model_type,
                    'algorithm': algo,
                    'accuracy': acc,
                    'precision': model.precision,
                    'recall': model.recall,
                    'f1_score': model.f1_score,
                    'loss': model.loss,
                    'mse': model.metrics_dict.get('test_mse') if model.metrics_dict else None,
                    'r2': model.metrics_dict.get('test_r2') if model.metrics_dict else None,
                    'duration': round(elapsed, 1),
                }
            else:
                status = job.status if job else '?'
                error_msg = job.error_message if job else 'Job lost'
                print(f'[FAIL] {status}: {error_msg}')
                return None

        except Exception as e:
            print(f'[ERROR] {str(e)[:100]}')
            import traceback
            traceback.print_exc()
            return None


def main():
    """主流程"""
    app = create_app()

    print('=' * 70)
    print('  Multi-Type Model Batch Training')
    print(f'  Start: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")}')
    print('=' * 70)

    with app.app_context():
        admin = find_admin_user()
        print(f'  Admin: {admin.username} (ID={admin.id})')

        # 验证数据集
        for config in TRAINING_CONFIGS:
            ds = db.session.get(Dataset, config['dataset_id'])
            if not ds:
                print(f'  [FAIL] Dataset not found: ID={config["dataset_id"]} ({config["model_type"]})')
                return
            file_exists = os.path.exists(ds.file_path) if ds.file_path else False
            status = '[OK]' if file_exists else '[MISSING]'
            print(f'  {status} [{ds.id}] {ds.name} -> {config["model_type"]} ({ds.row_count} rows)')

    print()

    total_models = sum(len(c['algorithms']) for c in TRAINING_CONFIGS)
    all_results = []
    global_start = time.time()
    model_count = 0

    for config in TRAINING_CONFIGS:
        model_type = config['model_type']
        type_name_cn = MODEL_TYPE_NAMES_CN.get(model_type, model_type)
        print(f'--- {type_name_cn} ({model_type}) - {len(config["algorithms"])} models ---')

        for i, algo in enumerate(config['algorithms']):
            model_count += 1
            algo_name_cn = ALGO_NAMES_CN.get(algo, algo)
            print(f'[{model_count}/{total_models}] {algo_name_cn} ({algo})')
            result = train_one_model(app, admin, config, algo, i + 1)
            if result:
                all_results.append(result)

        print()

    # ============ Summary Report ============
    total_elapsed = time.time() - global_start
    print('=' * 70)
    print('  Training Complete Summary')
    print('=' * 70)
    print(f'  Total time: {total_elapsed/60:.1f} min')
    print(f'  Success: {len(all_results)}/{total_models}')
    print()

    # Summary by model type
    for config in TRAINING_CONFIGS:
        mt = config['model_type']
        type_name_cn = MODEL_TYPE_NAMES_CN.get(mt, mt)
        type_results = [r for r in all_results if r['model_type'] == mt]
        if type_results:
            accs = [r['accuracy'] for r in type_results if r['accuracy']]
            r2s = [r['r2'] for r in type_results if r['r2'] is not None]
            if accs:
                print(f'  [{type_name_cn}] ({len(type_results)} models): '
                      f'accuracy avg={sum(accs)/len(accs):.4f}, '
                      f'max={max(accs):.4f}, min={min(accs):.4f}')
            elif r2s:
                print(f'  [{type_name_cn}] ({len(type_results)} models): '
                      f'R2 avg={sum(r2s)/len(r2s):.4f}, '
                      f'max={max(r2s):.4f}, min={min(r2s):.4f}')

    # Save results to file
    results_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'experiments',
        f'multitype_training_results_{datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")}.json'
    )
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'total_duration_seconds': round(total_elapsed, 1),
            'successful': len(all_results),
            'total_attempted': total_models,
            'results': all_results,
        }, f, ensure_ascii=False, indent=2)

    print(f'\n  Results saved to: {results_file}')
    print('=' * 70)


if __name__ == '__main__':
    main()
