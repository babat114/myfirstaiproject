"""
============================================
补全训练脚本 — 仅训练缺失的模型
NLP(6) + CV(1) + RL(6) + Gen(6) + Other(6) = 25 models
============================================
"""
import os
import sys
import json
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app, db
from app.models.user import User
from app.models.dataset import Dataset
from app.models.training_job import TrainingJob
from app.models.model_record import ModelRecord
from app.services.training_service import TrainingService

ALGO_NAMES_CN = {
    'random_forest': 'RandomForest', 'logistic_regression': 'LogisticReg',
    'svm': 'SVM', 'knn': 'KNN', 'gradient_boosting': 'GradientBoost',
    'decision_tree': 'DecisionTree',
}

# Only the missing configurations
TRAINING_CONFIGS = [
    {
        'model_type': 'nlp',
        'dataset_id': 25,
        'target_column': 'sentiment',
        'ml_task_type': 'classification',
        'algorithms': ['random_forest','logistic_regression','svm','knn','gradient_boosting','decision_tree'],
        'total_epochs': 5,
    },
    {
        'model_type': 'computer_vision',
        'dataset_id': 26,
        'target_column': 'object_class',
        'ml_task_type': 'classification',
        'algorithms': ['decision_tree'],  # only missing this one
        'total_epochs': 5,
    },
    {
        'model_type': 'reinforcement',
        'dataset_id': 27,
        'target_column': 'optimal_action',
        'ml_task_type': 'classification',
        'algorithms': ['random_forest','logistic_regression','svm','knn','gradient_boosting','decision_tree'],
        'total_epochs': 5,
    },
    {
        'model_type': 'generative',
        'dataset_id': 28,
        'target_column': 'gen_class',
        'ml_task_type': 'classification',
        'algorithms': ['random_forest','logistic_regression','svm','knn','gradient_boosting','decision_tree'],
        'total_epochs': 5,
    },
    {
        'model_type': 'other',
        'dataset_id': 29,
        'target_column': 'anomaly_label',
        'ml_task_type': 'classification',
        'algorithms': ['random_forest','logistic_regression','svm','knn','gradient_boosting','decision_tree'],
        'total_epochs': 5,
    },
]


def find_admin_user():
    admin = User.query.filter_by(username='admin').first()
    if not admin:
        admin = User.query.first()
    return admin


def train_one_model(app, admin, config, algo, idx):
    model_type = config['model_type']
    algo_name = ALGO_NAMES_CN.get(algo, algo)
    job_name = f'{algo_name}-{model_type}-{idx}'

    with app.app_context():
        try:
            job, error = TrainingService.create_job(
                user=admin,
                name=job_name,
                dataset_id=config['dataset_id'],
                description=f'{model_type} model - {algo} (#{idx})',
                task_type='training',
                framework='sklearn',
                total_epochs=config['total_epochs'],
                cpu_cores=1, memory_gb=2.0,
                ml_task_type=config['ml_task_type'],
                algorithm=algo,
                target_column=config['target_column'],
                test_size=0.2,
                model_type=model_type,
            )
            if error:
                print(f'  [FAIL] Create job: {error}')
                return None

            job = db.session.get(TrainingJob, job.id)
            dataset = db.session.get(Dataset, config['dataset_id'])

            hyperparams = {}
            if job.model and job.model.hyperparameters_json:
                try:
                    hyperparams = json.loads(job.model.hyperparameters_json)
                except Exception:
                    pass

            from app.executor.trainers.sklearn_trainer import SklearnTrainer
            trainer = SklearnTrainer(job, dataset, hyperparams)

            print(f'  [TRAIN] {job_name} ...', end=' ', flush=True)
            start_time = time.time()
            trainer.run()
            elapsed = time.time() - start_time

            job = db.session.get(TrainingJob, job.id)
            model = db.session.get(ModelRecord, job.model_id) if job and job.model_id else None

            if job and job.status == 'completed' and model:
                acc = model.accuracy or 0
                r2 = model.metrics_dict.get('test_r2') if model.metrics_dict else None
                if r2 is not None:
                    print(f'[OK] R2={r2:.4f} ({elapsed:.1f}s)')
                else:
                    print(f'[OK] acc={acc:.4f} ({elapsed:.1f}s)')
                return {'job_id': job.id, 'model_id': model.id, 'name': job_name,
                        'model_type': model_type, 'algorithm': algo,
                        'accuracy': acc, 'r2': r2, 'duration': round(elapsed, 1)}
            else:
                status = job.status if job else '?'
                err = job.error_message if job else 'lost'
                print(f'[FAIL] {status}: {err}')
                return None
        except Exception as e:
            print(f'[ERROR] {str(e)[:120]}')
            import traceback
            traceback.print_exc()
            return None


def main():
    app = create_app()
    print('=' * 60)
    print('  Complete Missing Models Training')
    print(f'  Start: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")}')
    print('=' * 60)

    with app.app_context():
        admin = find_admin_user()
        print(f'  Admin: {admin.username}')

        for config in TRAINING_CONFIGS:
            ds = db.session.get(Dataset, config['dataset_id'])
            f_exists = os.path.exists(ds.file_path) if ds and ds.file_path else False
            print(f'  [{"OK" if f_exists else "MISS"}] [{ds.id}] {ds.name} -> {config["model_type"]}')

    total = sum(len(c['algorithms']) for c in TRAINING_CONFIGS)
    all_results = []
    global_start = time.time()
    count = 0

    for config in TRAINING_CONFIGS:
        mt = config['model_type']
        print(f'\n--- {mt} ({len(config["algorithms"])} models) ---')
        for i, algo in enumerate(config['algorithms']):
            count += 1
            print(f'[{count}/{total}] {algo}')
            result = train_one_model(app, admin, config, algo, i + 1)
            if result:
                all_results.append(result)

    elapsed = time.time() - global_start
    print(f'\n{"="*60}')
    print(f'  Done: {len(all_results)}/{total} models in {elapsed/60:.1f} min')
    print(f'{"="*60}')

    # Save results
    results_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'experiments',
        f'completion_results_{datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")}.json'
    )
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({'timestamp': datetime.now(timezone.utc).isoformat(),
                    'duration': round(elapsed, 1), 'success': len(all_results),
                    'total': total, 'results': all_results}, f, ensure_ascii=False, indent=2)
    print(f'  Results: {results_file}')


if __name__ == '__main__':
    main()
