"""
============================================
批量训练脚本 v2 — 串行直接训练 (不走线程池)
运行: python scripts/batch_train_v2.py
============================================
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app, db
from app.models.user import User
from app.models.dataset import Dataset
from app.models.training_job import TrainingJob
from app.services.training_service import TrainingService
from app.executor.trainers.sklearn_trainer import SklearnTrainer
from app.executor.trainers.pytorch_trainer import PyTorchTrainer
from app.executor.callbacks import TrainingCallback

JOBS = [
    {'file': 'iris_classification.csv',          'name': 'Iris-随机森林',     'task': 'classification', 'algo': 'random_forest',              'target': 'species'},
    {'file': 'wine_classification.csv',          'name': 'Wine-SVM分类',      'task': 'classification', 'algo': 'svm',                       'target': 'wine_class'},
    {'file': 'breast_cancer_classification.csv', 'name': '乳腺癌-逻辑回归',     'task': 'classification', 'algo': 'logistic_regression',       'target': 'diagnosis'},
    {'file': 'digits_classification.csv',        'name': 'Digits-随机森林',    'task': 'classification', 'algo': 'random_forest',              'target': 'digit'},
    {'file': 'synthetic_binary_classification.csv','name': '合成二分类-KNN',    'task': 'classification', 'algo': 'knn',                       'target': 'label'},
    {'file': 'diabetes_regression.csv',          'name': '糖尿病-线性回归',     'task': 'regression',     'algo': 'linear_regression',          'target': 'disease_progression'},
    {'file': 'housing_regression.csv',           'name': '房价-随机森林回归',   'task': 'regression',     'algo': 'random_forest_regressor',    'target': 'median_house_value'},
    {'file': 'synthetic_regression.csv',         'name': '合成回归-SVR',      'task': 'regression',     'algo': 'svr',                       'target': 'target_value'},
]

app = create_app('development')


def train_all():
    with app.app_context():
        admin = User.query.filter_by(username='admin').first()
        if not admin:
            print('[ERROR] admin user not found')
            return

        print(f'User: {admin.username}\n')

        results = []
        for i, cfg in enumerate(JOBS, 1):
            t0 = time.time()
            print(f'[{i}/8] {cfg["name"]} ...', end=' ', flush=True)

            # 1. 创建或获取 Dataset
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'datasets', cfg['file'])
            ds = Dataset.query.filter_by(file_path=file_path).first()
            if not ds:
                import pandas as pd
                import uuid as _uuid
                df = pd.read_csv(file_path)
                ds = Dataset(
                    name=cfg['file'].replace('_', ' ').replace('.csv', ''),
                    description=f'{cfg["task"]} / {cfg["algo"]}',
                    file_path=file_path,
                    file_size=os.path.getsize(file_path),
                    file_format='csv',
                    category='tabular',
                    is_public=True,
                    owner_id=admin.id,
                    status='ready',
                    uuid=str(_uuid.uuid4()),
                    row_count=len(df),
                    column_count=len(df.columns),
                    summary_json=json.dumps({
                        'columns': list(df.columns),
                        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
                        'missing_values': {col: int(df[col].isna().sum()) for col in df.columns if df[col].isna().sum() > 0},
                    }, ensure_ascii=False),
                )
                db.session.add(ds)
                db.session.flush()
            db.session.commit()

            # 2. 创建 TrainingJob + ModelRecord (直接用 service)
            job, error = TrainingService.create_job(
                user=admin,
                name=cfg['name'],
                dataset_id=ds.id,
                description=f'{cfg["task"]} / {cfg["algo"]}',
                framework='sklearn',
                total_epochs=1,
                ml_task_type=cfg['task'],
                algorithm=cfg['algo'],
                target_column=cfg['target'],
                test_size=0.2,
            )
            if error:
                print(f'FAIL: {error}')
                results.append({'name': cfg['name'], 'result': f'创建失败: {error}'})
                continue

            # 3. 直接串行训练 (不走线程池)
            hyperparams = {
                'task_type': cfg['task'],
                'algorithm': cfg['algo'],
                'target_column': cfg['target'],
                'test_size': 0.2,
            }
            trainer = SklearnTrainer(job, ds, hyperparams)

            try:
                trainer.run()
                db.session.expire_all()
                job = db.session.get(TrainingJob, job.id)

                metrics = json.loads(job.final_metrics_json) if job.final_metrics_json else {}
                elapsed = time.time() - t0

                if cfg['task'] == 'classification':
                    acc = metrics.get('test_accuracy', 'N/A')
                    f1 = metrics.get('test_f1_score', 'N/A')
                    score_str = f'Acc={acc:.4f}  F1={f1:.4f}' if isinstance(acc, float) else f'Acc={acc}'
                else:
                    r2 = metrics.get('test_r2', 'N/A')
                    mse = metrics.get('test_mse', 'N/A')
                    score_str = f'R2={r2:.4f}  MSE={mse:.4f}' if isinstance(r2, float) else f'R2={r2}'

                print(f'OK  [{job.status}]  {score_str}  ({elapsed:.1f}s)')
                results.append({
                    'name': cfg['name'], 'status': job.status,
                    'accuracy': metrics.get('test_accuracy') or metrics.get('accuracy'),
                    'f1': metrics.get('test_f1_score') or metrics.get('f1_score'),
                    'r2': metrics.get('test_r2') or metrics.get('r2'),
                    'mse': metrics.get('test_mse') or metrics.get('mse'),
                    'duration': f'{elapsed:.1f}s',
                })

            except Exception as e:
                print(f'ERROR: {e}')
                results.append({'name': cfg['name'], 'result': f'训练异常: {e}'})

        # Summary
        print('\n' + '=' * 70)
        print(f'  {"Model":<25s} {"Status":<12s} {"Acc/F1":>22s} {"R2/MSE":>22s} {"Time"}')
        print('=' * 70)
        for r in results:
            name = r['name']
            status = r.get('status', r.get('result', '?'))
            cls = f"{r.get('accuracy',''):.4f} / {r.get('f1',''):.4f}" if r.get('accuracy') else '-'
            reg = f"{r.get('r2',''):.4f} / {r.get('mse',''):.4f}" if r.get('r2') else '-'
            dur = r.get('duration', '-')
            print(f'  {name:<25s} {status:<12s} {cls:>22s} {reg:>22s} {dur}')

        # 保存结果
        out = os.path.join(app.config['UPLOAD_FOLDER'], 'datasets', 'training_results.json')
        with open(out, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f'\n[OK] Results saved to: {out}')


if __name__ == '__main__':
    train_all()
