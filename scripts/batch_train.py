"""
============================================
批量训练脚本 — 一键训练全部 8 个数据集
运行: python scripts/batch_train.py
============================================
"""
import os
import sys
import time
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app, db
from app.models.user import User
from app.models.dataset import Dataset
from app.models.training_job import TrainingJob
from app.services.dataset_service import DatasetService
from app.services.training_service import TrainingService
from app.executor.engine import get_executor

# 8 个数据集的训练配置
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


def batch_train():
    with app.app_context():
        # 获取 admin 用户
        admin = User.query.filter_by(username='admin').first()
        if not admin:
            print('[ERROR] admin 用户不存在，请先运行 seed_data.py')
            return

        print(f'用户: {admin.username} (id={admin.id})')
        print('=' * 60)

        results = []
        for i, cfg in enumerate(JOBS, 1):
            print(f'\n[{i}/8] {cfg["name"]}')
            print(f'  文件: {cfg["file"]}  |  {cfg["task"]} / {cfg["algo"]}')

            # 1. 检查或创建 Dataset
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'datasets', cfg['file'])
            if not os.path.exists(file_path):
                print(f'  [SKIP] 文件不存在: {file_path}')
                results.append({'name': cfg['name'], 'result': '文件不存在'})
                continue

            # 检查是否已有这个数据集
            ds = Dataset.query.filter_by(file_path=file_path).first()
            if not ds:
                # 直接用 model 创建
                import uuid as _uuid
                ds = Dataset(
                    name=cfg['file'].replace('_', ' ').replace('.csv', ''),
                    description=f'批量训练 - {cfg["task"]} / {cfg["algo"]}',
                    file_path=file_path,
                    file_size=os.path.getsize(file_path),
                    file_format='csv',
                    category='tabular',
                    is_public=True,
                    owner_id=admin.id,
                    status='ready',
                    uuid=str(_uuid.uuid4()),
                )
                db.session.add(ds)
                db.session.flush()

                # 自动解析
                import pandas as pd
                try:
                    df = pd.read_csv(file_path)
                    ds.row_count = len(df)
                    ds.column_count = len(df.columns)
                    summary = {
                        'columns': list(df.columns),
                        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
                        'missing_values': {col: int(df[col].isna().sum()) for col in df.columns if df[col].isna().sum() > 0},
                    }
                    ds.summary_json = json.dumps(summary, ensure_ascii=False)
                except Exception as e:
                    print(f'  [WARN] 自动解析失败: {e}')

                db.session.commit()
                print(f'  [NEW] 数据集已创建: id={ds.id}, {ds.row_count}行 x {ds.column_count}列')
            else:
                print(f'  [OK] 数据集已存在: id={ds.id}')

            # 2. 创建训练任务
            job, error = TrainingService.create_job(
                user=admin,
                name=cfg['name'],
                dataset_id=ds.id,
                description=f'批量训练: {cfg["task"]} / {cfg["algo"]}',
                framework='sklearn',
                total_epochs=1,
                ml_task_type=cfg['task'],
                algorithm=cfg['algo'],
                target_column=cfg['target'],
                test_size=0.2,
            )
            if error:
                print(f'  [ERROR] 创建任务失败: {error}')
                results.append({'name': cfg['name'], 'result': f'创建失败: {error}'})
                continue

            print(f'  [OK] 任务已创建: id={job.id}')

            # 3. 启动训练
            success, start_err = TrainingService.start_job(job)
            if not success:
                print(f'  [ERROR] 启动失败: {start_err}')
                results.append({'name': cfg['name'], 'result': f'启动失败: {start_err}'})
                continue

            print(f'  [RUN] 训练已提交...', end=' ')

            # 4. 等待完成
            timeout = 60
            waited = 0
            while waited < timeout:
                time.sleep(1)
                waited += 1
                # 刷新 job
                db.session.expire_all()
                job = db.session.get(TrainingJob, job.id)
                if job and job.is_finished:
                    break

            if job.is_finished:
                status = job.status
                metrics = json.loads(job.final_metrics_json) if job.final_metrics_json else {}
                acc = metrics.get('test_accuracy') or metrics.get('accuracy', 'N/A')
                if isinstance(acc, float):
                    acc = f'{acc:.2%}'
                r2 = metrics.get('test_r2') or metrics.get('r2', 'N/A')
                if isinstance(r2, float):
                    r2 = f'{r2:.4f}'

                print(f'{status}  |  Acc={acc}  R²={r2}  |  {job.duration_display}')
                results.append({
                    'name': cfg['name'],
                    'result': status,
                    'accuracy': acc,
                    'r2': r2,
                    'duration': job.duration_display,
                })
            else:
                print(f'超时 (仍在 {job.status})')
                results.append({'name': cfg['name'], 'result': f'超时({job.status})'})

        # 汇总
        print('\n' + '=' * 60)
        print('  训练结果汇总')
        print('=' * 60)
        for r in results:
            print(f"  {r['name']:<20s}  {r.get('result',''):<12s}  {r.get('accuracy',''):>8s}  {r.get('r2',''):>8s}  {r.get('duration','')}")

        # 保存汇总 JSON
        summary_path = os.path.join(app.config['UPLOAD_FOLDER'], 'datasets', 'training_results.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f'\n[OK] 结果已保存到: {summary_path}')


if __name__ == '__main__':
    batch_train()
