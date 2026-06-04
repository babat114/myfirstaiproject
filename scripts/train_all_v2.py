"""
Complete training script - creates and runs training for ALL 8 datasets
Uses the TrainingService + Executor with proper thread-safe session handling
"""
import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app, db
from app.models.user import User
from app.models.dataset import Dataset
from app.models.training_job import TrainingJob
from app.services.training_service import TrainingService

# Dataset configs
CONFIGS = [
    # (dataset_id, name_key, task_type, algorithm, target_column)
    (6,  'iris_classification',             'classification', 'random_forest',              'species'),
    (7,  'wine_classification',             'classification', 'svm',                        'wine_class'),
    (8,  'breast_cancer_classification',    'classification', 'logistic_regression',        'diagnosis'),
    (9,  'digits_classification',           'classification', 'random_forest',              'digit'),
    (10, 'synthetic_binary_classification', 'classification', 'knn',                        'label'),
    (11, 'diabetes_regression',             'regression',     'linear_regression',           'disease_progression'),
    (12, 'housing_regression',              'regression',     'random_forest_regressor',     'median_house_value'),
    (13, 'synthetic_regression',            'regression',     'svr',                        'target_value'),
]

app = create_app()

with app.app_context():
    admin = User.query.filter_by(username='admin').first()
    if not admin:
        print("[ERROR] Admin user not found")
        sys.exit(1)

    print(f"User: {admin.username} (ID: {admin.id})")
    print(f"{'='*60}")

    created_jobs = []

    for ds_id, name_key, task_type, algorithm, target_col in CONFIGS:
        ds = Dataset.query.get(ds_id)
        if not ds:
            print(f"[SKIP] Dataset {ds_id} not found")
            continue
        if ds.status != 'ready':
            print(f"[SKIP] Dataset {ds_id} ({ds.name}) not ready: {ds.status}")
            continue

        print(f"\n[Dataset {ds_id}] {ds.name}")
        print(f"  Task: {task_type}, Algorithm: {algorithm}")

        job_name = f"{ds.name} - {algorithm}"
        job, error = TrainingService.create_job(
            user=admin,
            name=job_name,
            dataset_id=ds.id,
            description=f"Train {ds.name} with {algorithm}",
            task_type='training',
            framework='sklearn',
            total_epochs=1,
            cpu_cores=2,
            memory_gb=4.0,
            ml_task_type=task_type,
            algorithm=algorithm,
            target_column=target_col,
            test_size=0.2,
        )

        if error:
            print(f"  [FAIL] Create: {error}")
            continue

        print(f"  [OK] Created (ID={job.id})")

        success, start_err = TrainingService.start_job(job)
        if success:
            print(f"  [OK] Started!")
            created_jobs.append(job)
        else:
            print(f"  [FAIL] Start: {start_err}")
            created_jobs.append(job)

    print(f"\n{'='*60}")
    print(f"Submitted {len(created_jobs)} jobs. Waiting for completion...")

    # Wait for all to finish
    max_wait = 300
    waited = 0
    while waited < max_wait:
        time.sleep(5)
        waited += 5

        running = 0
        completed = 0
        failed = 0
        queued = 0

        for job in created_jobs:
            db.session.refresh(job)
            if job.status == 'running':
                running += 1
            elif job.status == 'completed':
                completed += 1
            elif job.status == 'failed':
                failed += 1
            elif job.status == 'queued':
                queued += 1

        print(f"  [{waited:>3}s] Running:{running} Completed:{completed} Failed:{failed} Queued:{queued}")

        if running == 0 and queued == 0:
            break

    # Print results
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS:")
    print(f"{'-'*65}")
    print(f"{'Job Name':<45} {'Status':<12} {'Score':<10}")
    print(f"{'-'*65}")

    for job in created_jobs:
        db.session.refresh(job)
        metrics = job.final_metrics_json
        if isinstance(metrics, str):
            metrics = json.loads(metrics) if metrics else {}
        elif metrics is None:
            metrics = {}

        task_type = job.ml_task_type or 'classification'
        if task_type == 'classification':
            score = metrics.get('test_accuracy', metrics.get('accuracy', 'N/A'))
        else:
            score = metrics.get('test_r2', metrics.get('r2_score', 'N/A'))

        if isinstance(score, float):
            score = f"{score:.4f}"
        print(f"{job.name:<45} {job.status:<12} {str(score):<10}")

    print(f"{'-'*65}")
    print("[DONE]")
