"""
============================================
One-click training for all datasets
Creates and starts training jobs for each dataset
Usage: python scripts/train_all_datasets.py
============================================
"""
import sys
import os
import json
import time
import io

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app, db
from app.models.user import User
from app.models.dataset import Dataset
from app.services.training_service import TrainingService

# Dataset training configs (from datasets_summary.json)
DATASET_CONFIGS = {
    'iris classification': {
        'task_type': 'classification',
        'algorithm': 'random_forest',
        'target_column': 'species',
        'test_size': 0.2,
        'total_epochs': 1,
    },
    'wine classification': {
        'task_type': 'classification',
        'algorithm': 'svm',
        'target_column': 'wine_class',
        'test_size': 0.2,
        'total_epochs': 1,
    },
    'breast cancer classification': {
        'task_type': 'classification',
        'algorithm': 'logistic_regression',
        'target_column': 'diagnosis',
        'test_size': 0.2,
        'total_epochs': 1,
    },
    'digits classification': {
        'task_type': 'classification',
        'algorithm': 'random_forest',
        'target_column': 'digit',
        'test_size': 0.2,
        'total_epochs': 1,
    },
    'synthetic binary classification': {
        'task_type': 'classification',
        'algorithm': 'knn',
        'target_column': 'label',
        'test_size': 0.2,
        'total_epochs': 1,
    },
    'diabetes regression': {
        'task_type': 'regression',
        'algorithm': 'linear_regression',
        'target_column': 'disease_progression',
        'test_size': 0.2,
        'total_epochs': 1,
    },
    'housing regression': {
        'task_type': 'regression',
        'algorithm': 'random_forest_regressor',
        'target_column': 'median_house_value',
        'test_size': 0.2,
        'total_epochs': 1,
    },
    'synthetic regression': {
        'task_type': 'regression',
        'algorithm': 'svr',
        'target_column': 'target_value',
        'test_size': 0.2,
        'total_epochs': 1,
    },
}


def train_all():
    app = create_app()

    with app.app_context():
        # Get admin user
        admin = User.query.filter_by(username='admin').first()
        if not admin:
            print("[ERROR] Admin user not found. Run scripts/seed_data.py first.")
            return

        print(f"[USER] Using: {admin.username} (ID: {admin.id})")
        print(f"{'='*60}")

        # Get all ready datasets
        datasets = Dataset.query.filter_by(status='ready').order_by(Dataset.id).all()
        print(f"[DATA] Found {len(datasets)} ready datasets\n")

        created_jobs = []
        skipped = []

        for ds in datasets:
            # Skip seed data without actual files (IDs 1-4)
            if ds.id <= 4:
                skipped.append((ds.id, ds.name, "seed data, no actual file"))
                continue

            # Skip duplicate Iris (ID 5)
            if ds.id == 5:
                skipped.append((ds.id, ds.name, "duplicate Iris, using ID 6"))
                continue

            ds_name_lower = ds.name.lower().strip()

            # Find matching config
            config = None
            for key, cfg in DATASET_CONFIGS.items():
                if key in ds_name_lower or ds_name_lower in key:
                    config = cfg
                    break

            if not config:
                print(f"[SKIP] [{ds.id}] {ds.name}: no matching config")
                skipped.append((ds.id, ds.name, "no matching config"))
                continue

            print(f"[INFO] [{ds.id}] {ds.name}")
            print(f"       Type: {config['task_type']}, Algorithm: {config['algorithm']}")
            print(f"       Target: {config['target_column']}, Test size: {config['test_size']}")

            # Create training job
            job_name = f"{ds.name} - {config['algorithm']}"
            job, error = TrainingService.create_job(
                user=admin,
                name=job_name,
                dataset_id=ds.id,
                description=f"Train {ds.name} with {config['algorithm']}",
                task_type='training',
                framework='sklearn',
                total_epochs=config['total_epochs'],
                cpu_cores=2,
                memory_gb=4.0,
                ml_task_type=config['task_type'],
                algorithm=config['algorithm'],
                target_column=config['target_column'],
                test_size=config['test_size'],
            )

            if error:
                print(f"  [FAIL] Create failed: {error}")
                continue

            print(f"  [OK] Job created (ID: {job.id}, UUID: {job.uuid})")

            # Start training
            success, start_error = TrainingService.start_job(job)
            if success:
                print(f"  [GO] Training started!")
            else:
                print(f"  [WARN] Start failed: {start_error}")

            created_jobs.append(job)
            print()

        print(f"{'='*60}")
        print(f"\n[SUMMARY]")
        print(f"  Created & started: {len(created_jobs)} jobs")
        print(f"  Skipped: {len(skipped)}")
        for s in skipped:
            print(f"    - [{s[0]}] {s[1]}: {s[2]}")

        # Wait for training to complete
        print(f"\n[WAIT] Waiting for training jobs to complete...")
        all_completed = False
        max_wait = 300
        waited = 0
        while not all_completed and waited < max_wait:
            time.sleep(5)
            waited += 5

            running = 0
            completed = 0
            failed = 0
            for job in created_jobs:
                db.session.refresh(job)
                if job.status == 'running':
                    running += 1
                elif job.status == 'completed':
                    completed += 1
                elif job.status == 'failed':
                    failed += 1

            print(f"  [{waited:>3}s] Running: {running}, Completed: {completed}, Failed: {failed}")

            if running == 0:
                all_completed = True

        # Final results
        print(f"\n{'='*60}")
        print(f"[RESULTS] Final Results:")
        print(f"{'-'*60}")
        print(f"{'Job Name':<50} {'Status':<12} {'Score':<10}")
        print(f"{'-'*60}")
        for job in created_jobs:
            db.session.refresh(job)
            metrics = job.final_metrics_json
            if isinstance(metrics, str):
                metrics = json.loads(metrics) if metrics else {}
            elif metrics is None:
                metrics = {}

            if job.ml_task_type == 'classification':
                score = metrics.get('test_accuracy', metrics.get('accuracy', 'N/A'))
                if isinstance(score, float):
                    score = f"{score:.4f}"
            else:
                score = metrics.get('test_r2', metrics.get('r2_score', 'N/A'))
                if isinstance(score, float):
                    score = f"{score:.4f}"

            print(f"{job.name:<50} {job.status:<12} {str(score):<10}")

        print(f"{'-'*60}")
        print("\n[DONE] All training jobs processed!")


if __name__ == '__main__':
    train_all()
