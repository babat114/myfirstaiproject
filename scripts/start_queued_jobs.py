"""
Start all queued training jobs via the running web server
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app, db
from app.models.training_job import TrainingJob
from app.services.training_service import TrainingService

app = create_app()

with app.app_context():
    # Find all queued jobs
    queued_jobs = TrainingJob.query.filter_by(status='queued').order_by(TrainingJob.id).all()
    print(f"Found {len(queued_jobs)} queued jobs")

    for job in queued_jobs:
        print(f"\n[Job {job.id}] {job.name}")
        print(f"  Status: {job.status}")
        print(f"  Dataset ID: {job.dataset_id}")
        print(f"  Framework: {job.framework}")

        # Verify dataset file exists
        if job.dataset:
            exists = os.path.exists(job.dataset.file_path)
            print(f"  Dataset file: {job.dataset.file_path}")
            print(f"  File exists: {exists}")
            if not exists:
                print(f"  [SKIP] Dataset file not found!")
                continue

        success, error = TrainingService.start_job(job)
        if success:
            print(f"  [OK] Started!")
        else:
            print(f"  [FAIL] {error}")
