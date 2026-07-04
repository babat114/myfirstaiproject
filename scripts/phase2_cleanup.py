"""Phase 2: Delete Tier 1 overfitting models from quality report."""
import sys, os, json, shutil
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app, db
from app.models.model_record import ModelRecord
from app.services.model_service import ModelService

REPORT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'experiments', 'model_quality_report.json')

with open(REPORT_PATH, 'r', encoding='utf-8') as f:
    report = json.load(f)

tier1 = report.get('definite_delete', [])
print(f'Tier 1 models to delete: {len(tier1)}')

app = create_app()
exp_dir = Path(__file__).resolve().parent.parent / 'experiments'

deleted = 0
for item in tier1:
    mid = item['model_id']
    with app.app_context():
        model = db.session.get(ModelRecord, mid)
        if model is None:
            print(f'  SKIP: id={mid} not found')
            continue
        name = model.name
        uuid = model.uuid
        success, err = ModelService.delete_model(model)
        if success:
            edir = exp_dir / (uuid or '')
            if edir.exists():
                shutil.rmtree(edir)
            deleted += 1
            print(f'  OK: {name}')
        else:
            print(f'  FAIL: {name} - {err}')

with app.app_context():
    remaining = ModelRecord.query.count()
    perfect = ModelRecord.query.filter(ModelRecord.accuracy == 1.0).count()
    print(f'\nDone: {deleted} deleted, {remaining} remaining, {perfect} with acc=1.0')
