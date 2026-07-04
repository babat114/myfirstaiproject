"""精确清理: 仅删除 unique_predictions==1 的真正常数预测器 (假阳性保留)"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import create_app, db
from app.models.model_record import ModelRecord
from app.models.training_job import TrainingJob
from sqlalchemy import text

REPORT = 'experiments/model_quality_report.json'

with open(REPORT, encoding='utf-8') as f:
    data = json.load(f)

# Collect model_ids where unique_predictions == 1 (true constant)
delete_ids = set()
keep_ids = set()
for cat_key in ['definite_delete', 'high_risk_delete']:
    for m in data.get(cat_key, []):
        cp = m.get('details', {}).get('constant_predictor', {})
        if cp.get('unique_predictions') == 1:
            delete_ids.add(m['model_id'])
        else:
            keep_ids.add(m['model_id'])

print(f'To DELETE: {len(delete_ids)} (unique_pred==1)')
print(f'To KEEP:   {len(keep_ids)} (false positives)')
print(f'IDs to delete: {sorted(delete_ids)}')

if not delete_ids:
    print('Nothing to delete.')
    sys.exit(0)

app = create_app()
with app.app_context():
    with db.session.no_autoflush:
        # Step 1: nullify FK references
        for mid in delete_ids:
            db.session.execute(
                text("UPDATE training_jobs SET model_id = NULL WHERE model_id = :mid"),
                {"mid": mid}
            )
        db.session.flush()

        # Step 2: delete models
        for mid in delete_ids:
            m = db.session.get(ModelRecord, mid)
            if m:
                print(f'  DELETE: id={mid} name={m.name[:50]}')
                db.session.delete(m)

        db.session.commit()

    remaining = ModelRecord.query.count()
    trained = ModelRecord.query.filter_by(status='trained').count()
    print(f'Done: {remaining} models, {trained} trained')
