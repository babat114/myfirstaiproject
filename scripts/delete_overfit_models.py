"""
Delete severely overfit models identified by independent test evaluation.
Keeps Wine Variety models (genuinely good with gap ~0%).
"""
import sys; sys.path.insert(0, '.')
from app import create_app, db
from app.models.model_record import ModelRecord
from app.services.model_service import ModelService

app = create_app()
with app.app_context():
    # Wine Variety models — genuinely generalize well (gap ~0%)
    KEEP_UUIDS = {
        'ae259d57-2b91-4d29-a6f2-69222d81ab96',  # Wine Variety RF
        '5d223b97-096a-41f9-a5b7-810563c41edd',  # Wine Variety KNN
    }

    t1 = ModelRecord.query.filter(
        ModelRecord.independent_accuracy.isnot(None)
    ).all()

    to_delete = []
    to_keep = []
    for m in t1:
        if m.accuracy and m.independent_accuracy:
            gap = m.accuracy - m.independent_accuracy
            if gap > 0.15 and m.uuid not in KEEP_UUIDS:
                to_delete.append(m)
            else:
                to_keep.append(m)

    print(f'Deleting {len(to_delete)} overfit models (gap > 15%):')
    for m in to_delete:
        gap = m.accuracy - m.independent_accuracy
        print(f'  [{m.id}] {m.name}: orig_acc={m.accuracy:.4f} ind_acc={m.independent_accuracy:.4f} gap={gap:.4f}')

    print(f'\nKeeping {len(to_keep)} models:')
    for m in to_keep:
        gap = m.accuracy - m.independent_accuracy if m.accuracy and m.independent_accuracy else 0
        print(f'  [{m.id}] {m.name}: gap={gap:.4f} ({m.model_type})')

    for m in to_delete:
        try:
            ModelService.delete_model(m)
            print(f'  [OK] {m.name}')
        except Exception as e:
            print(f'  [FAIL] {m.name}: {e}')

    db.session.commit()
    print(f'\nDone. Deleted {len(to_delete)}, kept {len(to_keep)}')
