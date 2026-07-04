"""快速清理 draft 模型 — 解除 FK 引用后删除"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import create_app, db
from app.models.model_record import ModelRecord
from sqlalchemy import text

app = create_app()
with app.app_context():
    with db.session.no_autoflush:
        drafts = ModelRecord.query.filter(ModelRecord.status != 'trained').all()
        print(f'Cleaning {len(drafts)} draft models...')

        # Step 1: nullify FK references in training_jobs
        for m in drafts:
            db.session.execute(
                text("UPDATE training_jobs SET model_id = NULL WHERE model_id = :mid"),
                {"mid": m.id}
            )
        db.session.flush()

        # Step 2: delete draft models
        for m in drafts:
            db.session.delete(m)

        db.session.commit()
        remaining = ModelRecord.query.filter(ModelRecord.status != 'trained').count()
        total = ModelRecord.query.count()
        print(f'Done: {total} models, {remaining} drafts remaining')
