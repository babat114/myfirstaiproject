"""添加 users.token_version + 索引优化

Revision ID: a1b2c3d4e5f6
Revises: 9d2f26e7bb46
Create Date: 2026-06-20 16:22:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'a1b2c3d4e5f6'
down_revision = '9d2f26e7bb46'
branch_labels = None
depends_on = None


def upgrade():
    # users.token_version — JWT 撤销支持
    with op.batch_alter_table('users', schema=None) as batch_op:
        try:
            batch_op.add_column(sa.Column('token_version', sa.Integer(), nullable=False, server_default='1'))
        except sa.exc.OperationalError:
            pass  # 列已存在 (开发环境已手动添加)
        batch_op.create_index(batch_op.f('ix_users_token_version'), ['token_version'], unique=False)

    # 索引优化 (if_not_exists 避免重复创建报错)
    with op.batch_alter_table('model_records', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_model_records_is_public'), ['is_public'], unique=False)
        batch_op.create_index(batch_op.f('ix_model_records_owner_id'), ['owner_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_model_records_training_dataset_id'), ['training_dataset_id'], unique=False)

    with op.batch_alter_table('training_jobs', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_training_jobs_is_public'), ['is_public'], unique=False)
        batch_op.create_index(batch_op.f('ix_training_jobs_owner_id'), ['owner_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_training_jobs_dataset_id'), ['dataset_id'], unique=False)

    with op.batch_alter_table('datasets', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_datasets_owner_id'), ['owner_id'], unique=False)


def downgrade():
    with op.batch_alter_table('datasets', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_datasets_owner_id'))

    with op.batch_alter_table('training_jobs', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_training_jobs_dataset_id'))
        batch_op.drop_index(batch_op.f('ix_training_jobs_owner_id'))
        batch_op.drop_index(batch_op.f('ix_training_jobs_is_public'))

    with op.batch_alter_table('model_records', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_model_records_training_dataset_id'))
        batch_op.drop_index(batch_op.f('ix_model_records_owner_id'))
        batch_op.drop_index(batch_op.f('ix_model_records_is_public'))

    with op.batch_alter_table('users', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_users_token_version'))
        batch_op.drop_column('token_version')
