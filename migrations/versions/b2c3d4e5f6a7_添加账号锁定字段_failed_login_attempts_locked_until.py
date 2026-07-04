"""添加 users.failed_login_attempts + users.locked_until 账号锁定支持

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-06-30 18:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'b2c3d4e5f6a7'
down_revision = 'a1b2c3d4e5f6'
branch_labels = None
depends_on = None


def upgrade():
    # failed_login_attempts + locked_until — 账号锁定支持
    try:
        with op.batch_alter_table('users', schema=None) as batch_op:
            # 连续登录失败计数 (MAX_FAILED_ATTEMPTS=5)
            batch_op.add_column(sa.Column(
                'failed_login_attempts', sa.Integer(),
                nullable=False, server_default='0'
            ))
            # 账号锁定截止时间 (LOCKOUT_DURATION=15分钟)
            batch_op.add_column(sa.Column(
                'locked_until', sa.DateTime(),
                nullable=True
            ))
    except sa.exc.OperationalError:
        pass  # 列已存在 (幂等重跑保护)


def downgrade():
    with op.batch_alter_table('users', schema=None) as batch_op:
        batch_op.drop_column('locked_until')
        batch_op.drop_column('failed_login_attempts')
