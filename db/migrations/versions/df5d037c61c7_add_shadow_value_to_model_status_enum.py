"""add shadow value to model_status enum

Revision ID: df5d037c61c7
Revises: 5cf139d7addf
Create Date: 2026-08-10 16:55:31.151583

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "df5d037c61c7"
down_revision: Union[str, None] = "5cf139d7addf"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TYPE model_status ADD VALUE IF NOT EXISTS 'shadow'")


def downgrade() -> None:
    # Postgres can't drop a single enum label directly; recreate the type
    # without it. Fails if any row still has status='shadow', which is the
    # correct behaviour (those rows must be reassigned before downgrading).
    # The production partial index's predicate is bound to the old type's
    # OID, so it has to be dropped and rebuilt around the type swap too.
    op.drop_index(
        "uq_models_single_production",
        table_name="models",
        postgresql_where=sa.text("status = 'production'"),
    )
    op.execute("ALTER TYPE model_status RENAME TO model_status_old")
    op.execute("CREATE TYPE model_status AS ENUM ('staging', 'production', 'archived')")
    op.execute(
        "ALTER TABLE models ALTER COLUMN status TYPE model_status "
        "USING status::text::model_status"
    )
    op.execute("DROP TYPE model_status_old")
    op.create_index(
        "uq_models_single_production",
        "models",
        ["status"],
        unique=True,
        postgresql_where=sa.text("status = 'production'"),
    )
