"""add partial unique index for shadow status

Revision ID: 01d6077d4632
Revises: df5d037c61c7
Create Date: 2026-08-10 16:55:31.301864

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "01d6077d4632"
down_revision: Union[str, None] = "df5d037c61c7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index(
        "uq_models_single_shadow",
        "models",
        ["status"],
        unique=True,
        postgresql_where=sa.text("status = 'shadow'"),
    )


def downgrade() -> None:
    op.drop_index(
        "uq_models_single_shadow",
        table_name="models",
        postgresql_where=sa.text("status = 'shadow'"),
    )
