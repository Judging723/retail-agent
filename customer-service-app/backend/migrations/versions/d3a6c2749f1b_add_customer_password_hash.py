"""add customer password hash

Revision ID: d3a6c2749f1b
Revises: c9b537cb2b9e
Create Date: 2026-11-01 10:00:00

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "d3a6c2749f1b"
down_revision: str | Sequence[str] | None = "c9b537cb2b9e"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "customers",
        sa.Column("password_hash", sa.String(length=255), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("customers", "password_hash")
