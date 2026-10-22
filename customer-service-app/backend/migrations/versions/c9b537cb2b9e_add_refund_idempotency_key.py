"""add refund idempotency key

Revision ID: c9b537cb2b9e
Revises: a73f0ef4829c
Create Date: 2026-10-22 20:00:00

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c9b537cb2b9e"
down_revision: str | Sequence[str] | None = "a73f0ef4829c"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "refunds",
        sa.Column("idempotency_key", sa.String(length=128), nullable=True),
    )
    op.create_unique_constraint(
        op.f("uq_refunds_idempotency_key"),
        "refunds",
        ["idempotency_key"],
    )


def downgrade() -> None:
    op.drop_constraint(
        op.f("uq_refunds_idempotency_key"),
        "refunds",
        type_="unique",
    )
    op.drop_column("refunds", "idempotency_key")
