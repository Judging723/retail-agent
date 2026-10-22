"""create tool audit logs

Revision ID: a73f0ef4829c
Revises: 8ea54f3435e6
Create Date: 2026-10-22 18:30:00

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a73f0ef4829c"
down_revision: str | Sequence[str] | None = "8ea54f3435e6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "tool_audit_logs",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("conversation_id", sa.BigInteger(), nullable=False),
        sa.Column("customer_id", sa.BigInteger(), nullable=False),
        sa.Column("tool_call_id", sa.String(length=128), nullable=False),
        sa.Column("tool_name", sa.String(length=128), nullable=False),
        sa.Column("arguments", sa.JSON(), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "success",
                "rejected",
                name="tool_audit_status",
                native_enum=False,
                create_constraint=True,
                length=16,
            ),
            nullable=False,
        ),
        sa.Column("result", sa.JSON(), nullable=True),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.id"],
            name=op.f("fk_tool_audit_logs_conversation_id_conversations"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["customer_id"],
            ["customers.id"],
            name=op.f("fk_tool_audit_logs_customer_id_customers"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_tool_audit_logs")),
    )
    op.create_index(
        op.f("ix_tool_audit_logs_conversation_id"),
        "tool_audit_logs",
        ["conversation_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_tool_audit_logs_customer_id"),
        "tool_audit_logs",
        ["customer_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_tool_audit_logs_customer_id"), table_name="tool_audit_logs")
    op.drop_index(op.f("ix_tool_audit_logs_conversation_id"), table_name="tool_audit_logs")
    op.drop_table("tool_audit_logs")
