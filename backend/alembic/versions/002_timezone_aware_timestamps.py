"""make timestamp columns timezone-aware

Revision ID: 002
Revises: 001
Create Date: 2026-08-17
"""
from collections.abc import Sequence

from alembic import op

revision: str = "002"
down_revision: str | None = "001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Columns currently TIMESTAMP WITHOUT TIME ZONE storing naive UTC values.
_COLUMNS = [
    ("users", ["created_at", "updated_at"]),
    ("inspections", ["created_at", "completed_at"]),
]


def upgrade() -> None:
    for table, columns in _COLUMNS:
        for column in columns:
            op.execute(
                f"ALTER TABLE {table} ALTER COLUMN {column} "
                "TYPE TIMESTAMPTZ USING " + column + " AT TIME ZONE 'UTC'"
            )


def downgrade() -> None:
    for table, columns in _COLUMNS:
        for column in columns:
            op.execute(
                f"ALTER TABLE {table} ALTER COLUMN {column} "
                "TYPE TIMESTAMP USING " + column + " AT TIME ZONE 'UTC'"
            )
