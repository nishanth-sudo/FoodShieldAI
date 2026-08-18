"""initial schema

Revision ID: 001
Revises:
Create Date: 2026-07-27
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import ENUM, JSON

revision: str = "001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    ENUM(
        "consumer", "qa_inspector", "admin", "government",
        name="userrole",
    ).create(op.get_bind(), checkfirst=True)
    ENUM(
        "pending", "processing", "completed", "failed",
        name="inspectionstatus",
    ).create(op.get_bind(), checkfirst=True)

    op.create_table(
        "users",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("email", sa.String(255), nullable=False),
        sa.Column("hashed_password", sa.String(255), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("role", sa.Enum("consumer", "qa_inspector", "admin", "government",
                                  name="userrole", create_type=False), nullable=False),
        sa.Column("is_active", sa.Integer(), server_default=sa.text("1")),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    op.create_table(
        "inspections",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("image_url", sa.String(512), nullable=False),
        sa.Column("image_thumbnail_url", sa.String(512), nullable=True),
        sa.Column("status", sa.Enum("pending", "processing", "completed", "failed",
                                    name="inspectionstatus", create_type=False), nullable=False),
        sa.Column("food_type", sa.String(255), nullable=True),
        sa.Column("freshness_score", sa.Float(), nullable=True),
        sa.Column("shelf_life_days", sa.Integer(), nullable=True),
        sa.Column("packaging_defects", JSON(), nullable=True),
        sa.Column("contamination_risks", JSON(), nullable=True),
        sa.Column("ocr_data", JSON(), nullable=True),
        sa.Column("xai_heatmap_url", sa.String(512), nullable=True),
        sa.Column("confidence_scores", JSON(), nullable=True),
        sa.Column("report", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_inspections_user_id", "inspections", ["user_id"]
    )


def downgrade() -> None:
    op.drop_table("inspections")
    op.drop_table("users")
    op.execute("DROP TYPE IF EXISTS userrole")
    op.execute("DROP TYPE IF EXISTS inspectionstatus")
