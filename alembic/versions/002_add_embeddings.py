"""add_embeddings_and_dedup_indexes

Revision ID: 002
Revises: 001
Create Date: 2026-06-27 10:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Ensure article_embeddings table and indexes are configured for deduplication
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = inspector.get_table_names()

    if "article_embeddings" not in tables:
        op.create_table(
            "article_embeddings",
            sa.Column("article_id", postgresql.UUID(as_uuid=True), nullable=False),
            sa.Column("embedding", postgresql.ARRAY(sa.Float()), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.ForeignKeyConstraint(["article_id"], ["articles.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("article_id"),
        )

    # Index for fast duplicate lookups on articles
    indexes = [idx["name"] for idx in inspector.get_indexes("articles")] if "articles" in tables else []
    if "ix_articles_is_duplicate" not in indexes:
        op.create_index("ix_articles_is_duplicate", "articles", ["is_duplicate"], unique=False)
    if "ix_articles_primary_article_id" not in indexes:
        op.create_index("ix_articles_primary_article_id", "articles", ["primary_article_id"], unique=False)


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    indexes = [idx["name"] for idx in inspector.get_indexes("articles")]

    if "ix_articles_primary_article_id" in indexes:
        op.drop_index("ix_articles_primary_article_id", table_name="articles")
    if "ix_articles_is_duplicate" in indexes:
        op.drop_index("ix_articles_is_duplicate", table_name="articles")
