"""upgraded link_hash length and added new indexes

Revision ID: 19afa3968cb2
Revises: 439b51d25c45
Create Date: 2025-06-22 19:16:37.345346

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '19afa3968cb2'
down_revision: Union[str, Sequence[str], None] = '439b51d25c45'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.alter_column('articles', 'link_hash',
               existing_type=sa.CHAR(length=16),
               type_=sa.CHAR(length=32),
               existing_nullable=False)
    op.drop_constraint(op.f('articles_link_key'), 'articles', type_='unique')
    op.create_index(op.f('ix_articles_link'), 'articles', ['link'], unique=True)
    # ### end Alembic commands ###


def downgrade() -> None:
    """Downgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index(op.f('ix_articles_link'), table_name='articles')
    op.create_unique_constraint(op.f('articles_link_key'), 'articles', ['link'], postgresql_nulls_not_distinct=False)
    op.alter_column('articles', 'link_hash',
               existing_type=sa.CHAR(length=32),
               type_=sa.CHAR(length=16),
               existing_nullable=False)
    # ### end Alembic commands ###
