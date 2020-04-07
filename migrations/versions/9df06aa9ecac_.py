"""empty message

Revision ID: 9df06aa9ecac
Revises: 036aa307db24
Create Date: 2020-03-30 10:37:59.871322

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '9df06aa9ecac'
down_revision = '036aa307db24'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('message_data',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('message', sa.String(), nullable=True),
    sa.Column('prediction', sa.Integer(), nullable=True),
    sa.Column('created_on', sa.DateTime(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_message_data_created_on'), 'message_data', ['created_on'], unique=False)
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index(op.f('ix_message_data_created_on'), table_name='message_data')
    op.drop_table('message_data')
    # ### end Alembic commands ###
