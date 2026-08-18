import ssl
import uuid

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from backend.config import settings
from backend.domain.models import Base

ssl_context = ssl.create_default_context()


def _prepared_statement_name() -> str:
    """Return a unique prepared statement name per call.

    asyncpg names prepared statements from a per-process counter that
    restarts at the same value on every boot. When a schema change happens
    (e.g. an Alembic migration) while pooled backend sessions still hold
    plans under those names, the new process collides with the stale plans
    and every query fails with InvalidCachedStatementError. Unique names
    sidestep that entirely.
    """
    return f"__fs_{uuid.uuid4().hex}__"


engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    connect_args={
        "ssl": ssl_context,
        "prepared_statement_name_func": _prepared_statement_name,
    },
)

async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_session() -> AsyncSession:
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db() -> None:
    await engine.dispose()


async def check_db_health() -> bool:
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
