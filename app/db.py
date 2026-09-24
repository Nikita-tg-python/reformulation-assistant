import logging
from pathlib import Path

import asyncpg
from pgvector.asyncpg import register_vector

logger = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"

# Arbitrary constant: serialises migrations if several app instances start at once.
_MIGRATION_LOCK_ID = 7_310_001


async def create_pool(database_url: str) -> asyncpg.Pool:
    """Apply migrations, then open the pool.

    Migrations run on a separate plain connection first: the pool's `init` registers the
    pgvector codec on every connection, which fails until the `vector` extension exists.
    """
    conn = await asyncpg.connect(dsn=database_url)
    try:
        await apply_migrations(conn)
    finally:
        await conn.close()
    return await asyncpg.create_pool(
        dsn=database_url, min_size=1, max_size=10, init=register_vector
    )


async def apply_migrations(conn: asyncpg.Connection) -> None:
    """Run every migrations/*.sql in name order. Each file must be idempotent."""
    files = sorted(MIGRATIONS_DIR.glob("*.sql"))
    async with conn.transaction():
        await conn.execute("SELECT pg_advisory_xact_lock($1)", _MIGRATION_LOCK_ID)
        for path in files:
            await conn.execute(path.read_text(encoding="utf-8"))
            logger.info("applied migration %s", path.name)


async def check_db(pool: asyncpg.Pool) -> bool:
    try:
        async with pool.acquire(timeout=2) as conn:
            await conn.fetchval("SELECT 1", timeout=2)
        return True
    except Exception:
        logger.exception("database health check failed")
        return False
