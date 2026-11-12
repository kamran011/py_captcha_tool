"""
Database engine, session factory and schema initialisation.

SQLAlchemy 2.0 synchronous style is used throughout — this is a local
single-process tool so async is unnecessary overhead.
"""
from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings
from app.models import Base

logger = logging.getLogger(__name__)


def _make_engine():
    # Ensure the data directory exists for SQLite databases
    db_url = settings.database_url
    if db_url.startswith("sqlite:///"):
        db_path = Path(db_url[len("sqlite:///"):])
        db_path.parent.mkdir(parents=True, exist_ok=True)

    is_sqlite = db_url.startswith("sqlite")
    eng = create_engine(
        db_url,
        # timeout: seconds a writer waits on a locked database before erroring
        connect_args={"check_same_thread": False, "timeout": 15} if is_sqlite else {},
        echo=settings.debug,
    )
    if is_sqlite:
        @event.listens_for(eng, "connect")
        def _sqlite_pragmas(dbapi_conn, _record):  # noqa: ANN001
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")      # readers never block the writer
            cur.execute("PRAGMA busy_timeout=15000")    # queue writers instead of failing
            cur.execute("PRAGMA synchronous=NORMAL")    # safe with WAL, far fewer fsyncs
            cur.close()
    return eng


engine = _make_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db() -> None:
    """Create all tables that do not yet exist.  Safe to call multiple times."""
    Base.metadata.create_all(bind=engine)
    insp = inspect(engine)
    missing = {
        t.name: [c.name for c in t.columns if c.name not in {x["name"] for x in insp.get_columns(t.name)}]
        for t in Base.metadata.sorted_tables
    }
    missing = {t: cols for t, cols in missing.items() if cols}
    if missing:
        raise RuntimeError(
            f"Database schema is out of date (missing columns: {missing}). "
            "Delete the SQLite file and restart; it is recreated automatically."
        )
    # Verify connectivity
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    logger.info("Database initialised at %s", settings.database_url)


def get_db() -> Session:
    """
    Dependency-injection helper for FastAPI.

    Usage::

        @router.get("/endpoint")
        def endpoint(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
