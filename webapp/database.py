
import os
import sqlite3
from pathlib import Path
from urllib.parse import unquote, urlsplit

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DATABASE_URL = os.environ["DATABASE_URL"]


def get_database_path():
    """Resolve the configured SQLite database location."""
    parsed = urlsplit(DATABASE_URL)

    if parsed.scheme != "sqlite" or parsed.netloc:
        raise ValueError(
            "Task 5 SQLite mode requires a DATABASE_URL "
            "such as sqlite:///visionguard.db"
        )

    raw_path = unquote(parsed.path)

    if not raw_path or raw_path == "/":
        raise ValueError("SQLite database filename is missing.")

    # sqlite:///visionguard.db -> project_root/visionguard.db
    # sqlite:////absolute/path.db -> /absolute/path.db
    if raw_path.startswith("//"):
        path = Path(raw_path[1:])
    else:
        path = ROOT / raw_path.lstrip("/")

    return path.resolve()


DB_PATH = get_database_path()


def get_connection():
    """Create a SQLite connection with named columns."""
    connection = sqlite3.connect(DB_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    """Create the predictions table if it does not exist."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                label TEXT NOT NULL,
                confidence REAL NOT NULL,
                ai_probability REAL NOT NULL,
                model_version TEXT NOT NULL,
                created_at TEXT NOT NULL
                    DEFAULT (strftime(
                        '%Y-%m-%d %H:%M:%S',
                        'now'
                    ))
            )
        """)

    print(f"Database ready: {DB_PATH}")


def save_prediction(filename, result):
    """Permanently store one successful prediction."""
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO predictions (
                filename,
                label,
                confidence,
                ai_probability,
                model_version
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                filename,
                result["label"],
                result["confidence"],
                result["ai_probability"],
                result["model_version"]
            )
        )
        return cursor.lastrowid


def get_prediction_history():
    """Retrieve saved predictions, newest first."""
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                id,
                filename,
                label,
                confidence,
                ai_probability,
                model_version,
                created_at
            FROM predictions
            ORDER BY id DESC
            """
        ).fetchall()

    return [dict(row) for row in rows]
