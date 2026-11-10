
import os
import sqlite3
from pathlib import Path
from urllib.parse import unquote, urlsplit

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DATABASE_URL = os.environ["DATABASE_URL"]
DATABASE_SCHEME = urlsplit(DATABASE_URL).scheme


def get_sqlite_path():
    parsed = urlsplit(DATABASE_URL)

    if parsed.scheme != "sqlite" or parsed.netloc:
        raise ValueError("Invalid SQLite database URL.")

    raw_path = unquote(parsed.path)

    if not raw_path or raw_path == "/":
        raise ValueError("SQLite database filename is missing.")

    if raw_path.startswith("//"):
        return Path(raw_path[1:]).resolve()

    return (ROOT / raw_path.lstrip("/")).resolve()


def get_connection():
    """Open a connection to the configured database."""
    if DATABASE_SCHEME == "sqlite":
        connection = sqlite3.connect(
            get_sqlite_path(),
            timeout=10
        )
        connection.row_factory = sqlite3.Row
        return connection

    if DATABASE_SCHEME in ("postgresql", "postgres"):
        import psycopg2
        return psycopg2.connect(DATABASE_URL)

    raise ValueError(
        f"Unsupported database scheme: {DATABASE_SCHEME}"
    )


def init_db():
    """Create the predictions table if necessary."""
    if DATABASE_SCHEME == "sqlite":
        get_sqlite_path().parent.mkdir(
            parents=True,
            exist_ok=True
        )

    connection = get_connection()

    try:
        cursor = connection.cursor()

        if DATABASE_SCHEME == "sqlite":
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    label TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    ai_probability REAL NOT NULL,
                    model_version TEXT NOT NULL,
                    created_at TEXT NOT NULL
                        DEFAULT (
                            strftime('%Y-%m-%d %H:%M:%S', 'now')
                        )
                )
            """)
        else:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id SERIAL PRIMARY KEY,
                    filename TEXT NOT NULL,
                    label TEXT NOT NULL,
                    confidence DOUBLE PRECISION NOT NULL,
                    ai_probability DOUBLE PRECISION NOT NULL,
                    model_version TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL
                        DEFAULT CURRENT_TIMESTAMP
                )
            """)

        connection.commit()
        cursor.close()

    finally:
        connection.close()

    print(f"Database initialized: {DATABASE_SCHEME}")


def save_prediction(filename, result):
    """Save one successful model prediction."""
    connection = get_connection()

    try:
        cursor = connection.cursor()

        values = (
            filename,
            result["label"],
            result["confidence"],
            result["ai_probability"],
            result["model_version"]
        )

        if DATABASE_SCHEME == "sqlite":
            cursor.execute("""
                INSERT INTO predictions (
                    filename, label, confidence,
                    ai_probability, model_version
                )
                VALUES (?, ?, ?, ?, ?)
            """, values)
            record_id = cursor.lastrowid
        else:
            cursor.execute("""
                INSERT INTO predictions (
                    filename, label, confidence,
                    ai_probability, model_version
                )
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
            """, values)
            record_id = cursor.fetchone()[0]

        connection.commit()
        cursor.close()
        return record_id

    finally:
        connection.close()


def get_prediction_history():
    """Return prediction records, newest first."""
    connection = get_connection()

    try:
        cursor = connection.cursor()

        cursor.execute("""
            SELECT
                id, filename, label, confidence,
                ai_probability, model_version, created_at
            FROM predictions
            ORDER BY id DESC
        """)

        columns = [
            column[0] for column in cursor.description
        ]

        records = [
            dict(zip(columns, row))
            for row in cursor.fetchall()
        ]

        cursor.close()
        return records

    finally:
        connection.close()
