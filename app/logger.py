import sqlite3
from datetime import datetime

DB_PATH = "logs.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            document_type TEXT,
            extracted_fields TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

def log_prediction(filename, document_type, fields):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions (filename, document_type, extracted_fields, timestamp)
        VALUES (?, ?, ?, ?)
    """, (filename, document_type, str(fields), datetime.now().isoformat()))
    conn.commit()
    conn.close()
