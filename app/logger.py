import sqlite3
from datetime import datetime

DB_PATH = "logs.db"

# Initialize DB
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
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

# Log a prediction
def log_prediction(filename, document_type, extracted_fields):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO predictions (filename, document_type, extracted_fields, timestamp)
        VALUES (?, ?, ?, ?)
    """, (filename, document_type, str(extracted_fields), datetime.now().isoformat()))
    conn.commit()
    conn.close()
