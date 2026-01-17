
import sqlite3
import json
from datetime import datetime
import os

DB_PATH = os.path.join(os.getcwd(), "predictions.db")  
def init_db():
    """Initialize the database and create table if not exists."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            document_type TEXT,
            extracted_fields TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def log_prediction(filename, document_type, extracted_fields):
    """Log a prediction to the database."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO logs (filename, document_type, extracted_fields, timestamp) VALUES (?, ?, ?, ?)",
        (filename, document_type, json.dumps(extracted_fields), datetime.now())
    )
    conn.commit()
    conn.close()
