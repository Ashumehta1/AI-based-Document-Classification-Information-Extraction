# import sqlite3
# from datetime import datetime

# DB_PATH = "logs.db"

# def init_db():
#     conn = sqlite3.connect(DB_PATH)
#     cursor = conn.cursor()
#     cursor.execute("""
#         CREATE TABLE IF NOT EXISTS predictions (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             filename TEXT,
#             document_type TEXT,
#             extracted_fields TEXT,
#             timestamp TEXT
#         )
#     """)
#     conn.commit()
#     conn.close()

# def log_prediction(filename, document_type, fields):
#     conn = sqlite3.connect(DB_PATH)
#     cursor = conn.cursor()
#     cursor.execute("""
#         INSERT INTO predictions (filename, document_type, extracted_fields, timestamp)
#         VALUES (?, ?, ?, ?)
#     """, (filename, document_type, str(fields), datetime.now().isoformat()))
#     conn.commit()
#     conn.close()

# app/logger.py
import sqlite3
import json
from datetime import datetime
import os

DB_PATH = os.path.join(os.getcwd(), "predictions.db")  # ensure correct absolute path

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
