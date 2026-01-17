
import sqlite3
import os
import json

DB_PATH = os.path.join(os.getcwd(), "predictions.db")

conn = sqlite3.connect(DB_PATH)
c = conn.cursor()

c.execute("SELECT * FROM logs")
rows = c.fetchall()

if not rows:
    print("No logs found yet. Make sure you called the /analyze endpoint.")
else:
    for row in rows:
        print("ID:", row[0])
        print("Filename:", row[1])
        print("Document Type:", row[2])
        print("Extracted Fields:", json.loads(row[3]))
        print("Timestamp:", row[4])
        print("--------------------")
