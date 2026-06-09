import sqlite3

# Connect database
conn = sqlite3.connect("resume_data.db")

# Create cursor
cursor = conn.cursor()

# Create table
cursor.execute("""
CREATE TABLE IF NOT EXISTS matches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    resume_name TEXT,
    match_score REAL
)
""")

conn.commit()
conn.close()

print("✅ Database created successfully!")