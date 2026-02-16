import pandas as pd
import sqlite3
import os
import glob

DATA_PATH = "dataset"
DB_PATH = "db/fintech.db"

# Create DB folder if not exists
os.makedirs("db", exist_ok=True)

# Connect to SQLite
conn = sqlite3.connect(DB_PATH)

# Load all CSV files into separate tables
csv_files = glob.glob(os.path.join(DATA_PATH, "*.csv"))

for file in csv_files:
    table_name = os.path.splitext(os.path.basename(file))[0]
    
    print(f"Loading {table_name}...")
    
    df = pd.read_csv(file)
    df.to_sql(table_name, conn, if_exists="replace", index=False)

conn.close()

print("Database created successfully.")