import sqlite3
import pandas as pd

conn = sqlite3.connect("db/fintech.db")

query = """
SELECT *
FROM customer
LIMIT 10
"""

df = pd.read_sql_query(query, conn)

print(df.head())

conn.close()