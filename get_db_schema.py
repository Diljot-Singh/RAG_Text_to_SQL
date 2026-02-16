# print_schema.py
from sqlalchemy import create_engine, inspect, text

DB_PATH = "db/fintech.db"  # adjust if needed

engine = create_engine(f"sqlite:///{DB_PATH}")
insp = inspect(engine)

print("=" * 80)
print("TABLE LIST")
print("=" * 80)

tables = insp.get_table_names()
for table in tables:
    print(table)

print("\n" + "=" * 80)
print("DETAILED SCHEMA")
print("=" * 80)

with engine.connect() as conn:
    for table in tables:
        print(f"\n--- TABLE: {table} ---")

        # Columns
        cols = insp.get_columns(table)
        print("Columns:")
        for col in cols:
            print(f"  - {col['name']} ({col['type']})")

        # Row count
        try:
            count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
            print(f"Row count: {count}")
        except:
            print("Row count: Could not compute")

        # Declared foreign keys (SQLite rarely has these, but check anyway)
        fks = insp.get_foreign_keys(table)
        if fks:
            print("Declared Foreign Keys:")
            for fk in fks:
                print(f"  - {fk}")
        else:
            print("Declared Foreign Keys: None")

print("\n" + "=" * 80)
print("END")
print("=" * 80)
