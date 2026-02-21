import os
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from langfuse import get_client
import pandas as pd
from sqlalchemy import create_engine, text, inspect

load_dotenv()

# Optional Langfuse (safe: script runs even if not installed or keys missing)
LANGFUSE_ENABLED = bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

try:
    if LANGFUSE_ENABLED:
        # Langfuse SDK v3+ style (works with current docs)
        langfuse = get_client()
    else:
        langfuse = None
except Exception:
    LANGFUSE_ENABLED = False
    langfuse = None

# ----------------------------- 
# Paths / DB
# ----------------------------- 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "db", "fintech.db")
OUT_JSONL = os.path.join(BASE_DIR, "eval_gold.jsonl")
engine = create_engine(f"sqlite:///{DB_PATH}")
inspector = inspect(engine)
ALL_TABLES = inspector.get_table_names()

if not ALL_TABLES:
    raise RuntimeError("No tables found. Check DB_PATH.")

print("DB:", DB_PATH)
print("Tables:", ALL_TABLES)
print("Langfuse enabled:", LANGFUSE_ENABLED)

# ----------------------------- 
# Gold question set (starter)
# ----------------------------- 
# Keep these simple & robust:
# - counts/averages
# - group-by across known columns
# - joins only along your join rules (loan<->customer, loan<->state_region, customer<->state_region)

GOLD_SET: List[Dict[str, Any]] = [
    {
        "id": "Q01",
        "question": "Average annual_inc for customers with loans in good standing (Fully Paid or Current).",
        "gold_sql": """
            SELECT AVG(customer.annual_inc) AS avg_annual_inc
            FROM loan
            JOIN customer ON loan.customer_id = customer.customer_id
            WHERE loan.loan_status IN ('Fully Paid', 'Current');
        """.strip(),
        "tags": ["aggregate", "join", "enum"]
    },
    {
        "id": "Q02",
        "question": "Count of loans by loan_status.",
        "gold_sql": """
            SELECT loan.loan_status, COUNT(*) AS loan_count
            FROM loan
            GROUP BY loan.loan_status
            ORDER BY loan_count DESC;
        """.strip(),
        "tags": ["groupby", "enum"]
    },
    {
        "id": "Q03",
        "question": "Count of loans by purpose.",
        "gold_sql": """
            SELECT loan.purpose, COUNT(*) AS loan_count
            FROM loan
            GROUP BY loan.purpose
            ORDER BY loan_count DESC;
        """.strip(),
        "tags": ["groupby", "enum"]
    },
    {
        "id": "Q04",
        "question": "Count of loans by grade.",
        "gold_sql": """
            SELECT loan.grade, COUNT(*) AS loan_count
            FROM loan
            GROUP BY loan.grade
            ORDER BY loan_count DESC;
        """.strip(),
        "tags": ["groupby", "enum"]
    },
    {
        "id": "Q05",
        "question": "Count of loans by term.",
        "gold_sql": """
            SELECT loan.term, COUNT(*) AS loan_count
            FROM loan
            GROUP BY loan.term
            ORDER BY loan_count DESC;
        """.strip(),
        "tags": ["groupby", "enum"]
    },
    {
        "id": "Q06",
        "question": "Count of loans by issue_year.",
        "gold_sql": """
            SELECT loan.issue_year, COUNT(*) AS loan_count
            FROM loan
            GROUP BY loan.issue_year
            ORDER BY loan.issue_year;
        """.strip(),
        "tags": ["groupby", "time"]
    },
    {
        "id": "Q07",
        "question": "Count of loans by region (join state_region via loan.state).",
        "gold_sql": """
            SELECT state_region.region, COUNT(*) AS loan_count
            FROM loan
            JOIN state_region ON loan.state = state_region.state
            GROUP BY state_region.region
            ORDER BY loan_count DESC;
        """.strip(),
        "tags": ["groupby", "join", "geo"]
    },
    {
        "id": "Q08",
        "question": "Count of customers by region (join state_region via customer.addr_state).",
        "gold_sql": """
            SELECT state_region.region, COUNT(*) AS customer_count
            FROM customer
            JOIN state_region ON customer.addr_state = state_region.state
            GROUP BY state_region.region
            ORDER BY customer_count DESC;
        """.strip(),
        "tags": ["groupby", "join", "geo"]
    },
    {
        "id": "Q09",
        "question": "Average annual_inc by home_ownership.",
        "gold_sql": """
            SELECT customer.home_ownership, AVG(customer.annual_inc) AS avg_annual_inc
            FROM customer
            GROUP BY customer.home_ownership
            ORDER BY avg_annual_inc DESC;
        """.strip(),
        "tags": ["groupby", "enum"]
    },
    {
        "id": "Q10",
        "question": "Count of customers by verification_status.",
        "gold_sql": """
            SELECT customer.verification_status, COUNT(*) AS customer_count
            FROM customer
            GROUP BY customer.verification_status
            ORDER BY customer_count DESC;
        """.strip(),
        "tags": ["groupby", "enum"]
    },
    {
        "id": "Q11",
        "question": "Average annual_inc by verification_status.",
        "gold_sql": """
            SELECT customer.verification_status, AVG(customer.annual_inc) AS avg_annual_inc
            FROM customer
            GROUP BY customer.verification_status
            ORDER BY avg_annual_inc DESC;
        """.strip(),
        "tags": ["groupby", "enum"]
    },
    {
        "id": "Q12",
        "question": "Count of loans by subregion (join state_region).",
        "gold_sql": """
            SELECT state_region.subregion, COUNT(*) AS loan_count
            FROM loan
            JOIN state_region ON loan.state = state_region.state
            GROUP BY state_region.subregion
            ORDER BY loan_count DESC;
        """.strip(),
        "tags": ["groupby", "join", "geo"]
    },
    {
        "id": "Q13",
        "question": "Count of loans by state.",
        "gold_sql": """
            SELECT loan.state, COUNT(*) AS loan_count
            FROM loan
            GROUP BY loan.state
            ORDER BY loan_count DESC;
        """.strip(),
        "tags": ["groupby", "geo"]
    },
    {
        "id": "Q14",
        "question": "Count of loans in good standing (Fully Paid or Current) by region.",
        "gold_sql": """
            SELECT state_region.region, COUNT(*) AS loan_count
            FROM loan
            JOIN state_region ON loan.state = state_region.state
            WHERE loan.loan_status IN ('Fully Paid', 'Current')
            GROUP BY state_region.region
            ORDER BY loan_count DESC;
        """.strip(),
        "tags": ["groupby", "join", "enum"]
    },
    {
        "id": "Q15",
        "question": "Average annual_inc by region (customers joined to state_region).",
        "gold_sql": """
            SELECT state_region.region, AVG(customer.annual_inc) AS avg_annual_inc
            FROM customer
            JOIN state_region ON customer.addr_state = state_region.state
            GROUP BY state_region.region
            ORDER BY avg_annual_inc DESC;
        """.strip(),
        "tags": ["groupby", "join", "geo"]
    },
    {
        "id": "Q16",
        "question": "Count of loans by customer home_ownership (loan joined to customer).",
        "gold_sql": """
            SELECT customer.home_ownership, COUNT(*) AS loan_count
            FROM loan
            JOIN customer ON loan.customer_id = customer.customer_id
            GROUP BY customer.home_ownership
            ORDER BY loan_count DESC;
        """.strip(),
        "tags": ["groupby", "join", "enum"]
    },
    {
        "id": "Q17",
        "question": "Average annual_inc for customers with charged off loans.",
        "gold_sql": """
            SELECT AVG(customer.annual_inc) AS avg_annual_inc
            FROM loan
            JOIN customer ON loan.customer_id = customer.customer_id
            WHERE loan.loan_status = 'Charged Off';
        """.strip(),
        "tags": ["aggregate", "join", "enum"]
    },
    {
        "id": "Q18",
        "question": "Count of loans by type.",
        "gold_sql": """
            SELECT loan.type, COUNT(*) AS loan_count
            FROM loan
            GROUP BY loan.type
            ORDER BY loan_count DESC;
        """.strip(),
        "tags": ["groupby", "enum"]
    },
]

# ----------------------------- 
# Storage: eval_gold table
# ----------------------------- 

def ensure_eval_table() -> None:
    ddl = """
        CREATE TABLE IF NOT EXISTS eval_gold (
            id TEXT PRIMARY KEY,
            qid TEXT,
            question TEXT NOT NULL,
            gold_sql TEXT NOT NULL,
            gold_result_json TEXT,
            result_shape TEXT,
            status TEXT NOT NULL,
            error TEXT,
            tags_json TEXT,
            created_at TEXT NOT NULL
        );
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))


def df_to_compact_json(df: pd.DataFrame, max_rows: int = 200) -> str:
    """
    Store a compact, stable representation:
    - truncate rows
    - keep columns + rows as JSON (orient=split)
    """
    df2 = df.copy()
    if len(df2) > max_rows:
        df2 = df2.head(max_rows)
    
    payload = {
        "orient": "split",
        "columns": df2.columns.tolist(),
        "index": df2.index.tolist(),
        "data": df2.where(pd.notnull(df2), None).values.tolist(),
        "truncated": len(df) > max_rows,
        "row_count": int(len(df)),
        "col_count": int(df.shape[1]),
    }
    return json.dumps(payload, ensure_ascii=False)


def append_jsonl(record: Dict[str, Any]) -> None:
    with open(OUT_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ----------------------------- 
# Execution + Langfuse logging
# ----------------------------- 

def run_gold_query(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns a record with status + gold result (json) or error.
    """
    qid = item["id"]
    question = item["question"]
    gold_sql = item["gold_sql"]
    tags = item.get("tags", [])
    run_id = str(uuid.uuid4())
    created_at = datetime.utcnow().isoformat() + "Z"
    
    trace = None
    if LANGFUSE_ENABLED and langfuse is not None:
        # Create a trace span for the gold build (useful when later comparing predictions)
        trace = langfuse.start_span(
            name="gold_build_text_to_sql",
            input={"qid": qid, "sql": gold_sql},
            metadata={"qid": qid, "run_id": run_id, "phase": "gold_build", "tags": ["gold_build"] + tags},
        )
    
    try:
        span = None
        if trace is not None:
            span = langfuse.start_span(name="execute_gold_sql", input={"sql": gold_sql})
        
        df = pd.read_sql(gold_sql, engine)
        
        if span is not None:
            span.update(output={"shape": df.shape, "preview": df.head(5).to_dict(orient="records")})
            span.end()
        
        gold_result_json = df_to_compact_json(df)
        result_shape = json.dumps({"rows": int(df.shape[0]), "cols": int(df.shape[1])})
        status = "ok"
        error = ""
        
        if trace is not None:
            trace.update(output={"status": status, "shape": df.shape})
            trace.end()
    
    except Exception as e:
        gold_result_json = ""
        result_shape = ""
        status = "error"
        error = str(e)
        
        if trace is not None:
            # end span safely
            try:
                error_span = langfuse.start_span(
                    name="execute_gold_sql_error",
                    input={"sql": gold_sql, "error": error}
                )
                error_span.end()
                trace.update(output={"status": status, "error": error})
                trace.end()
            except Exception:
                pass
    
    record = {
        "id": run_id,
        "qid": qid,
        "question": question,
        "gold_sql": gold_sql,
        "gold_result_json": gold_result_json,
        "result_shape": result_shape,
        "status": status,
        "error": error,
        "tags_json": json.dumps(tags),
        "created_at": created_at,
    }
    return record


def upsert_eval_gold(record: Dict[str, Any]) -> None:
    sql = """
        INSERT INTO eval_gold 
        (id, qid, question, gold_sql, gold_result_json, result_shape, status, error, tags_json, created_at)
        VALUES (:id, :qid, :question, :gold_sql, :gold_result_json, :result_shape, :status, :error, :tags_json, :created_at);
    """
    with engine.begin() as conn:
        conn.execute(text(sql), record)


def main() -> None:
    ensure_eval_table()
    
    # Reset JSONL for clean runs (optional)
    if os.path.exists(OUT_JSONL):
        os.remove(OUT_JSONL)
    
    ok = 0
    err = 0
    
    for item in GOLD_SET:
        rec = run_gold_query(item)
        upsert_eval_gold(rec)
        append_jsonl(rec)
        
        if rec["status"] == "ok":
            ok += 1
            shape = json.loads(rec["result_shape"]) if rec["result_shape"] else {}
            print(f"[OK] {rec['qid']} rows={shape.get('rows')} cols={shape.get('cols')}")
        else:
            err += 1
            print(f"[ERR] {rec['qid']} -> {rec['error']}")
    
    print("\nDone.")
    print("Gold OK:", ok, "| Errors:", err)
    print("Wrote:", OUT_JSONL)
    print("Stored in sqlite table: eval_gold")
    
    # Ensure Langfuse flush in short-lived scripts
    if LANGFUSE_ENABLED and langfuse is not None:
        try:
            langfuse.flush()
        except Exception:
            pass


if __name__ == "__main__":
    main()