import os
import json
import uuid
from datetime import datetime
from typing import Any, Dict, Tuple, List

import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# ----------------------------- 
# CONFIG
# ----------------------------- 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "db", "fintech.db")

# Change to your module filename (without .py) that contains run_text_to_sql(...)
PIPELINE_MODULE = "schema_extraction"

# Model runner parameters
K_RETRIEVAL = 4
MAX_RETRIES = 3

# Output
REPORT_CSV = os.path.join(BASE_DIR, "eval_report.csv")

# Numeric comparison tolerance
ABS_TOL = 1e-6
REL_TOL = 1e-6

# ----------------------------- 
# Load .env FIRST (for Langfuse + OpenAI + any other keys)
# ----------------------------- 
load_dotenv()

# ----------------------------- 
# Langfuse (optional)
# ----------------------------- 
LANGFUSE_ENABLED = bool(
    os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")
)

# Langfuse host/base URL (support both common env names)
LANGFUSE_BASE_URL = (
    os.getenv("LANGFUSE_BASE_URL")
    or os.getenv("LANGFUSE_HOST")
    or "https://cloud.langfuse.com"
)

try:
    if LANGFUSE_ENABLED:
        # Langfuse SDK uses env vars; base URL can be set via env as well.
        # Keeping this import inside try so script still runs without langfuse installed.
        from langfuse import get_client
        langfuse = get_client()
    else:
        langfuse = None
except Exception:
    LANGFUSE_ENABLED = False
    langfuse = None

engine = create_engine(f"sqlite:///{DB_PATH}")

# Import your pipeline
pipeline = __import__(PIPELINE_MODULE)
run_text_to_sql = getattr(pipeline, "run_text_to_sql")

# ----------------------------- 
# Helpers: JSON result encoding/decoding
# ----------------------------- 

def df_to_compact_json(df: pd.DataFrame, max_rows: int = 200) -> str:
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


def compact_json_to_df(s: str) -> pd.DataFrame:
    obj = json.loads(s)
    if obj.get("orient") != "split":
        raise ValueError("Unexpected gold/pred JSON format; expected orient=split")
    return pd.DataFrame(obj["data"], columns=obj["columns"])


# ----------------------------- 
# Helpers: canonicalize + compare (order-insensitive)
# ----------------------------- 

def _is_number(x: Any) -> bool:
    try:
        if x is None:
            return False
        float(x)
        return True
    except Exception:
        return False


def canonicalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Order-insensitive canonicalization:
    - sort columns by name
    - sort rows by all columns (stable string key)
    - reset index
    """
    cols = sorted(df.columns.tolist())
    df2 = df.copy()[cols]
    
    if df2.empty:
        return df2.reset_index(drop=True)
    
    tmp = df2.copy()
    for c in cols:
        tmp[c] = tmp[c].astype(str)
    
    tmp["_sort_key"] = tmp[cols].agg("||".join, axis=1)
    tmp = tmp.sort_values(by="_sort_key", kind="mergesort").drop(columns=["_sort_key"])
    return tmp.reset_index(drop=True)


def compare_dfs(gold: pd.DataFrame, pred: pd.DataFrame) -> Tuple[bool, str]:
    """
    Compare dataframes based on VALUES only, ignoring column names.
    Two queries are correct if they return the same data, regardless of how 
    the columns are named.
    
    Example: COUNT(*) as Count vs COUNT(*) as count_of_col should both pass 
    if the data is identical.
    """
    if gold is None or pred is None:
        return False, "One of the result sets is None."
    
    g = canonicalize_df(gold)
    p = canonicalize_df(pred)
    
    # Check if shapes match (number of rows and columns)
    if g.shape != p.shape:
        return False, f"Shape mismatch. gold={g.shape} pred={p.shape}"
    
    # Compare values only, ignoring column names
    # Reset both dataframes to numeric indices for positional comparison
    g_values = g.reset_index(drop=True).values
    p_values = p.reset_index(drop=True).values
    
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            gv = g_values[r, c]
            pv = p_values[r, c]
            
            if gv is None or (isinstance(gv, float) and pd.isna(gv)):
                gv = None
            if pv is None or (isinstance(pv, float) and pd.isna(pv)):
                pv = None
            
            if gv is None and pv is None:
                continue
            
            if (gv is None) != (pv is None):
                return False, f"Null mismatch at row={r} col={c}. gold={gv} pred={pv}"
            
            if _is_number(gv) and _is_number(pv):
                gfn = float(gv)
                pfn = float(pv)
                diff = abs(gfn - pfn)
                tol = max(ABS_TOL, REL_TOL * max(abs(gfn), abs(pfn)))
                if diff > tol:
                    return False, f"Numeric mismatch at row={r} col={c}. gold={gfn} pred={pfn} diff={diff} tol={tol}"
            else:
                if str(gv) != str(pv):
                    return False, f"Value mismatch at row={r} col={c}. gold={gv} pred={pv}"
    
    return True, ""


# ----------------------------- 
# DB storage: eval_pred
# ----------------------------- 

def ensure_eval_pred_table() -> None:
    ddl = """
        CREATE TABLE IF NOT EXISTS eval_pred (
            id TEXT PRIMARY KEY,
            qid TEXT NOT NULL,
            question TEXT NOT NULL,
            pred_sql TEXT,
            pred_result_json TEXT,
            result_shape TEXT,
            status TEXT NOT NULL,
            error TEXT,
            attempts INTEGER,
            tables_json TEXT,
            joins_json TEXT,
            passed INTEGER,
            compare_error TEXT,
            created_at TEXT NOT NULL
        );
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))


def insert_eval_pred(rec: Dict[str, Any]) -> None:
    sql = """
        INSERT INTO eval_pred 
        (id, qid, question, pred_sql, pred_result_json, result_shape, 
         status, error, attempts, tables_json, joins_json, passed, 
         compare_error, created_at)
        VALUES (:id, :qid, :question, :pred_sql, :pred_result_json, 
                :result_shape, :status, :error, :attempts, :tables_json, 
                :joins_json, :passed, :compare_error, :created_at);
    """
    with engine.begin() as conn:
        conn.execute(text(sql), rec)


def load_gold_items() -> List[Dict[str, Any]]:
    q = """
        SELECT qid, question, gold_sql, gold_result_json, status, error, tags_json
        FROM eval_gold
        ORDER BY qid;
    """
    df = pd.read_sql(q, engine)
    return df.to_dict(orient="records")


# ----------------------------- 
# Langfuse helpers
# ----------------------------- 

def start_langfuse_span(
    qid: str,
    question: str,
    tags: List[str],
    metadata: Dict[str, Any]
):
    """
    Returns a context manager if langfuse enabled; else None.
    """
    if not LANGFUSE_ENABLED or langfuse is None:
        return None
    
    # base url typically comes from env; keeping metadata for visibility
    metadata = {**metadata, "langfuse_base_url": LANGFUSE_BASE_URL, "tags": tags}
    
    return langfuse.start_as_current_observation(
        as_type="span",
        name="eval_item",
        input={"qid": qid, "question": question},
        metadata=metadata,
    )


# ----------------------------- 
# Main evaluation loop
# ----------------------------- 

def main():
    ensure_eval_pred_table()
    gold_items = load_gold_items()
    
    rows_for_report = []
    total = 0
    ok_exec = 0
    ok_correct = 0
    skipped = 0
    
    for item in gold_items:
        total += 1
        qid = item["qid"]
        question = item["question"]
        gold_status = item["status"]
        tags = json.loads(item["tags_json"] or "[]")
        
        if gold_status != "ok" or not item.get("gold_result_json"):
            skipped += 1
            rows_for_report.append({
                "qid": qid,
                "question": question,
                "gold_status": gold_status,
                "pred_status": "skipped",
                "passed": 0,
                "attempts": None,
                "error": f"Gold not usable: {item.get('error','')}",
                "compare_error": "",
                "pred_sql": "",
            })
            continue
        
        gold_df = compact_json_to_df(item["gold_result_json"])
        run_id = str(uuid.uuid4())
        created_at = datetime.utcnow().isoformat() + "Z"
        
        span_ctx = start_langfuse_span(
            qid=qid,
            question=question,
            tags=["eval_pred"] + tags,
            metadata={"phase": "prediction_eval", "run_id": run_id},
        )
        
        if span_ctx is not None:
            span = span_ctx.__enter__()
        else:
            span = None
        
        # --- Run model pipeline
        try:
            child = None
            if span is not None:
                child = span.start_span(
                    name="run_text_to_sql",
                    input={"k": K_RETRIEVAL, "max_retries": MAX_RETRIES}
                )
            
            out = run_text_to_sql(question, k=K_RETRIEVAL, max_retries=MAX_RETRIES)
            
            if child is not None:
                # Log planner agent output
                planner_output = out.get("plan", {})
                # Log SQL generation details
                child.update(output={
                    "planner_output": planner_output,  # What the planner agent decided
                    "tables": out.get("tables"),
                    "joins": out.get("join_clauses"),
                    "sql_query": out.get("sql"),  # What SQL was generated
                    "attempts": out.get("attempts"),
                    "error": out.get("error"),
                })
                child.end()
            
            pred_status = "ok" if (out.get("df") is not None and not out.get("error")) else "error"
            pred_error = out.get("error", "") or ""
            attempts = int(out.get("attempts", 0) or 0)
            pred_df = out.get("df") if pred_status == "ok" else None
        
        except Exception as e:
            out = {}
            pred_status = "error"
            pred_error = str(e)
            attempts = None
            pred_df = None
        
        # --- Compare to gold
        passed = 0
        compare_error = ""
        pred_result_json = ""
        result_shape = ""
        
        if pred_status == "ok" and pred_df is not None:
            ok_exec += 1
            pred_result_json = df_to_compact_json(pred_df)
            result_shape = json.dumps({"rows": int(pred_df.shape[0]), "cols": int(pred_df.shape[1])})
            passed_bool, compare_error = compare_dfs(gold_df, pred_df)
            passed = 1 if passed_bool else 0
            if passed:
                ok_correct += 1
        
        # --- Store in eval_pred
        rec = {
            "id": run_id,
            "qid": qid,
            "question": question,
            "pred_sql": (out.get("sql") or ""),
            "pred_result_json": pred_result_json,
            "result_shape": result_shape,
            "status": pred_status,
            "error": pred_error,
            "attempts": attempts,
            "tables_json": json.dumps(out.get("tables") or []),
            "joins_json": json.dumps(out.get("join_clauses") or []),
            "passed": int(passed),
            "compare_error": compare_error,
            "created_at": created_at,
        }
        insert_eval_pred(rec)
        
        # --- Langfuse scoring
        if span is not None:
            span.score_trace(
                name="exec_success",
                value=float(1.0 if pred_status == "ok" and pred_df is not None else 0.0),
                data_type="NUMERIC",
            )
            span.score_trace(
                name="sql_correctness",
                value=float(passed),
                data_type="NUMERIC",
                comment=compare_error or ("match" if passed else "mismatch"),
            )
            span.update(output={
                "pred_status": pred_status,
                "attempts": attempts,
                "pred_sql": (out.get("sql") or ""),
                "pred_result_shape": result_shape,
                "passed": passed,
                "correctness_verdict": "CORRECT ✓" if passed else "INCORRECT ✗",
                "compare_error": compare_error,
                "pred_error": pred_error,
            })
        
        rows_for_report.append({
            "qid": qid,
            "question": question,
            "gold_status": gold_status,
            "pred_status": pred_status,
            "passed": passed,
            "attempts": attempts,
            "error": pred_error,
            "compare_error": compare_error,
            "pred_sql": (out.get("sql") or ""),
        })
        
        if span_ctx is not None:
            span_ctx.__exit__(None, None, None)
        
        print(f"[{qid}] pred={pred_status} passed={passed} attempts={attempts} err={pred_error[:120]}")
    
    rep = pd.DataFrame(rows_for_report)
    rep.to_csv(REPORT_CSV, index=False)
    
    denom = max(1, (total - skipped))
    exec_rate = ok_exec / denom
    corr_rate = ok_correct / denom
    
    print("\n=== Summary ===")
    print("Total items:", total)
    print("Skipped (bad gold):", skipped)
    print("Executed OK:", ok_exec)
    print("Correct:", ok_correct)
    print("Execution success rate:", round(exec_rate, 4))
    print("Correctness rate:", round(corr_rate, 4))
    print("Report:", REPORT_CSV)
    
    if LANGFUSE_ENABLED and langfuse is not None:
        try:
            langfuse.flush()
        except Exception:
            pass


if __name__ == "__main__":
    main()