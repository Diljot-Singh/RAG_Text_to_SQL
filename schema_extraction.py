import os
import json
import re
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from collections import deque, defaultdict
from datetime import datetime

import pandas as pd
import sqlglot
from sqlalchemy import inspect, create_engine, text
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.document import Document


# ============================================================
# ENV + DATABASE SETUP
# ============================================================

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "db", "fintech.db")
PROFILE_PATH = os.path.join(BASE_DIR, "enum_profile.json")

print("DB:", DB_PATH, "exists:", os.path.exists(DB_PATH))
engine = create_engine(f"sqlite:///{DB_PATH}")

inspector = inspect(engine)
ALL_TABLES = inspector.get_table_names()
print("Tables:", ALL_TABLES)

if not ALL_TABLES:
    raise RuntimeError("No tables found in DB.")

print("OpenAI key loaded:", bool(os.getenv("OPENAI_API_KEY")))

# ============================================================
# JOIN RULES + JOIN ADJACENCY
# ============================================================

join_rules: List[Dict[str, str]] = [
    {"left_table": "loan", "left_key": "customer_id", "right_table": "customer", "right_key": "customer_id"},
    {"left_table": "loan", "left_key": "state", "right_table": "state_region", "right_key": "state"},
    {"left_table": "customer", "left_key": "addr_state", "right_table": "state_region", "right_key": "state"},
    {"left_table": "loan", "left_key": "purpose", "right_table": "loan_purposes", "right_key": "purpose"},
    {"left_table": "loan", "left_key": "issue_year", "right_table": "loan_count_by_year", "right_key": "issue_year"},
    {"left_table": "loan_with_region", "left_key": "loan_id", "right_table": "loan", "right_key": "loan_id"},
]

def build_join_adjacency(rules: List[Dict[str, str]]) -> Dict[str, List[str]]:
    adj: Dict[str, List[str]] = {}
    for r in rules:
        a, b = r["left_table"], r["right_table"]
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    return adj

join_adj = build_join_adjacency(join_rules)

def shortest_path(graph: Dict[str, List[str]], start: str, goal: str) -> Optional[List[str]]:
    if start == goal:
        return [start]
    q = deque([(start, [start])])
    seen = {start}
    while q:
        node, path = q.popleft()
        for nxt in graph.get(node, []):
            if nxt in seen:
                continue
            if nxt == goal:
                return path + [nxt]
            seen.add(nxt)
            q.append((nxt, path + [nxt]))
    return None

def join_clause_for_edge(a: str, b: str) -> Optional[str]:
    for r in join_rules:
        if r["left_table"] == a and r["right_table"] == b:
            return f"JOIN {b} ON {a}.{r['left_key']} = {b}.{r['right_key']}"
        if r["left_table"] == b and r["right_table"] == a:
            # mirrors your original logic
            return f"JOIN {b} ON {b}.{r['left_key']} = {a}.{r['right_key']}"
    return None

def compute_join_clauses(picked_tables: List[str]) -> List[str]:
    """
    Connect every table to base table using BFS path over join_adj, then
    convert edges into JOIN clauses using join_rules.
    """
    if len(picked_tables) <= 1:
        return []

    base = picked_tables[0]
    clauses: List[str] = []
    for t in picked_tables[1:]:
        path = shortest_path(join_adj, base, t)
        if not path:
            continue
        for i in range(len(path) - 1):
            c = join_clause_for_edge(path[i], path[i + 1])
            if c:
                clauses.append(c)

    return list(dict.fromkeys(clauses))  # dedupe keep order

def validate_joins_against_rules(sql: str) -> tuple[bool, str]:
    """
    Extract ON equality patterns like: table1.col1 = table2.col2
    Ensure each equality matches an allowed join rule (either direction).
    """
    allowed = set()
    for r in join_rules:
        allowed.add((r["left_table"].lower(), r["left_key"].lower(), r["right_table"].lower(), r["right_key"].lower()))
        allowed.add((r["right_table"].lower(), r["right_key"].lower(), r["left_table"].lower(), r["left_key"].lower()))

    pairs = re.findall(r"(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)", sql.lower())
    for t1, c1, t2, c2 in pairs:
        if (t1, c1, t2, c2) not in allowed:
            return False, f"Illegal join detected: {t1}.{c1} = {t2}.{c2}"
    return True, ""

def expand_with_bridge_tables(selected_tables: List[str]) -> List[str]:
    """
    If plan says ["customer","state_region"] but the join path requires a bridge,
    include it automatically. Keeps base table first.
    """
    if not selected_tables:
        return selected_tables

    base = selected_tables[0]
    expanded: List[str] = [base]

    for t in selected_tables[1:]:
        path = shortest_path(join_adj, base, t)
        if not path:
            if t not in expanded:
                expanded.append(t)
            continue
        for node in path[1:]:
            if node not in expanded:
                expanded.append(node)
    return expanded


# ============================================================
# VALUE GROUNDING (DISCOVERY + PROFILE CACHES)
# ============================================================

LOW_CARDINALITY_MAX = 50          # include all values in prompt when <= 50
MAX_VALUES_IN_PROMPT = 50         # cap values shown even for low-card
ENUM_DOC_CHUNK_SIZE = 15          # enum store doc chunk size (values per doc)
ENUM_RETRIEVE_K = 3               # small k to avoid drowning schema retrieval

# Caches (populated from profile file or computed once at startup)
CARDINALITY_CACHE: Dict[Tuple[str, str], int] = {}
ROWCOUNT_CACHE: Dict[str, int] = {}
LOW_CARD_VALUES: Dict[Tuple[str, str], List[str]] = {}
LOW_CARD_VALUE_SET: Dict[Tuple[str, str], set] = {}
TOP_VALUES_CACHE: Dict[Tuple[str, str], List[str]] = {}
CATEGORICAL_COLUMNS: List[Tuple[str, str]] = []  # discovered candidates (low + high)

# Heuristics to skip obvious high-card/ID-like columns
_SKIP_COLNAME_PATTERNS = [
    r"^id$",
    r"_id$",
    r"uuid",
    r"guid",
    r"email",
    r"phone",
    r"timestamp",
    r"created",
    r"updated",
    r"date",
    r"time",
    r"zip",
    r"postal",
    r"address",
]
_SKIP_COLNAME_RE = re.compile("|".join(_SKIP_COLNAME_PATTERNS), re.IGNORECASE)

def _schema_fingerprint() -> str:
    """
    Stable-ish fingerprint to detect schema changes and recompute the profile.
    """
    parts: List[str] = []
    for t in sorted(ALL_TABLES):
        cols = inspector.get_columns(t)
        parts.append(t + ":" + ",".join(sorted([f"{c['name']}:{str(c['type'])}" for c in cols])))
    blob = "\n".join(parts).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]

SCHEMA_FINGERPRINT = _schema_fingerprint()

def _is_textish(sqlalchemy_type: Any) -> bool:
    s = str(sqlalchemy_type).lower()
    return any(x in s for x in ["char", "text", "clob", "varchar", "string"])

def _is_intish(sqlalchemy_type: Any) -> bool:
    s = str(sqlalchemy_type).lower()
    return any(x in s for x in ["int", "integer"])

def table_row_count(table: str) -> int:
    if table in ROWCOUNT_CACHE:
        return ROWCOUNT_CACHE[table]
    q = f"SELECT COUNT(*) AS c FROM {table};"
    try:
        df = pd.read_sql(q, engine)
        c = int(df["c"].iloc[0])
    except Exception:
        c = 0
    ROWCOUNT_CACHE[table] = c
    return c

def distinct_count(table: str, column: str) -> int:
    key = (table, column)
    if key in CARDINALITY_CACHE:
        return CARDINALITY_CACHE[key]
    q = f"SELECT COUNT(DISTINCT {column}) AS c FROM {table};"
    try:
        df = pd.read_sql(q, engine)
        c = int(df["c"].iloc[0])
    except Exception:
        c = 10**9
    CARDINALITY_CACHE[key] = c
    return c

def get_distinct_values(table: str, column: str, limit: int = MAX_VALUES_IN_PROMPT) -> List[str]:
    q = f"SELECT DISTINCT {column} AS v FROM {table} WHERE {column} IS NOT NULL LIMIT {limit};"
    try:
        df = pd.read_sql(q, engine)
        return [str(x) for x in df["v"].tolist() if str(x).strip() != ""]
    except Exception:
        return []

def top_values(table: str, column: str, limit: int = 10) -> List[str]:
    q = f"""
    SELECT {column} AS v, COUNT(*) AS cnt
    FROM {table}
    WHERE {column} IS NOT NULL
    GROUP BY {column}
    ORDER BY cnt DESC
    LIMIT {limit};
    """
    try:
        df = pd.read_sql(q, engine)
        return df["v"].astype(str).tolist()
    except Exception:
        return []

def value_exists_high_card(table: str, column: str, value: str) -> bool:
    q = f"SELECT 1 FROM {table} WHERE {column} = :val LIMIT 1;"
    try:
        with engine.connect() as conn:
            row = conn.execute(text(q), {"val": value}).fetchone()
        return row is not None
    except Exception:
        return False

def suggest_values_like(table: str, column: str, value: str, limit: int = 10) -> List[str]:
    q = f"""
    SELECT {column} AS v, COUNT(*) AS cnt
    FROM {table}
    WHERE {column} IS NOT NULL AND {column} LIKE :pat
    GROUP BY {column}
    ORDER BY cnt DESC
    LIMIT {limit};
    """
    try:
        pat = f"%{value}%"
        df = pd.read_sql(q, engine, params={"pat": pat})
        return df["v"].astype(str).tolist()
    except Exception:
        return []

def _discover_categorical_candidates() -> List[Tuple[str, str]]:
    """
    Simple discovery: scan all tables/columns, keep text-ish columns + some small ints.
    Skip obvious IDs/time/etc by column name pattern.
    """
    candidates: List[Tuple[str, str]] = []
    for t in ALL_TABLES:
        for c in inspector.get_columns(t):
            col_name = c["name"]
            col_type = c["type"]
            if _SKIP_COLNAME_RE.search(col_name):
                continue
            if _is_textish(col_type):
                candidates.append((t, col_name))
            elif _is_intish(col_type):
                # keep some int columns that might be enum-like by name
                if re.search(r"(status|grade|term|type|region|state|purpose|plan)$", col_name, re.IGNORECASE):
                    candidates.append((t, col_name))
    # dedupe while preserving order
    return list(dict.fromkeys(candidates))

def _build_or_load_profile() -> None:
    """
    Builds/loads:
    - CATEGORICAL_COLUMNS (discovered candidates)
    - CARDINALITY_CACHE (distinct counts for those candidates)
    - LOW_CARD_VALUES + LOW_CARD_VALUE_SET (allowed values for low-card columns)
    - TOP_VALUES_CACHE (top values for high-card columns; used for debugging/suggestions)
    - ROWCOUNT_CACHE (row counts per table; optional but helpful)
    """
    global CATEGORICAL_COLUMNS

    if os.path.exists(PROFILE_PATH):
        try:
            with open(PROFILE_PATH, "r", encoding="utf-8") as f:
                prof = json.load(f)
            if prof.get("schema_fingerprint") == SCHEMA_FINGERPRINT:
                # Load caches
                ROWCOUNT_CACHE.update({k: int(v) for k, v in prof.get("rowcounts", {}).items()})
                for k, v in prof.get("distinct_counts", {}).items():
                    t, c = k.split(".", 1)
                    CARDINALITY_CACHE[(t, c)] = int(v)
                for k, vals in prof.get("low_card_values", {}).items():
                    t, c = k.split(".", 1)
                    vals = [str(x) for x in vals]
                    LOW_CARD_VALUES[(t, c)] = vals
                    LOW_CARD_VALUE_SET[(t, c)] = set(vals)
                for k, vals in prof.get("top_values", {}).items():
                    t, c = k.split(".", 1)
                    TOP_VALUES_CACHE[(t, c)] = [str(x) for x in vals]
                CATEGORICAL_COLUMNS = [tuple(x) for x in prof.get("categorical_columns", [])]
                print(f"✅ Loaded enum profile from {PROFILE_PATH} (schema match).")
                print(f"   categorical candidates: {len(CATEGORICAL_COLUMNS)} | low-card cols: {len(LOW_CARD_VALUES)}")
                return
            else:
                print("ℹ️ enum_profile.json found but schema fingerprint changed. Recomputing profile...")
        except Exception as e:
            print(f"⚠️ Failed to load profile ({e}). Recomputing profile...")

    # Compute profile fresh
    print("🔎 Building enum profile (startup, one-time per run)...")
    CATEGORICAL_COLUMNS = _discover_categorical_candidates()

    # Precompute row counts (cheap)
    for t in ALL_TABLES:
        table_row_count(t)

    # Compute distinct counts and values
    for (t, c) in CATEGORICAL_COLUMNS:
        dc = distinct_count(t, c)
        if dc <= LOW_CARDINALITY_MAX:
            vals = get_distinct_values(t, c, limit=MAX_VALUES_IN_PROMPT)
            if vals:
                LOW_CARD_VALUES[(t, c)] = vals
                LOW_CARD_VALUE_SET[(t, c)] = set(vals)
        else:
            TOP_VALUES_CACHE[(t, c)] = top_values(t, c, limit=10)

    # Persist profile
    prof = {
        "schema_fingerprint": SCHEMA_FINGERPRINT,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "categorical_columns": [[t, c] for (t, c) in CATEGORICAL_COLUMNS],
        "rowcounts": {t: ROWCOUNT_CACHE.get(t, 0) for t in ALL_TABLES},
        "distinct_counts": {f"{t}.{c}": CARDINALITY_CACHE[(t, c)] for (t, c) in CATEGORICAL_COLUMNS if (t, c) in CARDINALITY_CACHE},
        "low_card_values": {f"{t}.{c}": LOW_CARD_VALUES[(t, c)] for (t, c) in LOW_CARD_VALUES},
        "top_values": {f"{t}.{c}": TOP_VALUES_CACHE[(t, c)] for (t, c) in TOP_VALUES_CACHE},
    }
    try:
        with open(PROFILE_PATH, "w", encoding="utf-8") as f:
            json.dump(prof, f, indent=2, ensure_ascii=False)
        print(f"✅ Wrote enum profile to {PROFILE_PATH}")
        print(f"   categorical candidates: {len(CATEGORICAL_COLUMNS)} | low-card cols: {len(LOW_CARD_VALUES)}")
    except Exception as e:
        print(f"⚠️ Failed to write profile ({e}). Continuing with in-memory caches.")

_build_or_load_profile()

def build_value_hints_for_tables(picked_tables: List[str]) -> str:
    """
    Deterministic, filtered value hints:
    - Only low-card columns (from profile)
    - Only for tables selected for this question
    """
    picked = set(picked_tables)
    lines: List[str] = []
    for (t, c), vals in LOW_CARD_VALUES.items():
        if t in picked and vals:
            lines.append(f"Allowed values for {t}.{c}: {vals}")
    return "\n".join(lines)

def _parse_equality_filters(sql: str) -> List[Tuple[str, str, str]]:
    """
    Returns list of (table, column, value) for patterns like:
    - table.col = 'value'
    Supports single-quoted strings only.
    """
    pattern = r"(\b\w+\b)\.(\b\w+\b)\s*=\s*'([^']*)'"
    return re.findall(pattern, sql)

def _parse_in_filters(sql: str) -> List[Tuple[str, str, List[str]]]:
    """
    Returns list of (table, column, [values]) for patterns like:
    - table.col IN ('a','b','c')
    Supports single-quoted strings only.
    """
    # naive but works for simple IN lists
    pattern = r"(\b\w+\b)\.(\b\w+\b)\s+in\s*\(\s*([^\)]*?)\s*\)"
    matches = re.findall(pattern, sql, flags=re.IGNORECASE)
    out: List[Tuple[str, str, List[str]]] = []
    for table, col, inner in matches:
        vals = re.findall(r"'([^']*)'", inner)
        out.append((table, col, vals))
    return out

def validate_filter_values(sql: str, allowed_tables: List[str]) -> Tuple[bool, str]:
    """
    Validates categorical filter values.
    - Low-card columns: validate purely in-memory against LOW_CARD_VALUE_SET (no DB hits)
    - High-card columns: DB existence check + suggestions
    Only validates columns discovered in CATEGORICAL_COLUMNS and within allowed_tables.
    """
    allowed_set = set(allowed_tables)
    candidate_set = set(CATEGORICAL_COLUMNS)

    # 1) Equality filters
    for table, col, val in _parse_equality_filters(sql):
        if table not in allowed_set:
            continue
        if (table, col) not in candidate_set:
            continue

        key = (table, col)
        dc = CARDINALITY_CACHE.get(key, 10**9)

        # Low-card: validate in-memory only
        if dc <= LOW_CARDINALITY_MAX and key in LOW_CARD_VALUE_SET:
            if val in LOW_CARD_VALUE_SET[key]:
                continue
            return False, (
                f"Invalid filter value: {table}.{col} = '{val}' not found.\n"
                f"Allowed values for {table}.{col}: {LOW_CARD_VALUES.get(key, [])}"
            )

        # High-card: existence query + suggestions
        if value_exists_high_card(table, col, val):
            continue

        like = suggest_values_like(table, col, val, limit=10)
        top = TOP_VALUES_CACHE.get(key) or top_values(table, col, limit=10)
        return False, (
            f"Invalid filter value: {table}.{col} = '{val}' not found.\n"
            f"Suggestions containing '{val}': {like}\n"
            f"Most common values for {table}.{col}: {top}\n"
            f"Tip: pick one of the above and rerun."
        )

    # 2) IN (...) filters
    for table, col, vals in _parse_in_filters(sql):
        if table not in allowed_set:
            continue
        if (table, col) not in candidate_set:
            continue

        key = (table, col)
        dc = CARDINALITY_CACHE.get(key, 10**9)

        if dc <= LOW_CARDINALITY_MAX and key in LOW_CARD_VALUE_SET:
            missing = [v for v in vals if v not in LOW_CARD_VALUE_SET[key]]
            if not missing:
                continue
            return False, (
                f"Invalid filter values: {table}.{col} IN {missing} not found.\n"
                f"Allowed values for {table}.{col}: {LOW_CARD_VALUES.get(key, [])}"
            )

        # High-card: check each (limit checks to keep cost sane)
        # If many IN values, we only verify a few; self-heal can correct later.
        for v in vals[:10]:
            if not value_exists_high_card(table, col, v):
                like = suggest_values_like(table, col, v, limit=10)
                top = TOP_VALUES_CACHE.get(key) or top_values(table, col, limit=10)
                return False, (
                    f"Invalid filter value: {table}.{col} = '{v}' not found.\n"
                    f"Suggestions containing '{v}': {like}\n"
                    f"Most common values for {table}.{col}: {top}\n"
                    f"Tip: pick one of the above and rerun."
                )

    return True, ""


# ============================================================
# BUILD 2 VECTOR STORES
#  - schema_store: table/columns (for table selection)
#  - enum_store: low-card values docs (for value grounding retrieval)
# ============================================================

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 1) Schema store
schema_docs: List[Document] = []
for table in ALL_TABLES:
    columns = inspector.get_columns(table)
    schema_text = f"Table: {table}\nColumns:\n" + "\n".join(
        f"{c['name']} ({c['type']})" for c in columns
    )
    schema_docs.append(Document(page_content=schema_text, metadata={"table": table, "doc_type": "schema"}))

schema_store = FAISS.from_documents(schema_docs, embeddings)
print("✅ Schema FAISS index built. Tables indexed:", len(schema_docs))

# 2) Enum store (low-card only)
enum_docs: List[Document] = []
for (t, c), vals in LOW_CARD_VALUES.items():
    if not vals:
        continue
    # chunk values to keep docs small and retrievable
    for i in range(0, len(vals), ENUM_DOC_CHUNK_SIZE):
        chunk = vals[i:i + ENUM_DOC_CHUNK_SIZE]
        enum_text = (
            f"Enum values for {t}.{c}\n"
            f"Column {t}.{c} is categorical. Allowed values include:\n"
            + "\n".join(f"- {v}" for v in chunk)
        )
        enum_docs.append(
            Document(
                page_content=enum_text,
                metadata={"table": t, "column": c, "doc_type": "enum", "chunk_index": i // ENUM_DOC_CHUNK_SIZE}
            )
        )

enum_store: Optional[FAISS] = None
if enum_docs:
    enum_store = FAISS.from_documents(enum_docs, embeddings)
    print("✅ Enum FAISS index built. Enum docs:", len(enum_docs))
else:
    print("ℹ️ No low-card enum docs found; enum_store disabled.")


def retrieve_tables(question: str, k: int = 4) -> Tuple[List[str], List[Document]]:
    retrieved_docs = schema_store.similarity_search(question, k=k)
    picked: List[str] = []
    for d in retrieved_docs:
        t = d.metadata.get("table")
        if t and t not in picked:
            picked.append(t)
    return picked, retrieved_docs

def retrieve_enum_hints(question: str, k: int = ENUM_RETRIEVE_K) -> List[Document]:
    if not enum_store:
        return []
    return enum_store.similarity_search(question, k=k)

def schema_retrieve(question: str, k: int = 4) -> Dict[str, Any]:
    picked_tables, retrieved_docs = retrieve_tables(question, k)
    joins = compute_join_clauses(picked_tables)
    schema_context = "\n\n".join(d.page_content for d in retrieved_docs)

    return {
        "question": question,
        "tables": picked_tables,
        "schema_context": schema_context,
        "join_clauses": joins,
    }


# ============================================================
# OPENAI LLM
# ============================================================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
print("LLM backend:", type(llm).__name__)


# ============================================================
# PROMPTS
# ============================================================

PLAN_PROMPT = """
Return ONLY valid JSON.

Keys:
required_tables
select
filters
group_by
order_by
limit
notes

Question:
{question}

Candidate tables:
{tables}

Schema (may include allowed values for some categorical columns):
{schema}

Output correctness requirement:
- If the question asks for a breakdown "by <dimension>" (e.g., "by region", "by year"),
  include that dimension in group_by AND also include it in select.
- If the question asks for a single overall metric (no "by <dimension>"),
  group_by should be empty and select should contain only the metric(s).
- For categorical filters, use ONLY values present in VALUE HINTS / ENUM CONTEXT when available.
"""

SQL_PROMPT = """
Generate valid SQLite SQL.

Hard rules:
1) Use ONLY these tables: {tables}
2) You MUST start the query with:
   FROM {from_table}
3) If joins are needed, you MUST use ONLY these exact JOIN lines (copy them verbatim, no changes):
{joins}
4) Do NOT invent any other JOIN conditions.
5) Return ONLY SQL text (no markdown).

Question:
{question}

Plan:
{plan}

Schema (may include allowed values for some categorical columns):
{schema}

Result format requirement:
- If group_by is non-empty, SQL MUST select the breakdown columns needed to label each group.

Minimize joins:
- Use the smallest set of tables necessary to answer the question.
- Do not join lookup tables unless their columns are used in SELECT, filters, or group_by.
"""

FIX_SQL_PROMPT = """
You are a SQLite SQL debugger.

Rules:
- Return ONLY corrected SQL (no markdown, no explanation).
- Use ONLY these tables: {tables}
- Use ONLY these JOIN clauses when joins are needed:
{joins}
- Use ONLY columns present in the schema.
- If a filter value doesn't exist (error message shows suggestions), pick a valid suggested value.

Schema:
{schema}

Original question:
{question}

Original SQL:
{sql}

Error:
{error}

Return corrected SQL:
"""


# ============================================================
# LLM HELPERS
# ============================================================

def _extract_json_best_effort(raw: str) -> dict:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        s = raw.find("{")
        e = raw.rfind("}")
        if s != -1 and e != -1 and e > s:
            return json.loads(raw[s:e+1])
        raise

def make_plan(ctx: Dict[str, Any]) -> Dict[str, Any]:
    prompt = PLAN_PROMPT.format(
        question=ctx["question"],
        tables=", ".join(ctx["tables"]),
        schema=ctx["schema_context"],
    )
    raw = llm.invoke(prompt).content.strip()
    return _extract_json_best_effort(raw)

def generate_sql(ctx: Dict[str, Any], plan: Dict[str, Any]) -> str:
    from_table = ctx["tables"][0]
    joins = "\n".join(ctx["join_clauses"]) if ctx["join_clauses"] else "(none)"
    prompt = SQL_PROMPT.format(
        question=ctx["question"],
        tables=", ".join(ctx["tables"]),
        from_table=from_table,
        plan=json.dumps(plan, indent=2),
        joins=joins,
        schema=ctx["schema_context"],
    )
    sql = llm.invoke(prompt).content.strip()
    return sql.replace("
sql", "").replace("
", "").strip()

def fix_sql(ctx: Dict[str, Any], sql: str, error: str) -> str:
    joins = "\n".join(ctx["join_clauses"]) if ctx["join_clauses"] else "(none)"
    prompt = FIX_SQL_PROMPT.format(
        question=ctx["question"],
        tables=", ".join(ctx["tables"]),
        joins=joins,
        schema=ctx["schema_context"],
        sql=sql,
        error=error,
    )
    fixed = llm.invoke(prompt).content.strip()
    return fixed.replace("
sql", "").replace("
", "").strip()


# ============================================================
# VALIDATION + EXECUTION
# ============================================================

def validate_sql_syntax(sql: str) -> Tuple[bool, str]:
    try:
        sqlglot.parse_one(sql, dialect="sqlite")
        return True, ""
    except Exception as e:
        return False, str(e)

def validate_tables_strict(sql: str, allowed_tables: List[str]) -> Tuple[bool, str]:
    sql_lower = sql.lower()
    mentioned = re.findall(r"\bfrom\s+([a-zA-Z_][\w]*)|\bjoin\s+([a-zA-Z_][\w]*)", sql_lower)
    used = set()
    for a, b in mentioned:
        if a:
            used.add(a)
        if b:
            used.add(b)
    if not used:
        return False, "SQL does not reference any tables in FROM/JOIN."

    allowed = {t.lower() for t in allowed_tables}
    illegal = [t for t in used if t not in allowed]
    if illegal:
        return False, f"SQL references tables not in allowed set: {illegal}"
    return True, ""

def execute_sql(sql: str) -> Tuple[bool, str, Optional[pd.DataFrame]]:
    try:
        df = pd.read_sql(sql, engine)
        return True, "", df
    except Exception as e:
        return False, str(e), None


# ============================================================
# END-TO-END RUNNER WITH SELF-HEALING
# ============================================================

def run_text_to_sql(question: str, k: int = 4, max_retries: int = 3) -> Dict[str, Any]:
    # 1) Retrieve schema for initial candidate tables
    step = schema_retrieve(question, k=k)

    # 2) Create plan based on schema context
    plan = make_plan(step)

    # 3) Determine required tables and expand with bridge tables
    required_tables = plan.get("required_tables") or step["tables"]
    required_tables = [t for t in required_tables if t in ALL_TABLES]
    if not required_tables:
        required_tables = step["tables"]

    required_tables = expand_with_bridge_tables(required_tables)

    # 4) Recompute joins for final allowed tables
    join_clauses = compute_join_clauses(required_tables)

    # 5) Deterministic value hints (low-card only) filtered to picked tables
    value_hints = build_value_hints_for_tables(required_tables)

    # 6) Enum store retrieval (low-card only) for extra grounding
    enum_hits = retrieve_enum_hints(question, k=ENUM_RETRIEVE_K)
    enum_context = ""
    if enum_hits:
        # keep enum context compact
        enum_context = "\n\n=== ENUM CONTEXT (LOW-CARD VALUE DOCS FROM RETRIEVAL) ===\n" + "\n\n".join(
            d.page_content for d in enum_hits
        )

    # 7) Final schema_context used for SQL generation + fixes
    schema_context = step["schema_context"]
    if value_hints:
        schema_context += "\n\n=== VALUE HINTS (LOW CARDINALITY, FILTERED BY PICKED TABLES) ===\n" + value_hints
    if enum_context:
        schema_context += enum_context

    # 8) Final ctx
    ctx = {
        **step,
        "tables": required_tables,
        "join_clauses": join_clauses,
        "schema_context": schema_context,
    }

    sql = generate_sql(ctx, plan)
    last_error = ""

    for attempt in range(max_retries + 1):
        ok, err = validate_sql_syntax(sql)
        if not ok:
            last_error = f"Parse error: {err}"
        else:
            ok2, err2 = validate_tables_strict(sql, ctx["tables"])
            if not ok2:
                last_error = f"Table error: {err2}"
            else:
                okj, errj = validate_joins_against_rules(sql)
                if not okj:
                    last_error = f"Join rule error: {errj}"
                else:
                    okv, errv = validate_filter_values(sql, ctx["tables"])
                    if not okv:
                        last_error = f"Filter value error: {errv}"
                    else:
                        ok3, err3, df = execute_sql(sql)
                        if ok3:
                            return {
                                "question": question,
                                "tables": ctx["tables"],
                                "join_clauses": ctx["join_clauses"],
                                "plan": plan,
                                "sql": sql,
                                "df": df,
                                "attempts": attempt,
                                "error": ""
                            }
                        last_error = f"Runtime error: {err3}"

        if attempt == max_retries:
            return {
                "question": question,
                "tables": ctx["tables"],
                "join_clauses": ctx["join_clauses"],
                "plan": plan,
                "sql": sql,
                "df": None,
                "attempts": attempt,
                "error": last_error
            }

        # fix and retry
        sql = fix_sql(ctx, sql, last_error)


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    q = "Average annual_inc for customers with approved loans"
    out = run_text_to_sql(q, k=4, max_retries=3)
    print("Tables:", out["tables"])
    print("Joins:", out["join_clauses"])
    print("Attempts:", out["attempts"])
    print("Error:", out["error"])
    print(out["sql"])
    if out["df"] is not None:
        print(out["df"].head())
