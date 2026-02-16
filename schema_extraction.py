import os
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from collections import deque

import pandas as pd
import sqlglot
from sqlalchemy import inspect, create_engine
from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

# ============================================================
# ENV + DATABASE SETUP
# ============================================================

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "db", "fintech.db")

print("DB:", DB_PATH, "exists:", os.path.exists(DB_PATH))

engine = create_engine(f"sqlite:///{DB_PATH}")
inspector = inspect(engine)

tables = inspector.get_table_names()
print("Tables:", tables)

if not tables:
    raise RuntimeError("No tables found in DB.")

# ============================================================
# BUILD SCHEMA DOCS + FAISS INDEX (LOCAL EMBEDDINGS)
# ============================================================

docs = []
for table in tables:
    columns = inspector.get_columns(table)
    text = f"Table: {table}\nColumns:\n" + "\n".join(
        f"{c['name']} ({c['type']})" for c in columns
    )
    docs.append(Document(page_content=text, metadata={"table": table}))

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = FAISS.from_documents(docs, embeddings)
print("✅ FAISS index built. Tables indexed:", len(docs))

# ============================================================
# FK GRAPH + JOIN RULES
# ============================================================

fk_graph = {
    "customer": ["loan", "state_region"],
    "loan": ["customer", "loan_purposes", "state_region", "loan_count_by_year", "loan_with_region"],
    "state_region": ["loan", "customer", "loan_with_region"],
    "loan_purposes": ["loan"],
    "loan_count_by_year": ["loan"],
    "loan_with_region": ["loan", "state_region"],
}

join_rules = [
    {"left_table": "loan", "left_key": "customer_id", "right_table": "customer", "right_key": "customer_id"},
    {"left_table": "loan", "left_key": "state", "right_table": "state_region", "right_key": "state"},
    {"left_table": "customer", "left_key": "addr_state", "right_table": "state_region", "right_key": "state"},
    {"left_table": "loan", "left_key": "purpose", "right_table": "loan_purposes", "right_key": "purpose"},
    {"left_table": "loan", "left_key": "issue_year", "right_table": "loan_count_by_year", "right_key": "issue_year"},
    {"left_table": "loan_with_region", "left_key": "loan_id", "right_table": "loan", "right_key": "loan_id"},
]

# ============================================================
# RETRIEVAL + JOIN LOGIC
# ============================================================

def retrieve_tables(question: str, k: int = 4):
    retrieved_docs = vector_store.similarity_search(question, k=k)
    tables = []
    for d in retrieved_docs:
        t = d.metadata.get("table")
        if t and t not in tables:
            tables.append(t)
    return tables, retrieved_docs


def shortest_path(graph, start, goal):
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


def join_clause_for_edge(a, b):
    for r in join_rules:
        if r["left_table"] == a and r["right_table"] == b:
            return f"JOIN {b} ON {a}.{r['left_key']} = {b}.{r['right_key']}"
        if r["left_table"] == b and r["right_table"] == a:
            return f"JOIN {b} ON {b}.{r['left_key']} = {a}.{r['right_key']}"
    return None


def compute_join_clauses(tables):
    if len(tables) <= 1:
        return []
    base = tables[0]
    clauses = []
    for t in tables[1:]:
        path = shortest_path(fk_graph, base, t)
        if not path:
            continue
        for i in range(len(path) - 1):
            c = join_clause_for_edge(path[i], path[i + 1])
            if c:
                clauses.append(c)
    return list(dict.fromkeys(clauses))


def schema_retrieve(question, k=4):
    tables, docs = retrieve_tables(question, k)
    joins = compute_join_clauses(tables)
    schema_context = "\n\n".join(d.page_content for d in docs)
    return {
        "question": question,
        "tables": tables,
        "schema_context": schema_context,
        "join_clauses": joins,
    }

# ============================================================
# OPENAI LLM (ONLY)
# ============================================================

llm = ChatAnthropic(
    model="claude-3-haiku-20240307",  # cheaper + fast
    temperature=0,
)

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

Schema:
{schema}
"""

SQL_PROMPT = """
Generate valid SQLite SQL.

Use ONLY these tables: {tables}
Use ONLY provided JOIN clauses.
Return ONLY SQL.

Question:
{question}

Plan:
{plan}

JOIN clauses:
{joins}

Schema:
{schema}
"""

# ============================================================
# CORE PIPELINE
# ============================================================

def make_plan(ctx):
    prompt = PLAN_PROMPT.format(
        question=ctx["question"],
        tables=", ".join(ctx["tables"]),
        schema=ctx["schema_context"],
    )
    raw = llm.invoke(prompt).strip()
    return json.loads(raw[raw.find("{"): raw.rfind("}") + 1])


def generate_sql(ctx, plan):
    joins = "\n".join(ctx["join_clauses"])
    prompt = SQL_PROMPT.format(
        question=ctx["question"],
        tables=", ".join(ctx["tables"]),
        plan=json.dumps(plan, indent=2),
        joins=joins,
        schema=ctx["schema_context"],
    )
    sql = llm.invoke(prompt).strip()
    return sql.replace("```", "").strip()


def validate_sql(sql):
    try:
        sqlglot.parse_one(sql, dialect="sqlite")
        return True, ""
    except Exception as e:
        return False, str(e)


def execute_sql(sql):
    try:
        df = pd.read_sql(sql, engine)
        return True, "", df
    except Exception as e:
        return False, str(e), None


def run_text_to_sql(question, k=4):
    ctx = schema_retrieve(question, k)
    plan = make_plan(ctx)
    sql = generate_sql(ctx, plan)

    ok, err = validate_sql(sql)
    if not ok:
        return {"error": err}

    ok, err, df = execute_sql(sql)
    return {
        "tables": ctx["tables"],
        "joins": ctx["join_clauses"],
        "sql": sql,
        "df": df,
        "error": err,
    }

# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    print("Claude key loaded:", bool(os.getenv("ANTHROPIC_API_KEY")))
    print(llm.invoke("Say hello in one word"))

    result = run_text_to_sql("Average annual_inc by region")

    print("\nTables:", result["tables"])
    print("Joins:", result["joins"])
    print("\nSQL:\n", result["sql"])

    if result["df"] is not None:
        print("\nResult:\n", result["df"].head())
    else:
        print("Error:", result["error"])
