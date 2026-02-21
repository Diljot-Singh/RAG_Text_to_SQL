# RAG Text-to-SQL

A retrieval-augmented generation (RAG) system for converting natural language questions into SQL queries. This project uses LLMs (OpenAI) to understand questions and generate accurate SQL queries against a fintech database, with optional tracing and evaluation via Langfuse.

## Project Overview

This project demonstrates:
- **Schema extraction & retrieval**: Efficiently retrieve relevant database tables and schemas for a user question
- **Text-to-SQL generation**: Convert natural language to SQL queries using LLMs
- **Gold label generation**: Create ground-truth datasets for evaluation
- **Prediction evaluation**: Compare generated SQL results against gold labels
- **Langfuse integration**: Optional tracing and observability for all pipeline steps

## Project Structure

```
RAG_Text_to_SQL/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .env                         # Environment configuration (see setup below)
│
├── db/                          # SQLite database
│   └── fintech.db              # Fintech database with loan and customer data
│
├── dataset/                     # Source CSV data
│   ├── customer.csv
│   ├── loan.csv
│   ├── loan_purposes.csv
│   ├── loan_count_by_year.csv
│   ├── loan_with_region.csv
│   └── state_region.csv
│
├── build_db.py                  # Build database from CSV files
├── get_db_schema.py             # Extract and display database schema
├── query_db.py                  # Run ad-hoc SQL queries
├── join_path.py                 # Define valid table relationships
│
├── schema_extraction.py          # Main pipeline: run_text_to_sql(question)
├── generate_gold_label.py        # Create ground-truth SQL+results for evaluation
├── run_eval_predictions.py       # Evaluate predictions against gold labels
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure `.env` File

Create a `.env` file in the project root with the following environment variables:

```env
# OpenAI API Key (required for text-to-SQL generation)
OPENAI_API_KEY=your_openai_api_key_here

# Langfuse Credentials (optional; traces are disabled if not provided)
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key

# Langfuse Host (optional; defaults to cloud.langfuse.com)
LANGFUSE_HOST=https://cloud.langfuse.com
```

**Where to get these credentials:**
- **OpenAI API Key**: Create an account at [openai.com](https://openai.com) and generate an API key in your account settings
- **Langfuse Credentials**: Create an account at [langfuse.com](https://langfuse.com) and obtain credentials from your dashboard
  - The script gracefully handles missing Langfuse credentials; tracing is optional

### 3. Build the Database

```bash
python build_db.py
```

This creates `db/fintech.db` from the CSV files in the `dataset/` folder.

### 4. View Database Schema

```bash
python get_db_schema.py
```

This displays all tables and their columns.

## Usage

### Running the Text-to-SQL Pipeline

```python
from schema_extraction import run_text_to_sql

question = "How many loans are in good standing by region?"
result = run_text_to_sql(question, k=4, max_retries=3)

print("SQL:", result.get("sql"))
print("Result DataFrame:", result.get("df"))
print("Tables Used:", result.get("tables"))
print("Join Clauses:", result.get("join_clauses"))
```

**Parameters:**
- `question` (str): Natural language question
- `k` (int, default=4): Number of relevant tables to retrieve
- `max_retries` (int, default=3): Max retries for SQL correction

**Returns:**
- `sql`: Generated SQL query
- `df`: Results as pandas DataFrame (or error details)
- `tables`: Tables retrieved via RAG
- `join_clauses`: Join relationships identified
- `attempts`: Number of execution attempts
- `error`: Error message if generation failed
- `plan`: Planner agent's reasoning

### Generate Gold Labels

Create a ground-truth evaluation dataset:

```bash
python generate_gold_label.py
```

This:
1. Creates an `eval_gold` SQLite table
2. Executes 18 sample questions and stores results
3. Outputs `eval_gold.jsonl` for inspection

### Evaluate Predictions

Compare your model's generated SQL against gold labels:

```bash
python run_eval_predictions.py
```

This:
1. Loads gold labels from `eval_gold` table
2. Runs the pipeline on each question
3. Compares results (order-insensitive comparison)
4. Creates `eval_report.csv` with detailed metrics
5. Logs all traces to Langfuse (if enabled)

**Output metrics:**
- Execution success rate: % of queries that executed without error
- Correctness rate: % of results matching gold labels exactly
- Detailed per-query results in `eval_report.csv`

### Run Ad-Hoc Queries

```bash
python query_db.py
```

Manually test SQL queries against the fintech database.

## Key Features

### Schema Retrieval (RAG)
- Embeddings-based retrieval of relevant tables
- Supports multi-table queries with join detection
- Defined join rules via `join_path.py`

### LLM Integration
- OpenAI GPT models for SQL generation
- Automatic retry and correction logic
- Preserves context via multi-turn interactions

### Evaluation
- Order-insensitive dataframe comparison
- Numeric tolerance handling (ABS_TOL, REL_TOL)
- Langfuse tracing for observability

### Langfuse Tracing (Optional)
All pipeline execution is traced:
- Plan/reasoning from planner agent
- Schema retrieval steps
- SQL generation attempts
- Execution results and comparison

View traces at your Langfuse dashboard.

## Configuration

Edit these constants in the respective files:

**schema_extraction.py:**
- `K_RETRIEVAL`: Number of tables to retrieve (default: 4)
- `MAX_RETRIES`: Retry limit for SQL correction (default: 3)

**run_eval_predictions.py:**
- `K_RETRIEVAL`, `MAX_RETRIES`: Same as above
- `ABS_TOL`, `REL_TOL`: Numeric comparison tolerances
- `REPORT_CSV`: Output report path

## Database Schema

The fintech database (SQLite) includes:

**Tables:**
- `loan` - Loan records with status, term, purpose, grade, etc.
- `customer` - Customer info with income, home ownership, verification status, etc.
- `state_region` - Mapping of states to regions/subregions
- `loan_purposes` - Distinct loan purposes
- `loan_count_by_year` - Aggregated loan counts

**Valid Joins:**
- loan ↔ customer (via customer_id)
- loan ↔ state_region (via state)
- customer ↔ state_region (via addr_state)

See `join_path.py` for join definitions.

## Troubleshooting

**"No tables found. Check DB_PATH"**
- Run `python build_db.py` to create the database

**"OPENAI_API_KEY not found"**
- Add your OpenAI API key to `.env` file

**"Name 'langfuse' is not defined"**
- This is normal if Langfuse is not installed; the script continues without tracing
- Install with `pip install langfuse` and add Langfuse credentials to `.env` if you want tracing

**SQL generation errors**
- Check if relevant tables are being retrieved (see `tables` in pipeline output)
- Increase `K_RETRIEVAL` to retrieve more tables
- Check database schema with `python get_db_schema.py`

## Files Overview

| File | Purpose |
|------|---------|
| `build_db.py` | Create SQLite database from CSV files |
| `get_db_schema.py` | Display database schema |
| `query_db.py` | Ad-hoc SQL query execution |
| `join_path.py` | Define valid table join relationships |
| `schema_extraction.py` | Main RAG pipeline (text→SQL) |
| `generate_gold_label.py` | Create evaluation gold labels |
| `run_eval_predictions.py` | Evaluate predictions vs gold |

## Next Steps

1. Set up `.env` with OpenAI API key
2. Run `build_db.py` to create the database
3. Test the pipeline: `python -c "from schema_extraction import run_text_to_sql; print(run_text_to_sql('How many loans exist?'))"`
4. Generate gold labels: `python generate_gold_label.py`
5. Evaluate: `python run_eval_predictions.py`
6. Review results in `eval_report.csv`

## License

[Add license info here]

## Contact

[Add contact info here]
