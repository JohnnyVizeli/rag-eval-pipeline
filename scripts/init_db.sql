-- scripts/init_db.sql

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id          SERIAL PRIMARY KEY,
    source      TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    content     TEXT NOT NULL,
    metadata    JSONB DEFAULT '{}',
    embedding   vector(1536),
    created_at  TIMESTAMPZ DEFAULT NOW()
);

CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

CREATE TABLE IF NOT EXISTS query_logs (
    id              SERIAL PRIMARY KEY,
    question        TEXT NOT NULL,
    answer          TEXT,
    context_docs    JSONB,
    input_tokens    INTEGER,
    output_tokens   INTEGER,
    cost_usd        FLOAT,
    latency_ms      INTEGER,
    created_at      TIMESTAMPZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS eval_results(
    id                  SERIAL PRIMARY KEY,
    run_id              TEXT NOT NULL,
    faithfulness        FLOAT,
    answer_relevancy    FLOAT,
    context_precision   FLOAT,
    context_recall      FLOAT,
    num_questions       INTEGER,
    created_at          TIMESTAMPZ DEFAULT NOW()
);
