import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from week3.day16_embed import embed_batch, embed_one
from week3.day18_chunk import chunk_markdown



DSN = "postgresql://postgres:course@localhost:5432/postgres"


def setup(conn):
    """
    Create the pgvector extension, chunks table,
    and normal PostgreSQL indexes.
    """
    with conn.cursor() as cur:
        cur.execute("""
            CREATE EXTENSION IF NOT EXISTS vector;

            CREATE TABLE IF NOT EXISTS chunks (
                id BIGSERIAL PRIMARY KEY,
                doc_id TEXT NOT NULL,
                tenant_id TEXT NOT NULL,
                section_path TEXT NOT NULL,
                content TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                embed_model TEXT NOT NULL,
                embedding vector(1536) NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),

                UNIQUE (doc_id, section_path, content_hash)
            );

            CREATE INDEX IF NOT EXISTS chunks_tenant_idx
                ON chunks (tenant_id);

            CREATE INDEX IF NOT EXISTS chunks_doc_idx
                ON chunks (doc_id);
        """)

    conn.commit()


def upsert_document(
    conn,
    tenant_id,
    doc_id,
    markdown,
    doc_title=None,
    embed_model="text-embedding-3-small",
):
    """
    Chunk a document, generate embeddings, and insert
    the chunks into PostgreSQL.

    Returns the number of rows inserted.
    """

    # 1. Split document into chunks
    chunks = chunk_markdown(markdown, doc_id, doc_title or doc_id)

    if not chunks:
        return 0

    # 2. Generate embeddings for all chunks.
    # `chunk.text` carries the contextual header, so that is
    # what gets embedded (same convention as day19_search).
    embeddings = embed_batch([c.text for c in chunks])

    rows = []

    # 3. Prepare database rows
    for chunk, embedding in zip(chunks, embeddings):
        # Hash the embedded text: if it changes, the vector
        # changes, so the row should count as new.
        content_hash = hashlib.sha256(
            chunk.text.encode("utf-8")
        ).hexdigest()[:16]

        rows.append(
            (
                doc_id,
                tenant_id,
                chunk.section_path,
                chunk.raw_text,
                content_hash,
                embed_model,
                np.asarray(embedding, dtype=np.float32),
            )
        )

    # 4. Insert all rows
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO chunks (
                doc_id,
                tenant_id,
                section_path,
                content,
                content_hash,
                embed_model,
                embedding
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (doc_id, section_path, content_hash)
            DO NOTHING
            """,
            rows,
        )

        inserted = cur.rowcount

    conn.commit()

    return inserted


def search(conn, tenant_id, query, k=10):
    """
    Search for the most similar chunks belonging
    to the specified tenant.
    """

    # 1. Convert query into an embedding
    query_embedding = embed_one(query)

    with conn.cursor() as cur:

        # 2. Tune HNSW search effort for this transaction
        cur.execute(
            "SET LOCAL hnsw.ef_search = 100"
        )

        # 3. Vector search + tenant filtering
        cur.execute(
            """
            SELECT
                id,
                section_path,
                content,
                1 - (embedding <=> %s) AS similarity
            FROM chunks
            WHERE tenant_id = %s
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            (
                np.asarray(query_embedding, dtype=np.float32),
                tenant_id,
                np.asarray(query_embedding, dtype=np.float32),
                k,
            ),
        )

        return cur.fetchall()


def build_index(conn):
    """
    Build the HNSW index on the embedding column.

    For a large live production table, use
    CREATE INDEX CONCURRENTLY and plan this as
    a maintenance operation.
    """

    with conn.cursor() as cur:

        # Give PostgreSQL more memory while building the index
        cur.execute(
            "SET maintenance_work_mem = '512MB'"
        )

        cur.execute("""
            CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
            ON chunks
            USING hnsw (embedding vector_cosine_ops)
            WITH (
                m = 16,
                ef_construction = 64
            );
        """)

    conn.commit()


def main():

    # Connect to PostgreSQL
    with psycopg.connect(DSN) as conn:

        # Create database structure (installs the vector extension)
        setup(conn)

        # Teach psycopg how to work with pgvector
        register_vector(conn)

        payments_doc = """
        # Payments

        ## Create Payment

        To create a payment, call POST /payments.
        The request must contain the amount and currency.

        ## Refund

        To refund a payment, call POST /refunds.
        The refund request requires the original payment ID.

        ## Failed Payments

        A failed payment can be retried after checking
        the payment status.
        """

        auth_doc = """
        # Authentication

        ## Access Tokens

        Access tokens expire after one hour.

        ## Refresh Tokens

        Refresh tokens can be used to obtain a new
        access token.

        ## Invalid Token

        An invalid token should return HTTP 401.
        """

        # Ingest documents
        inserted = upsert_document(
            conn,
            tenant_id="acme",
            doc_id="payments",
            markdown=payments_doc,
        )

        print("Payments inserted:", inserted)

        inserted = upsert_document(
            conn,
            tenant_id="acme",
            doc_id="auth",
            markdown=auth_doc,
        )

        print("Auth inserted:", inserted)

        # Idempotency test:
        # inserting the same document again should
        # insert zero new rows.
        inserted = upsert_document(
            conn,
            tenant_id="acme",
            doc_id="payments",
            markdown=payments_doc,
        )

        print("Second payments insert:", inserted)

        # Build vector index
        build_index(conn)

        # Search
        results = search(
            conn,
            tenant_id="acme",
            query="How do I issue a refund?",
            k=5,
        )

        print("\nSearch results:")

        for row in results:
            row_id, section_path, content, similarity = row

            print(
                f"\nID: {row_id}"
                f"\nPath: {section_path}"
                f"\nSimilarity: {similarity:.4f}"
                f"\nContent: {content}"
            )

        # Tenant isolation test.
        # Globex has no documents in this example,
        # so this should return no results.
        globex_results = search(
            conn,
            tenant_id="globex",
            query="How do I issue a refund?",
            k=5,
        )

        print(
            "\nGlobex results:",
            globex_results,
        )


if __name__ == "__main__":
    main()