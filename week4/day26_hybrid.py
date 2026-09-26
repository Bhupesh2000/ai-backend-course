from collections import defaultdict
from typing import List, Tuple, Dict
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psycopg
from pgvector.psycopg import register_vector

from week3.day16_embed import embed_one


DSN = "postgresql://postgres:course@localhost:5432/postgres"


# PostgreSQL full-text search setup.
# content_tsv stores a searchable representation of content.
FTS_DDL = """
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS content_tsv tsvector
  GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;

CREATE INDEX IF NOT EXISTS chunks_fts_idx
ON chunks USING gin (content_tsv);
"""


def setup(conn):
    """Create the PostgreSQL full-text search column and index."""
    with conn.cursor() as cur:
        cur.execute(FTS_DDL)

    conn.commit()


def lexical_search(
    conn,
    query: str,
    tenant_id: str,
    k: int,
) -> List[Tuple[int, float]]:
    """
    Search using PostgreSQL full-text search.

    Returns:
        [(document_id, lexical_score), ...]
    """

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                id,
                ts_rank(
                    content_tsv,
                    websearch_to_tsquery('english', %s)
                ) AS r
            FROM chunks
            WHERE tenant_id = %s
              AND content_tsv @@
                  websearch_to_tsquery('english', %s)
            ORDER BY r DESC
            LIMIT %s
            """,
            (query, tenant_id, query, k),
        )


        return [
            (row[0], float(row[1]))
            for row in cur.fetchall()
        ]


def vector_search(
    conn,
    query: str,
    tenant_id: str,
    k: int,
) -> List[Tuple[int, float]]:
    """
    Search using pgvector semantic similarity.

    Returns:
        [(document_id, vector_similarity), ...]
    """
    # Convert the query into an embedding.
    q = embed_one(query)

    with conn.cursor() as cur:
        # Runtime HNSW tuning for this transaction.
        cur.execute("SET LOCAL hnsw.ef_search = 100;")

        cur.execute(
            """
            SELECT
                id,
                1 - (embedding <=> %s) AS sim
            FROM chunks
            WHERE tenant_id = %s
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            (q, tenant_id, q, k),
        )

        return [
            (row[0], float(row[1]))
            for row in cur.fetchall()
        ]


def rrf(
    rankings: List[List[Tuple[int, float]]],
    k_const: int = 60,
    top_n: int = 10,
) -> List[Tuple[int, float]]:
    """
    Combine ranked lists using Reciprocal Rank Fusion.

    RRF uses rank, not the original search scores.

        RRF(doc) = SUM(1 / (k_const + rank))

    This means BM25 and vector scores do not need
    to be normalized to the same scale.
    """

    scores: Dict[int, float] = defaultdict(float)

    for ranking in rankings:
        for rank, (doc_id, _score) in enumerate(
            ranking,
            start=1,
        ):
            scores[doc_id] += 1.0 / (k_const + rank)

    return sorted(
        scores.items(),
        key=lambda x: -x[1],
    )[:top_n]


def hybrid_search(
    conn,
    query: str,
    tenant_id: str,
    k: int = 5,
    fetch: int = 30,
):
    """
    Run lexical + vector search, fuse them using RRF,
    and return the final documents.

    fetch:
        Number of candidates retrieved from EACH retriever
        before RRF fusion.

    k:
        Number of final results returned.
    """

    # Get a larger candidate set from both retrievers.
    lex = lexical_search(
        conn,
        query,
        tenant_id,
        fetch,
    )

    vec = vector_search(
        conn,
        query,
        tenant_id,
        fetch,
    )

    # Fuse the two ranked lists.
    fused = rrf(
        [lex, vec],
        top_n=k,
    )

    ids = [doc_id for doc_id, _ in fused]

    if not ids:
        return []

    # Fetch the actual document content for the final IDs.
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, section_path, content
            FROM chunks
            WHERE id = ANY(%s)
            """,
            (ids,),
        )

        rows = {
            row[0]: (row[1], row[2])
            for row in cur.fetchall()
        }

    # Keep the original ranks so we can inspect
    # why each document appeared in the final result.
    lex_ranks = {
        doc_id: rank
        for rank, (doc_id, _) in enumerate(lex, start=1)
    }

    vec_ranks = {
        doc_id: rank
        for rank, (doc_id, _) in enumerate(vec, start=1)
    }

    return [
        (
            rows[doc_id][0],              # section_path
            rows[doc_id][1],              # content
            rrf_score,
            lex_ranks.get(doc_id),
            vec_ranks.get(doc_id),
        )
        for doc_id, rrf_score in fused
        if doc_id in rows
    ]


if __name__ == "__main__":

    with psycopg.connect(DSN) as conn:

        # Allow NumPy arrays to be passed as pgvector parameters.
        register_vector(conn)

        # Create FTS column and index.
        setup(conn)

        queries = [
            "refund",              # both contribute
            "decline code 51",                       # lexical should win
            "how long do I have to get money back",  # semantic should win
        ]

        for q in queries:

            print(f"\nQ: {q}")

            print(
                f"{'rrf':>8}  "
                f"{'lex':>4} "
                f"{'vec':>4}  "
                f"section"
            )

            print("-" * 60)

            results = hybrid_search(
                conn,
                q,
                "acme",
                k=5,
            )

            for (
                path,
                _content,
                score,
                lexical_rank,
                vector_rank,
            ) in results:

                print(
                    f"{score:>8.5f}  "
                    f"{lexical_rank or '-':>4} "
                    f"{vector_rank or '-':>4}  "
                    f"{path}"
                )