import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    Range,
)

from week3.day16_embed import embed_batch, embed_one
from week3.day18_chunk import chunk_markdown


client = QdrantClient(url="http://localhost:6333")

COLLECTION = "docs"


def setup():
    # Create the collection if it doesn't already exist.
    if not client.collection_exists(COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(
                size=1536,
                distance=Distance.COSINE,
            ),
        )

    # Payload indexes are NOT automatic.
    # Create indexes for fields we will filter on.
    for field, schema in [
        ("tenant_id", "keyword"),
        ("doc_id", "keyword"),
        ("version", "integer"),
    ]:
        client.create_payload_index(
            collection_name=COLLECTION,
            field_name=field,
            field_schema=schema,
        )


def point_id(doc_id: str, section_path: str) -> int:
    """
    Generate a deterministic point ID.

    The same doc_id + section_path will always
    produce the same ID.

    This makes repeated ingestion idempotent.
    """

    h = hashlib.sha256(
        f"{doc_id}::{section_path}".encode()
    ).hexdigest()

    return int(h[:15], 16)


def upsert_document(
    md: str,
    doc_id: str,
    title: str,
    tenant_id: str,
    version: int,
):
    # 1. Split the document into chunks.
    chunks = chunk_markdown(
        md,
        doc_id=doc_id,
        doc_title=title,
    )

    # 2. Generate embeddings for all chunks.
    vecs = embed_batch(
        [c.text for c in chunks]
    )

    # 3. Convert every chunk into a Qdrant Point.
    points = [
        PointStruct(
            id=point_id(
                doc_id,
                c.section_path,
            ),

            vector=v.tolist(),

            payload={
                "doc_id": doc_id,
                "tenant_id": tenant_id,
                "section_path": c.section_path,
                "content": c.raw_text,
                "version": version,
            },
        )

        for c, v in zip(chunks, vecs)
    ]

    # 4. Insert/update the points.
    client.upsert(
        collection_name=COLLECTION,
        points=points,
    )

    return len(points)


def search(
    query: str,
    tenant_id: str,
    min_version: int = 0,
    k: int = 5,
):
    # Convert the user's query into an embedding.
    q = embed_one(query).tolist()

    # Search Qdrant using:
    #   1. vector similarity
    #   2. payload filters
    res = client.query_points(
        collection_name=COLLECTION,
        query=q,

        query_filter=Filter(
            must=[
                FieldCondition(
                    key="tenant_id",
                    match=MatchValue(
                        value=tenant_id
                    ),
                ),

                FieldCondition(
                    key="version",
                    range=Range(
                        gte=min_version
                    ),
                ),
            ]
        ),

        limit=k,

        # Return the payload along with each result.
        with_payload=True,
    )

    return [
        (
            p.payload["section_path"],
            p.payload["content"],
            p.score,
        )
        for p in res.points
    ]


if __name__ == "__main__":

    # Create collection + payload indexes.
    setup()

    MD = """
# Payments Guide

## Refunds

Refunds are processed within 5 business days.

### Partial Refunds

A partial refund can be issued for up to 90 days after the transaction.

## Troubleshooting

Decline code 51 means insufficient funds.
"""

    # First ingestion.
    print(
        "upserted:",
        upsert_document(
            MD,
            "payments",
            "Payments Guide",
            "acme",
            3,
        ),
    )

    # Ingest the same document again.
    # Same deterministic IDs → existing points
    # are overwritten instead of duplicated.
    print(
        "re-upsert:",
        upsert_document(
            MD,
            "payments",
            "Payments Guide",
            "acme",
            3,
        ),
        "(same deterministic IDs → overwrite, not duplicate)",
    )

    print(
        "count:",
        client.count(COLLECTION).count,
        "\n",
    )

    # Test vector search.
    for q in [
        "how long for a partial refund",
        "card declined",
    ]:
        print(f"Q: {q}")

        for path, content, score in search(
            q,
            tenant_id="acme",
            k=2,
        ):
            print(
                f"  {score:.4f}  {path}"
            )

        print()

    # Tenant isolation test.
    print(
        "tenant isolation:",
        search(
            "refund",
            tenant_id="globex",
        ),
    )

    # Version filtering test.
    print(
        "version filter (>=99):",
        search(
            "refund",
            "acme",
            min_version=99,
        ),
    )