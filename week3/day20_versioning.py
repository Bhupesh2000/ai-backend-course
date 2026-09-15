import hashlib
import json

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Set

from day18_chunk import Chunk, chunk_markdown


# =========================================================
# Configuration
# =========================================================

# Stores information about what has already been indexed.
STATE_PATH = Path("index/state.json")

# Embedding model currently used by our system.
EMBED_MODEL = "text-embedding-3-small"

# Version of the embedding model/configuration.
#
# We store this with every indexed chunk so that later
# we can detect which chunks need to be re-embedded
# during a model migration.
EMBED_VERSION = "v3"


# =========================================================
# Indexed chunk metadata
# =========================================================

@dataclass
class IndexedChunk:
    """
    Metadata about a chunk that has already been indexed.

    This does NOT contain the actual embedding vector.
    It tells us what we know about the vector that exists
    in our index.
    """

    chunk_id: str
    doc_id: str
    content_hash: str
    section_path: str

    # Which embedding model generated the vector?
    embed_model: str

    # Which version of that model/configuration?
    embed_version: str

    # When was this chunk indexed?
    indexed_at: str


# =========================================================
# Content hashing
# =========================================================

def content_hash(text: str) -> str:
    """
    Generate a fingerprint for the chunk's content.

    Same content  -> same hash
    Changed content -> different hash

    This lets us avoid re-embedding unchanged content.
    """

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()[:16]


# =========================================================
# Chunk ID
# =========================================================

def chunk_id(
    doc_id: str,
    section_path: str,
    h: str
) -> str:
    """
    Create an identifier for a chunk.

    Example:

        payments::Payments > Refunds::abc123
    """

    return f"{doc_id}::{section_path}::{h}"


# =========================================================
# Load indexing state
# =========================================================

def load_state() -> Dict[str, IndexedChunk]:
    """
    Load previously indexed chunk metadata from disk.

    If no state exists, start with an empty index state.
    """

    if not STATE_PATH.exists():
        return {}

    raw = json.loads(
        STATE_PATH.read_text()
    )

    # Convert dictionaries back into IndexedChunk objects.
    return {
        k: IndexedChunk(**v)
        for k, v in raw.items()
    }


# =========================================================
# Save indexing state
# =========================================================

def save_state(
    state: Dict[str, IndexedChunk]
):
    """
    Save the current indexing state to disk.

    IndexedChunk objects
        ↓
    dictionaries
        ↓
    JSON
        ↓
    state.json
    """

    STATE_PATH.parent.mkdir(
        exist_ok=True
    )

    json_data = {
        k: asdict(v)
        for k, v in state.items()
    }

    STATE_PATH.write_text(
        json.dumps(
            json_data,
            indent=2
        )
    )


# =========================================================
# Plan synchronization
# =========================================================

def plan_sync(
    new_chunks: List[Chunk],
    state: Dict[str, IndexedChunk]
):
    """
    Compare the latest document chunks with the existing
    indexing state.

    Returns:

        to_embed
        to_delete
        unchanged

    This is the core of incremental indexing.
    """

    to_embed = []

    # Tracks every chunk that exists in the NEW version
    # of the document.
    seen = set()


    # -----------------------------------------------------
    # Find new or changed chunks
    # -----------------------------------------------------

    for c in new_chunks:

        # Generate a fingerprint of the current content.
        h = content_hash(c.text)

        # Generate the ID for this version of the chunk.
        cid = chunk_id(
            c.doc_id,
            c.section_path,
            h
        )

        # Remember that this chunk exists in the new
        # version of the document.
        seen.add(cid)

        # Check whether this exact chunk already exists
        # in our previous indexing state.
        existing = state.get(cid)


        # -------------------------------------------------
        # Case 1: Completely new content
        # -------------------------------------------------

        if existing is None:

            # We have never indexed this chunk before.
            to_embed.append(
                (cid, c, h)
            )


        # -------------------------------------------------
        # Case 2: Embedding model/version changed
        # -------------------------------------------------

        elif (
            existing.embed_model != EMBED_MODEL
            or
            existing.embed_version != EMBED_VERSION
        ):

            # Content may be unchanged, but the vector was
            # created using an older embedding model/version.
            #
            # Therefore we need a new embedding.
            to_embed.append(
                (cid, c, h)
            )

        # Otherwise:
        #
        # Same content
        # + same embedding model/version
        #
        # => nothing to do.


    # -----------------------------------------------------
    # Find deleted chunks
    # -----------------------------------------------------

    # Which documents are present in the new source?
    doc_ids: Set[str] = {
        c.doc_id
        for c in new_chunks
    }


    # Anything that:
    #
    # 1. belongs to one of these documents
    # 2. existed in the old state
    # 3. was NOT seen in the new version
    #
    # has effectively been deleted.
    to_delete = [
        cid
        for cid, indexed_chunk in state.items()
        if (
            indexed_chunk.doc_id in doc_ids
            and cid not in seen
        )
    ]


    # Everything that wasn't newly embedded is considered
    # unchanged for this simplified exercise.
    unchanged = (
        len(new_chunks)
        - len(to_embed)
    )


    return (
        to_embed,
        to_delete,
        unchanged
    )


# =========================================================
# Apply synchronization
# =========================================================

def apply_sync(
    to_embed,
    to_delete,
    state
):
    """
    Actually apply the plan created by plan_sync().
    """

    now = datetime.now(
        timezone.utc
    ).isoformat()


    # -----------------------------------------------------
    # Add / update chunks
    # -----------------------------------------------------

    for cid, c, h in to_embed:

        # In a real system this would be something like:
        #
        # vec = embed_one(c.text)
        #
        # vector_db.upsert(
        #     cid,
        #     vec,
        #     metadata=...
        # )
        #
        # The lesson leaves the actual vector DB operation
        # as a placeholder because the focus here is the
        # synchronization logic.

        state[cid] = IndexedChunk(
            chunk_id=cid,
            doc_id=c.doc_id,
            content_hash=h,
            section_path=c.section_path,
            embed_model=EMBED_MODEL,
            embed_version=EMBED_VERSION,
            indexed_at=now
        )


    # -----------------------------------------------------
    # Delete removed chunks
    # -----------------------------------------------------

    for cid in to_delete:

        # Real implementation:
        #
        # vector_db.delete(cid)

        # Remove the chunk from our indexing state.
        state.pop(
            cid,
            None
        )


    return state


# =========================================================
# Example / Test
# =========================================================

if __name__ == "__main__":

    # -----------------------------------------------------
    # Version 1 of the document
    # -----------------------------------------------------

    v1 = """
# Payments

## Refunds
Refunds take 5 business days.

## Chargebacks
Response deadline is 7 days.
"""


    # -----------------------------------------------------
    # Version 2 of the document
    #
    # Refunds changed:
    #     5 days -> 3 days
    #
    # Chargebacks disappeared.
    #
    # Disputes is new.
    # -----------------------------------------------------

    v2 = """
# Payments

## Refunds
Refunds take 3 business days.

## Disputes
Disputes must be filed within 14 days.
"""


    # Load whatever we indexed previously.
    state = load_state()


    # Process both document versions to demonstrate
    # incremental synchronization.
    for label, doc in (
        ("first index", v1),
        ("after doc update", v2)
    ):

        # Reuse our Day 18 chunking system.
        chunks = chunk_markdown(
            doc,
            doc_id="payments",
            doc_title="Payments"
        )


        # Determine what needs to happen.
        emb, dele, unch = plan_sync(
            chunks,
            state
        )


        print(
            f"\n--- {label} ---"
        )

        print(
            f"  to embed  : {len(emb)} "
            f"{[c.section_path for _, c, _ in emb]}"
        )

        print(
            f"  to delete : {len(dele)} "
            f"{dele}"
        )

        print(
            f"  unchanged : {unch} "
            f"(free)"
        )


        # Actually apply the changes.
        state = apply_sync(
            emb,
            dele,
            state
        )


    # Persist the final state.
    save_state(state)