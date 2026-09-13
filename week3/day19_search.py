import numpy as np
import pickle
import time
from pathlib import Path
from typing import List, Tuple

from day16_embed import embed_batch, embed_one
from day18_chunk import Chunk, chunk_markdown


# ---------------------------------------------------------
# Index storage
# ---------------------------------------------------------
# We keep the vector data and chunk metadata in separate files.
#
# vectors.npy -> NumPy array containing all embeddings
# chunks.pkl  -> Python objects containing chunk text + metadata

INDEX_DIR = Path("index")
INDEX_DIR.mkdir(exist_ok=True)

VEC_PATH = INDEX_DIR / "vectors.npy"
META_PATH = INDEX_DIR / "chunks.pkl"


class VectorIndex:
    """
    A simple in-memory semantic search index.

    `vectors` contains all document embeddings.
    `chunks` contains the corresponding chunk metadata.

    IMPORTANT:
        vectors[i] must always correspond to chunks[i].
    """

    def __init__(self):
        self.vectors: np.ndarray | None = None
        self.chunks: List[Chunk] = []

    def add(self, chunks: List[Chunk]):
        """
        Add new chunks to the index.

        Flow:
            chunks
              ↓
            generate embeddings
              ↓
            add vectors to NumPy matrix
              ↓
            add chunks to metadata list
        """

        # Extract the text that should be embedded.
        #
        # This includes the contextual header added by Day 18.
        texts = [c.text for c in chunks]

        # Generate embeddings for all chunks.
        vecs = embed_batch(texts)

        # If this is the first batch, simply use its vectors.
        #
        # Otherwise, vertically append the new vectors
        # to the existing NumPy matrix.
        if self.vectors is None:
            self.vectors = vecs
        else:
            self.vectors = np.vstack([self.vectors, vecs])

        # Add the chunk metadata.
        #
        # This must remain aligned with self.vectors.
        self.chunks.extend(chunks)

    def search(
        self,
        query: str,
        k: int = 5
    ) -> List[Tuple[Chunk, float]]:
        """
        Search for the top-k chunks most similar to the query.

        Flow:

            Query
              ↓
            embedding
              ↓
            compare with all document vectors
              ↓
            find top-k
              ↓
            return chunks + scores
        """

        # -------------------------------------------------
        # 1. Convert the user's query into an embedding
        # -------------------------------------------------

        t0 = time.perf_counter()

        q = embed_one(query)

        t_embed = time.perf_counter() - t0


        # -------------------------------------------------
        # 2. Calculate similarity with every document
        # -------------------------------------------------

        t1 = time.perf_counter()

        # self.vectors has shape:
        #
        #     (number_of_chunks, 1536)
        #
        # q has shape:
        #
        #     (1536,)
        #
        # Matrix × vector gives:
        #
        #     (number_of_chunks,)
        #
        # Each value is the similarity score of
        # one document against the query.
        scores = self.vectors @ q


        # -------------------------------------------------
        # 3. Find the top-k results
        # -------------------------------------------------

        # Don't ask for more results than we actually have.
        k = min(k, len(scores))

        # argpartition finds the indices of the top-k scores
        # without fully sorting all scores.
        #
        # We use -scores because argpartition works toward
        # smaller values, while we want the largest scores.
        top = np.argpartition(-scores, k - 1)[:k]

        # argpartition does NOT guarantee that those k results
        # are ordered.
        #
        # So now sort only the k selected results.
        top = top[np.argsort(-scores[top])]

        t_search = time.perf_counter() - t1


        # -------------------------------------------------
        # 4. Print timing information
        # -------------------------------------------------

        print(
            f"    [embed={t_embed * 1000:.1f}ms  "
            f"search={t_search * 1000:.2f}ms  "
            f"n={len(self.chunks)}]"
        )


        # -------------------------------------------------
        # 5. Return the actual chunks + similarity scores
        # -------------------------------------------------

        # `top` contains indices such as:
        #
        #     [42, 781, 1532]
        #
        # Since vectors[i] corresponds to chunks[i],
        # we use the same index to retrieve the chunk.
        return [
            (self.chunks[i], float(scores[i]))
            for i in top
        ]

    def save(self):
        """
        Persist the in-memory index to disk.

        vectors.npy -> embedding matrix
        chunks.pkl  -> chunk metadata
        """

        np.save(VEC_PATH, self.vectors)

        META_PATH.write_bytes(
            pickle.dumps(self.chunks)
        )

    @classmethod
    def load(cls) -> "VectorIndex":
        """
        Reconstruct a VectorIndex from the saved files.

        This allows the index to survive a service restart.
        """

        # Create a new VectorIndex object.
        idx = cls()

        # Only load if the vector file exists.
        if VEC_PATH.exists():

            # Load the NumPy embedding matrix.
            idx.vectors = np.load(VEC_PATH)

            # Load the corresponding chunk metadata.
            idx.chunks = pickle.loads(
                META_PATH.read_bytes()
            )

        return idx

    def memory_mb(self) -> float:
        """
        Return the amount of memory occupied by the
        embedding matrix in MB.
        """

        if self.vectors is None:
            return 0.0

        # nbytes = total memory occupied by the NumPy array.
        return self.vectors.nbytes / 1e6


# ---------------------------------------------------------
# Example / Test
# ---------------------------------------------------------

if __name__ == "__main__":

    # A small Markdown document that we'll index.
    DOC = """
# Payments Guide

## Refunds
Refunds are processed within 5 business days to the original payment method.

### Partial Refunds
A partial refund can be issued for up to 90 days after the original transaction.
Partial refunds require the original order ID and a reason code.

## Chargebacks
A chargeback is initiated by the cardholder's bank, not by us. Response
deadline is 7 days.

## Troubleshooting
If a payment fails, check the card expiry date and the billing address match.
Declines with code 51 mean insufficient funds.
"""


    # -----------------------------------------------------
    # Create the index
    # -----------------------------------------------------

    idx = VectorIndex()


    # Day 18:
    #     Markdown → chunks
    #
    # Day 19:
    #     chunks → embeddings → vector index
    idx.add(
        chunk_markdown(
            DOC,
            doc_id="payments",
            doc_title="Payments Guide"
        )
    )


    # Save the index so that we don't lose it
    # when the program stops.
    idx.save()

    print(
        f"indexed {len(idx.chunks)} chunks, "
        f"{idx.memory_mb():.3f} MB\n"
    )


    # -----------------------------------------------------
    # Test semantic search
    # -----------------------------------------------------

    queries = [
        "how long do I have to get money back on part of an order",
        "my card was declined",

        # Nothing in our document talks about France.
        # This demonstrates the relevance-floor problem:
        # the system will still return the closest chunks.
        "what is the capital of France"
    ]


    for q in queries:

        print(f"Q: {q}")

        results = idx.search(q, k=2)

        for chunk, score in results:

            print(
                f"  {score:.4f}  {chunk.section_path}"
            )

            print(
                f"          {chunk.raw_text[:60]}..."
            )

        print()