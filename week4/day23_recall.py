import numpy as np
import time
from typing import List, Set


# Fixed seed so the experiment produces reproducible results.
rng = np.random.default_rng(42)


def make_corpus(n: int, dims: int = 1536) -> np.ndarray:
    """
    Create n random, unit-normalized vectors.
    These simulate embedding vectors.
    """
    v = rng.standard_normal((n, dims)).astype(np.float32)

    # Normalize each vector to unit length.
    # For normalized vectors, dot product gives the same ranking
    # as cosine similarity.
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def exact_topk(
    vectors: np.ndarray,
    query: np.ndarray,
    k: int
) -> List[int]:
    """
    Exact/brute-force vector search.

    Compares the query against every vector and returns
    the indices of the Top-K most similar vectors.
    """
    scores = vectors @ query

    # Sort by descending similarity and take the first k.
    return np.argsort(-scores)[:k].tolist()


def simulated_ann_topk(
    vectors: np.ndarray,
    query: np.ndarray,
    k: int,
    candidate_pool: int
) -> List[int]:
    """
    Crude IVF/ANN stand-in.

    Instead of searching the entire corpus, randomly select
    only candidate_pool vectors and search within that subset.

    Larger candidate_pool:
        → more work
        → generally better recall
        → generally higher latency

    This is NOT a real ANN implementation.
    It only demonstrates the recall/latency trade-off.
    """

    # Randomly select a subset of the corpus.
    idx = rng.choice(
        len(vectors),
        size=min(candidate_pool, len(vectors)),
        replace=False
    )

    # Calculate similarity only for the selected candidates.
    sub_scores = vectors[idx] @ query

    # Find Top-K within the candidate subset.
    top_local = np.argsort(-sub_scores)[:k]

    # Convert local positions back to original corpus indices.
    return idx[top_local].tolist()


def recall_at_k(
    truth: List[int],
    got: List[int]
) -> float:
    """
    Calculate Recall@K.

    truth = actual Top-K from exact search
    got   = Top-K returned by approximate search
    """

    # Count how many results overlap.
    overlap = len(set(truth) & set(got))

    return overlap / len(truth)


if __name__ == "__main__":

    # Experiment configuration.
    N = 50_000
    K = 10
    QUERIES = 30

    # Create the corpus and query vectors.
    corpus = make_corpus(N)
    queries = make_corpus(QUERIES)

    print(
        f"corpus: {N:,} vectors  "
        f"memory: {corpus.nbytes / 1e6:.0f} MB\n"
    )

    # ---------------------------------------------------------
    # 1. EXACT SEARCH
    # ---------------------------------------------------------
    #
    # We first perform brute-force search.
    # These results become our ground truth.
    #

    t0 = time.perf_counter()

    truths = [
        exact_topk(corpus, q, K)
        for q in queries
    ]

    exact_ms = (
        (time.perf_counter() - t0)
        / QUERIES
        * 1000
    )

    print(
        f"exact (brute force): "
        f"{exact_ms:.2f} ms/query   "
        f"recall = 1.000\n"
    )

    # ---------------------------------------------------------
    # 2. APPROXIMATE SEARCH
    # ---------------------------------------------------------
    #
    # Try different candidate pool sizes and see how
    # recall and latency change.
    #

    print(
        f"{'pool':>8} "
        f"{'ms/query':>10} "
        f"{'recall@10':>10}  "
        f"{'speedup':>8}"
    )

    print("-" * 42)

    for pool in (
        500,
        1000,
        2500,
        5000,
        10_000,
        25_000
    ):

        # Measure approximate-search latency.
        t0 = time.perf_counter()

        results = [
            simulated_ann_topk(
                corpus,
                q,
                K,
                pool
            )
            for q in queries
        ]

        ms = (
            (time.perf_counter() - t0)
            / QUERIES
            * 1000
        )

        # Calculate average Recall@10 across all queries.
        r = float(
            np.mean([
                recall_at_k(t, g)
                for t, g in zip(truths, results)
            ])
        )

        # Compare exact-search latency against
        # approximate-search latency.
        speedup = exact_ms / ms

        print(
            f"{pool:>8} "
            f"{ms:>10.2f} "
            f"{r:>10.3f}  "
            f"{speedup:>7.1f}×"
        )

    print(
        "\nThis is the recall/latency curve. "
        "Every ANN index has one."
    )

    print(
        "Real HNSW achieves far better recall per unit "
        "of work —"
    )

    print(
        "but the SHAPE of the trade-off is exactly this."
    )