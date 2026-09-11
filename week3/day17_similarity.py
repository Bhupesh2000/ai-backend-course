import numpy as np
from day16_embed import embed_batch, embed_one

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

def dot(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b)

def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))

def rank(query_vec: np.ndarray, doc_vecs: np.ndarray, k: int = 5):
    """One matrix multiply scores ALL documents. No Python loop."""
    scores = doc_vecs @ query_vec            # shape: (n_docs,)
    idx = np.argsort(-scores)[:k]
    return [(int(i), float(scores[i])) for i in idx]

if __name__ == "__main__":
    docs = [
        "To request a refund, open Settings > Billing and click Request Refund.",
        "Returns are accepted within 30 days of purchase for a full reimbursement.",
        "Our shipping partner delivers within 3 business days.",
        "Password reset links expire after 15 minutes.",
        "Money-back guarantee: contact support within 30 days.",
        "The mitochondrion is the powerhouse of the cell.",
    ]
    dv = embed_batch(docs)
    q = embed_one("how do I get my money back")

    print("--- three metrics agree on ranking (normalised vectors) ---")
    for i, d in enumerate(docs):
        print(f"cos={cosine(q, dv[i]):.4f}  dot={dot(q, dv[i]):.4f}  "
              f"l2={euclidean(q, dv[i]):.4f}  | {d[:48]}")

    print("\n--- top 3 ---")
    for i, s in rank(q, dv, k=3):
        print(f"{s:.4f}  {docs[i]}")

    print("\n--- the relevance floor problem ---")
    q2 = embed_one("what is the airspeed velocity of an unladen swallow")
    for i, s in rank(q2, dv, k=3):
        print(f"{s:.4f}  {docs[i]}   ← nothing relevant exists, "
              f"yet we still return 3 results")