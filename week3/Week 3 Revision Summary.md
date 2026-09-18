# Week 3 — Embeddings: Revision & Personal Notes

> **Goal of Week 3:** Learn how to turn a large document corpus into searchable semantic representations, retrieve relevant chunks efficiently, and understand the production problems around embeddings.

---

# 1. Week 3 in One Picture

```text
Large document corpus
        ↓
      Chunking
        ↓
     Embeddings
        ↓
  Vector representations
        ↓
 Similarity / nearest neighbours
        ↓
    Top-k retrieval
        ↓
 Relevant chunks for RAG
```

Week 3 answers the Week 1 problem:

> "If I have 50,000 documents but only enough prompt space for ~20, how do I find the right 20?"

The core idea:

> **Embedding turns "does this text mean something similar?" into "are these vectors close?"**

---

# 2. Day-by-Day Summary

| Day | Topic | Main Learning |
|---|---|---|
| 15 | What is an embedding? | Represent text meaning as a vector |
| 16 | Embeddings API | Generate vectors, batch requests, count tokens, choose model/dimensions |
| 17 | Similarity metrics | Compare vectors using cosine/dot/L2; understand score limitations |
| 18 | Chunking | Split documents into useful retrieval units |
| 19 | Search engine in NumPy | Build brute-force semantic search yourself |
| 20 | Production pitfalls | Keep embeddings/index correct as documents/models change |
| 21 | Consolidation | Connect everything into one semantic-search pipeline |

---

# 3. Day 15 — What Is an Embedding?

## Mental model

An embedding is:

> **A fixed-length array of numbers representing the semantic characteristics of text.**

Example:

```text
"How do I get a refund?"
        ↓
[0.021, -0.184, 0.077, ...]
```

Another sentence:

```text
"Can I return my purchase?"
        ↓
similar vector
```

The exact numbers are not human-readable. Think of the vector as **coordinates for meaning**.

## Why embeddings exist

Traditional keyword/lexical search looks for matching words.

Example:

```text
Query:
"my card got charged twice"

Document:
"duplicate transaction resolution"
```

The meanings are related, but the words barely overlap.

Keyword search can miss it.

Semantic search:

```text
query → embedding
document → embedding
        ↓
compare vectors
        ↓
find nearby meanings
```

So the documents do not need to use the same words.

## Useful backend analogy

### Geohash analogy

Geohashing converts location into a representation that helps find nearby locations.

Embeddings do something conceptually similar for meaning:

```text
Location:
coordinates → nearby places

Text:
embedding → nearby meanings
```

### Semantic hash analogy

Cryptographic hash:

```text
similar input → very different hash
```

Embedding:

```text
similar meaning → similar vector
```

Same broad idea of converting input into a compact representation, but with almost the opposite objective.

## What an embedding is NOT

1. **Not a summary**
   - You cannot read the vector as a summary.
   - Store the original text separately.

2. **Not reversible in practice**
   - It is a lossy representation.
   - However, research shows partial reconstruction can be possible, which is why embeddings can still be sensitive.

3. **Not comparable across models**
   - Vectors from different embedding models live in different spaces.
   - Even model versions should be treated as incompatible.

4. **Not a truth signal**
   - Similarity means "about the same thing", not "agrees with it."

Example:

```text
"The service is up."
"The service is down."
```

These can be semantically close because both discuss service status.

5. **Not free**
   - Documents cost embedding API calls during indexing.
   - Queries cost embedding calls during search.

## Main uses

- Semantic search
- RAG retrieval
- Deduplication
- Clustering
- Classification
- Recommendations
- Anomaly detection

## Important rule

> **Semantic similarity = aboutness, not correctness.**

It does not automatically understand:

- factual agreement
- recency
- authority
- exact matching
- numeric constraints

This is why later systems combine semantic search with filtering, lexical search, reranking, etc.

---

# 4. Day 16 — Embeddings API

## Mental model

Day 15 was:

```text
What is an embedding?
```

Day 16 was:

```text
How do I actually generate one?
```

Basic API idea:

```python
resp = client.embeddings.create(
    model="text-embedding-3-small",
    input=["first text", "second text"]
)
```

A list of inputs means one batched request.

The returned vectors preserve input order.

---

## Model choice

The lesson's examples:

| Model | Dimensions | Main trade-off |
|---|---:|---|
| `text-embedding-3-small` | 1536 | Default / cheaper |
| `text-embedding-3-large` | 3072 | Better retrieval on harder cases, more storage/cost |
| Open-source models | 384–1024-ish | Can keep data inside your network, but adds operational burden |

Main lesson:

> **Start with the smaller model and upgrade only when evaluation shows that the larger model actually helps your corpus.**

Don't choose a bigger model just because it sounds better.

---

## Dimensions

A vector might be:

```text
1536 dimensions
```

or:

```text
3072 dimensions
```

More dimensions generally mean:

- more storage
- more memory
- more computation

The v3 models support a `dimensions` parameter.

So a larger model can potentially be configured with fewer dimensions.

This is an infrastructure/quality trade-off.

---

## Token limit

A very important lesson:

> **Count tokens before embedding.**

If a document is too large for the embedding model's input limit, it must be chunked first.

Why?

```text
20,000-token document
        ↓
model limit ~8K
        ↓
error OR problematic truncation
```

If content is silently truncated, retrieval quality can degrade without an obvious application error.

This connects directly to Day 18.

---

## Batching

Don't do:

```text
100 chunks
→ 100 API calls
```

Prefer:

```text
100 chunks
→ a few batched API calls
```

But batch by **token count**, not only number of items.

Example:

```text
100 × 100 tokens = 10,000 tokens

100 × 2,000 tokens = 200,000 tokens
```

Both have 100 items, but completely different token usage.

So token limits/TPM matter.

---

## Cost mental model

Embedding is cheap per token, but you embed a lot of data.

Example from the lesson:

```text
1M chunks × 400 tokens
= 400M tokens

At $0.02 / million tokens:
≈ $8
```

The important point is not memorizing the exact price.

Remember:

> **The expensive event is often a full re-embedding during a model migration.**

Also distinguish:

```text
one-time indexing cost
vs
recurring cost of changed documents
vs
full migration cost
```

---

## Normalization

Modern providers may return normalized vectors.

Normalized vector:

```text
vector length ≈ 1
```

Why it matters:

For unit-normalized vectors:

```text
cosine similarity
        ≈
dot product
```

So dot product can be used efficiently.

### Python check

```python
np.linalg.norm(v)
```

If the vector is normalized, the result should be approximately:

```text
1.0
```

---

# 5. Day 17 — Similarity Metrics

We have vectors.

Now we need to answer:

> **How do we decide which vectors are "near" each other?**

Three metrics:

## Cosine similarity

Measures the **angle/direction** between vectors.

Think:

```text
same direction → similar
different direction → less similar
```

It largely ignores vector magnitude.

Good default mental model for text embeddings.

---

## Dot product

Measures direction + magnitude.

But if vectors are normalized:

```text
all vectors have length 1
```

so magnitude is constant.

Therefore:

```text
dot product
≈
cosine similarity
```

for ranking.

This is why dot product is convenient for normalized embeddings.

---

## Euclidean / L2

Straight-line distance.

```text
close points → small distance
far points   → large distance
```

For normalized vectors, it gives the same ranking as cosine, just inverted:

```text
cosine:
higher = better

L2:
lower = better
```

---

## Practical decision tree

```text
Are vectors normalized?

YES
 ↓
dot product is a good choice

NO
 ↓
cosine is safer for semantic similarity
```

Important:

> **Never mix normalized and unnormalized vectors in the same index without understanding the consequences.**

---

# 6. Similarity Scores — Major Production Lesson

A similarity score of:

```text
0.82
```

does NOT mean:

```text
82% relevant
```

Similarity scores are not calibrated probabilities.

They depend on:

- embedding model
- corpus
- query
- query length
- data distribution

So:

```text
0.82 on corpus A
```

cannot automatically be treated the same as:

```text
0.82 on corpus B
```

And thresholds don't automatically transfer between models.

## Rank is generally safer than raw score

Prefer:

```text
Top 5 results
```

over blindly saying:

```text
all results with score > 0.75
```

If you use thresholds, tune them against a labelled evaluation set.

---

# 7. Relevance Floor Problem

Vector search normally returns top-k.

Even when nothing is relevant.

Example:

```text
Query:
"What is the airspeed velocity of an unladen swallow?"

Your corpus:
refunds, payments, passwords
```

The system can still return:

```text
top 3 documents
```

They are simply the least irrelevant ones.

This is dangerous in RAG:

```text
irrelevant documents
      ↓
LLM context
      ↓
LLM answers using bad context
```

So production systems need a **relevance gate**.

Possible approaches:

- tuned similarity threshold
- reranking
- query classification
- hybrid retrieval

---

# 8. Day 18 — Chunking

## Why chunk?

Two reasons:

### 1. Input limits

A huge document cannot fit into one embedding request.

### 2. Dilution

More importantly:

```text
50-page document
    ↓
ONE vector
```

The vector represents many topics at once.

A query about refunds may match weakly because the document also contains:

- installation
- configuration
- troubleshooting
- API reference

So split it:

```text
Document
 ├── Installation
 ├── Configuration
 ├── Refunds
 └── Troubleshooting
```

Now each section has a more focused representation.

---

## Core trade-off

| Small chunks | Large chunks |
|---|---|
| More precise | More contextual |
| Less context | Less precise |
| Cheaper per retrieved chunk | More prompt tokens |
| Can create pronoun/context problems | Can mix multiple topics |

The problem is:

> **Small = precise but context-poor. Large = contextual but less precise.**

---

# 9. Chunking Strategies

### 1. Fixed-size

Split every N tokens/characters.

Simple baseline.

Problem:

```text
sentence gets cut
concept gets cut
```

---

### 2. Fixed-size + overlap

Repeat some content between chunks.

Example:

```text
Chunk 1: [AAAA BBBB CCCC DDDD]
Chunk 2:             [DDDD EEEE FFFF GGGG]
```

The overlap helps concepts spanning boundaries survive.

Typical idea from the lesson:

```text
~10–20% overlap
```

---

### 3. Recursive splitting

Try:

```text
paragraph
   ↓
sentence
   ↓
word
```

Use larger natural boundaries first, then smaller ones if necessary.

Good general-purpose strategy.

---

### 4. Structure-aware chunking

Use the structure already present in the data:

```text
Markdown → headings
HTML     → sections
Code     → functions/classes
Legal    → clauses
Chat     → turns/threads
```

This is particularly valuable for documentation.

---

### 5. Semantic chunking

Embed sentences and detect changes in semantic meaning.

Potentially higher quality, but more expensive.

Usually not necessary if your corpus already has good structure.

---

# 10. Contextual Headers ⭐

One of the most important Day 18 learnings.

Bad chunk:

```text
"It can be issued for up to 90 days."
```

Problem:

> What is "it"?

Better:

```text
[Payments Guide > Refunds > Partial Refunds]

It can be issued for up to 90 days.
```

Now the chunk carries its own context.

This is cheap and often gives a significant retrieval improvement.

### Personal takeaway

> **If I do one chunking improvement first, use contextual headers.**

---

# 11. Small-to-Big / Parent Retrieval

Another way to solve the small-vs-large trade-off:

```text
Small chunk
    ↓
used for precise retrieval
    ↓
find matching chunk
    ↓
retrieve its larger parent section
    ↓
send parent section to LLM
```

So:

```text
Search → small
Generation → larger
```

This gives:

> **Precision during retrieval + context during generation.**

---

# 12. What to Store with Each Chunk

Don't store only:

```text
vector
```

Also store:

- raw chunk text
- document ID
- section path
- character offsets
- embedding model/version
- content hash

Why?

Because later you may need:

- citations
- source links
- deletion
- re-embedding
- migration
- debugging
- staleness detection

---

# 13. Day 19 — Semantic Search in NumPy

This was an important mental-model day.

We built semantic search **without a vector DB**.

We had:

```text
chunks
  ↓
embeddings
  ↓
NumPy matrix
```

Conceptually:

```text
vectors[i] ↔ chunks[i]
```

The vector at index `i` always corresponds to the chunk at index `i`.

---

## NumPy basics needed

### NumPy

NumPy = Numerical Python.

It provides efficient numerical arrays and operations.

### Vector

One-dimensional array:

```text
[0.1, 0.2, 0.3, ...]
```

In our case:

```text
one embedding
```

### Matrix

Many vectors together:

```text
[
  [0.1, 0.2, ...],
  [0.3, 0.4, ...],
  [0.5, 0.6, ...]
]
```

If there are 100K chunks and 1536 dimensions:

```text
shape = (100000, 1536)
```

---

# 14. NumPy Semantic Search

The entire search idea is:

```python
scores = doc_vectors @ query_vector
```

If:

```text
doc_vectors = (N, 1536)
query       = (1536,)
```

then:

```text
scores = (N,)
```

So we get one score per document/chunk.

Then find top-k.

---

## `argsort` vs `argpartition`

### `argsort`

Fully sorts everything.

Conceptually:

```text
sort all N results
        ↓
take top k
```

Cost:

```text
O(N log N)
```

### `argpartition`

Only identifies the top-k region.

Conceptually similar to C++:

```text
std::sort
vs
std::nth_element
```

It avoids fully sorting all N scores.

For large N and small k, this is useful.

---

# 15. NumPy Index vs Vector DB

This was one of my important doubts.

### What we actually built on Day 19

```text
Chunks
  ↓
Embedding API
  ↓
NumPy matrix in memory
  ↓
matrix @ query
  ↓
top-k
```

**We did NOT build a vector database.**

A "NumPy index" simply meant:

```text
embedding matrix in RAM
+
parallel chunk metadata
+
search using NumPy
```

---

## So when does Vector DB enter the picture?

A vector DB is a **production alternative/upgrade**, not what we built yesterday.

Think:

```text
Small/simple system
        ↓
NumPy in-memory index

Need more operational capabilities
        ↓
Vector DB
```

You move toward a vector DB when you need things such as:

- persistence
- multiple service replicas
- concurrent writes
- metadata filtering
- replication
- backups
- incremental updates
- large memory requirements

### Important personal understanding

> **Don't switch to a vector DB just because "vector DB = production".**

First ask what operational requirement NumPy cannot satisfy.

---

# 16. Day 20 — Embedding Pitfalls & Versioning

Day 20 is basically:

> **Yesterday we built a vector search system. Today we learn how to keep it correct when the real world changes.**

Main problems:

| Problem | Resolution |
|---|---|
| Embedding model changes | Zero-downtime migration |
| Document changes | Content hash → re-embed |
| Document/chunk deleted | Detect deletion → delete vector |
| Multilingual retrieval is poor | Multilingual model/translation + language-specific eval |
| Embeddings contain sensitive information | Treat according to source-data classification + deletion/security controls |
| Retrieval quality slowly degrades | Golden evaluation set + metrics |

---

# 17. Model Migration

Suppose:

```text
10M vectors
      ↓
old embedding model
```

You want:

```text
new embedding model
```

You cannot simply convert old vectors.

You need to re-embed.

For zero downtime:

```text
1. Create new index
2. Dual-write
3. Backfill existing corpus
4. Shadow-test old vs new retrieval
5. Cut over
6. Keep old index for rollback
7. Delete old index
```

This is:

> **Expand → Migrate → Contract**

Same broad pattern as online schema migrations.

### AI-specific addition

Normal migration asks:

> Did the data migrate?

Embedding migration must also ask:

> **Did retrieval quality remain good/improve?**

Therefore:

```text
old index
new index
    ↓
compare retrieval quality
```

---

# 18. Stale Embeddings

Problem:

```text
Source:
Refunds take 5 days.

Vector:
represents "Refunds take 5 days"
```

Then source changes:

```text
Refunds take 3 days.
```

But vector wasn't updated.

Now search can return stale information.

### Solution

Hash the content:

```text
chunk
 ↓
SHA-256
 ↓
content_hash
```

On the next ingestion:

```text
new hash == stored hash
    ↓
SKIP

new hash != stored hash
    ↓
RE-EMBED
```

This makes ingestion:

- incremental
- cheap
- idempotent

---

# 19. Deletion Detection

Suppose old version:

```text
Refunds
Chargebacks
```

New version:

```text
Refunds
Disputes
```

Then:

```text
Refunds      → unchanged
Chargebacks  → deleted
Disputes     → new
```

Conceptually:

```text
old chunks - new chunks
        =
deleted chunks
```

This is **set difference**.

Important:

> You need a document → chunk relationship so you can find all affected chunks.

---

# 20. Multilingual Retrieval

A system can work well in English and badly in Hindi/Japanese without throwing an error.

That's a **silent quality failure**.

Possible solutions:

1. Use a multilingual embedding model.
2. Translate queries into the corpus language.
3. Evaluate every supported language.

The important part:

> **If you claim to support a language, test that language.**

---

# 21. Embeddings as Sensitive Data

Embeddings look like:

```text
[0.123, -0.421, 0.083, ...]
```

but they are derived from source data.

So don't assume:

```text
"it's just numbers"
```

means:

```text
"it's safe/public"
```

Engineering approach from the lesson:

- protect embeddings similarly to their source data
- keep them inside the appropriate security boundary
- account for deletion requirements
- review sending customer data to third-party embedding providers
- delete associated vectors when source data must be erased

For legal/compliance questions, distinguish engineering controls from the final legal determination.

---

# 22. Golden Eval Set

Problem:

```text
Month 1 → good retrieval
Month 3 → slightly worse
Month 6 → noticeably worse
```

Nothing necessarily crashes.

Users may be the first to notice.

Solution:

Create a labelled evaluation set:

```text
Query
+
expected relevant result(s)
```

The lesson suggests roughly:

```text
50–200 representative queries
```

Run it:

- during deployments
- periodically
- after model changes
- after chunking changes

Useful retrieval metrics include:

- Recall@k
- MRR

Main lesson:

> **You cannot reliably manage retrieval quality if you don't measure it.**

---

# 23. What Every Stored Vector Should Know

The lesson's important metadata:

```text
embed_model
embed_model_version
content_hash
doc_id
chunk_id
indexed_at
```

Think:

```text
Vector
  +
"What created me?"
"When?"
"From which document?"
"From which content?"
```

Without this metadata, future migration/debugging/deletion becomes much harder.

---

# 24. Day 20 Code — What It Is Doing

The Day 20 code is an **incremental indexing planner**.

It compares:

```text
NEW document chunks
        vs
CURRENT indexing state
```

and produces:

```text
to_embed
to_delete
unchanged
```

Core flow:

```text
new document
     ↓
chunk_markdown()
     ↓
calculate content hash
     ↓
create chunk ID
     ↓
compare with previous state
     ↓
 ┌──────────┬──────────┬───────────┐
 │ new      │ changed  │ deleted   │
 │ / model  │          │           │
 ▼          ▼          ▼
embed      re-embed   delete
```

The most important function is:

```python
plan_sync()
```

The persistence helpers (`load_state`, `save_state`) are supporting infrastructure.

---

# 25. Important Python Learned in Week 3

## List comprehension

```python
[d.embedding for d in resp.data]
```

Transform/collect values from an iterable.

---

## `zip`

```python
zip(texts, vectors)
```

Pairs elements:

```text
text 0 ↔ vector 0
text 1 ↔ vector 1
```

---

## NumPy matrix multiplication

```python
matrix @ vector
```

Efficiently computes similarity against all rows.

---

## `np.linalg.norm`

```python
np.linalg.norm(v)
```

Vector length.

Useful for checking normalization.

---

## `np.argsort`

Gets indices in sorted order.

```python
np.argsort(-scores)[:k]
```

Top-k in descending score order.

---

## `np.argpartition`

Finds the top-k region without fully sorting everything.

---

## `@dataclass`

Used for simple structured records:

```python
@dataclass
class Chunk:
    ...
```

Think of it roughly as a lightweight struct/data object.

---

## `hashlib.sha256`

Used for content fingerprints:

```python
hashlib.sha256(
    text.encode("utf-8")
).hexdigest()
```

---

## Set difference

```python
set(a) - set(b)
```

Useful for finding things present in A but absent from B.

---

## `dict.setdefault`

```python
d.setdefault(k, []).append(v)
```

Useful for grouping values by key without manually checking whether the key exists.

---

## `datetime.now(timezone.utc)`

Use timezone-aware UTC timestamps for stored indexing metadata.

---

# 26. My Important Doubts / Questions From Week 3

These are the questions I asked during the week and the conclusion we reached.

### Q1. What exactly is an embedding?

**Answer:**

A numerical vector representing semantic characteristics/meaning of text.

It is not a summary.

---

### Q2. What is semantic search?

**Answer:**

Search based on **meaning**, not exact word overlap.

```text
query → embedding
documents → embeddings
        ↓
compare vectors
        ↓
closest chunks
```

---

### Q3. What is RAG?

**Answer:**

Retrieval-Augmented Generation.

```text
User query
    ↓
retrieve relevant documents/chunks
    ↓
put them into LLM context
    ↓
LLM generates answer
```

RAG is an **architecture/pattern**, not an LLM.

---

### Q4. What is LangChain?

**Answer:**

A framework/library that provides building blocks for LLM applications, such as:

- model integrations
- embedding integrations
- document loaders
- retrievers
- vector stores
- tools
- chains

Important personal takeaway:

> **Understand RAG and the underlying building blocks first. Don't depend on LangChain to hide the concepts.**

---

### Q5. What is NumPy?

**Answer:**

NumPy = Numerical Python.

It provides efficient arrays/matrices and numerical operations.

For Week 3:

```text
vector = one embedding
matrix = many embeddings
```

---

### Q6. What is a NumPy index?

**Answer:**

Not a special NumPy product.

It simply means:

```text
NumPy matrix containing embeddings
+
metadata list
+
search using matrix operations
```

with:

```text
vectors[i] ↔ chunks[i]
```

---

### Q7. Why do we need a vector DB?

**Answer:**

We don't automatically need one.

NumPy is fine for small/simple systems.

Move toward a vector DB when operational requirements justify it:

- persistence
- concurrent writes
- multiple replicas
- metadata filtering
- replication
- backups
- incremental updates
- memory/scale

---

### Q8. When did the vector DB enter the picture?

**Answer:**

It was introduced as a **production alternative**, not as what we built on Day 19.

Day 19:

```text
NumPy index
```

Day 20:

```text
operational problems that exist regardless of
whether the underlying index is NumPy or a vector DB
```

---

### Q9. What does normalized vector mean?

Think of a vector as an arrow.

Normalization makes its length:

```text
1
```

while keeping its direction.

Example:

```text
[3, 4]
```

has length 5.

Normalized:

```text
[0.6, 0.8]
```

has length 1.

For normalized embeddings:

```text
dot product ≈ cosine similarity
```

---

### Q10. Why not just use cosine?

You can.

For normalized embeddings, cosine and dot product produce the same ranking.

Dot product avoids the extra normalization/division work, so it's convenient and fast.

---

### Q11. Why not just embed the whole document?

Two reasons:

1. Input token limit.
2. More importantly, **semantic dilution**.

One vector representing many unrelated topics becomes less precise for focused queries.

---

### Q12. Why overlap?

If a concept crosses a chunk boundary:

```text
Chunk 1:
... part A part B

Chunk 2:
part C part D ...
```

overlap gives at least one chunk more of the complete context.

---

### Q13. Why contextual headers?

Because chunks can otherwise be ambiguous.

Bad:

```text
"It supports up to 90 days."
```

Better:

```text
[Payments > Refunds > Partial Refunds]
It supports up to 90 days.
```

The header provides context before embedding.

---

### Q14. What happens if nothing relevant exists?

Vector search still returns top-k.

So:

```text
nothing relevant
        ↓
least irrelevant documents
        ↓
LLM receives them
        ↓
potentially bad answer
```

This is the relevance-floor problem.

---

### Q15. Does a similarity score of 0.82 mean 82% relevant?

**No.**

Similarity scores are not calibrated probabilities.

Absolute scores don't reliably transfer across:

- models
- corpora
- query types

Rank is generally more useful.

---

### Q16. Why can't vectors from different models be compared?

Because each embedding model defines its own vector space.

So:

```text
Model A vector
vs
Model B vector
```

doesn't have a meaningful direct comparison.

Model migration therefore requires re-embedding.

---

### Q17. Why does Day 20 talk about a vector DB when Day 19 used NumPy?

Because Day 20 is teaching **production operation/versioning**, not changing the implementation we built.

The concepts apply to either:

```text
NumPy index
```

or:

```text
vector DB
```

---

# 27. The Most Important Connections Across the Week

## Connection 1 — Chunking → Embedding

```text
Large document
     ↓
chunk
     ↓
embedding
```

Chunking determines what each vector represents.

Therefore bad chunking can produce bad retrieval even with a great embedding model.

---

## Connection 2 — Embedding → Similarity

```text
text
 ↓
vector
 ↓
similarity metric
 ↓
ranking
```

Embedding creates the representation.

Similarity determines how we compare representations.

---

## Connection 3 — Similarity → RAG

```text
query
 ↓
embedding
 ↓
top-k search
 ↓
relevant chunks
 ↓
LLM
```

This is the retrieval half of RAG.

---

## Connection 4 — Chunking → Search quality

If chunks are:

```text
too small → missing context
too large → diluted meaning
```

So chunking directly affects retrieval quality.

---

## Connection 5 — Day 19 → Vector DB

Day 19 taught:

> "I can build vector search myself."

That makes it easier to understand what a vector DB actually provides.

It isn't magic.

At a basic level:

```text
vectors
+
similarity search
```

A production vector DB adds operational capabilities around that.

---

## Connection 6 — Day 20 → Backend engineering

Day 20 is where embeddings start looking like normal backend infrastructure.

```text
Source of truth
       ↓
derived index
       ↓
sync
       ↓
versioning
       ↓
reconciliation
       ↓
monitoring
```

This is very similar to:

- caches
- materialized views
- CDC consumers
- derived data pipelines

---

# 28. Production Architecture Mental Model

A production RAG retrieval system can be thought of as:

```text
                 SOURCE OF TRUTH
                  Documents
                      │
                      │ change events / sync
                      ▼
             ┌─────────────────┐
             │ Ingestion       │
             │ Pipeline        │
             └────────┬────────┘
                      │
              ┌───────┴────────┐
              │                │
           Chunking          Hashing
              │                │
              └───────┬────────┘
                      │
                      ▼
              Embedding API
                      │
                      ▼
                Vector Index
                      │
                      │
USER QUERY ──► Query Embedding
                      │
                      ▼
                 Top-k Search
                      │
                      ▼
             Relevant Chunks
                      │
                      ▼
                    RAG
                      │
                      ▼
                    LLM
```

Week 3 mostly covers the left/middle portion:

```text
Documents
   ↓
Chunking
   ↓
Embedding
   ↓
Vector search
   ↓
Operational maintenance
```

---

# 29. What I Should Be Able to Explain in an Interview

## Basic

> What is an embedding?

A fixed-length numerical representation of text semantics where similar meanings tend to produce nearby vectors.

---

## Why?

> Why not keyword search?

Keyword search requires vocabulary overlap. Embeddings can retrieve semantically similar content even when different words are used.

---

## Chunking

> Why chunk documents?

Because of model input limits and semantic dilution. Smaller chunks improve precision but lose context; larger chunks preserve context but dilute the representation.

---

## Context

> How do you improve context without making chunks huge?

Use contextual headers and/or small-to-big parent retrieval.

---

## Search

> How would you implement semantic search?

```text
query → embedding
        ↓
matrix @ query
        ↓
top-k
```

For a small corpus, brute-force NumPy can be sufficient.

---

## Vector DB

> When would you introduce a vector DB?

When operational needs such as persistence, concurrent writes, filtering, replication, incremental updates, or memory/scale make an in-memory NumPy index impractical.

---

## Migration

> How do you change embedding models with zero downtime?

```text
new index
→ dual-write
→ backfill
→ shadow evaluation
→ cutover
→ rollback window
→ remove old index
```

---

## Staleness

> How do you know a chunk changed?

Content hash.

```text
same hash → skip
different hash → re-embed
```

---

## Deletion

> How do you detect deleted chunks?

Compare the old indexed set with the new source set.

```text
old - new = deleted
```

---

## Quality

> How do you know retrieval got worse?

Golden evaluation set + retrieval metrics such as Recall@k and MRR.

---

# 30. Week 3 "Don't Forget These" List

```text
1. Embedding = semantic coordinates
2. Semantic search ≠ lexical search
3. Embedding ≠ summary
4. Similarity ≠ truth
5. Different models → incompatible vector spaces
6. Chunk before embedding
7. Small vs large chunks = precision vs context
8. Contextual headers are a cheap, high-value improvement
9. Small-to-big = small for retrieval, parent for generation
10. Normalized vectors → dot product and cosine give same ranking
11. Similarity score ≠ probability/relevance percentage
12. Vector search can return irrelevant top-k results
13. NumPy can implement basic vector search
14. Vector DB is not automatically required
15. Content hash → incremental/idempotent indexing
16. Deletion detection → old chunks vs new chunks
17. Model migration → expand-migrate-contract
18. Shadow phase compares retrieval quality
19. Embeddings inherit sensitivity of source data
20. Golden eval set protects against quality decay
```

---

# 31. My Current Week 3 Mental Model

The simplest version I should remember:

```text
DOCUMENT
   ↓
CHUNK
   ↓
EMBED
   ↓
VECTOR
   ↓
COMPARE WITH QUERY VECTOR
   ↓
TOP-K CHUNKS
   ↓
RAG / LLM
```

And in production:

```text
SOURCE CHANGES
   ↓
HASH
   ↓
ONLY RE-EMBED CHANGED CONTENT

SOURCE DELETES
   ↓
DELETE ASSOCIATED CHUNKS/VECTORS

MODEL CHANGES
   ↓
NEW INDEX
   ↓
BACKFILL
   ↓
SHADOW TEST
   ↓
CUTOVER

QUALITY CHANGES
   ↓
GOLDEN EVAL SET
```

---

# 32. Before Moving to Week 4

I should be comfortable answering these without looking at notes:

1. What is an embedding?
2. Why does semantic search solve a problem keyword search cannot?
3. Why isn't an embedding a summary?
4. Why can "service is up" and "service is down" be close?
5. Why do we chunk?
6. What is the small-vs-large chunk trade-off?
7. Why are contextual headers useful?
8. Explain small-to-big retrieval.
9. Explain cosine vs dot product for normalized vectors.
10. Why is a similarity score not a probability?
11. What is the relevance-floor problem?
12. Explain a NumPy index.
13. When would you move from NumPy to a vector DB?
14. How does content hashing prevent unnecessary re-embedding?
15. How do you detect deletions?
16. How do you migrate 10M vectors to a new model without downtime?
17. Why do we need a shadow phase?
18. Why should embeddings be treated as sensitive data?
19. What is a golden eval set?
20. Explain the entire Week 3 pipeline in 2 minutes.

---

# 33. Final Week 3 Takeaway

> **Week 3 taught me how semantic retrieval works underneath RAG.**

I now have the basic chain:

```text
Documents
   ↓
Chunking
   ↓
Embeddings
   ↓
Vector similarity
   ↓
Top-k retrieval
```

And I understand that production quality depends not just on the embedding model, but heavily on:

```text
chunking
+ context
+ similarity/relevance handling
+ indexing strategy
+ synchronization
+ model versioning
+ evaluation
```

The biggest conceptual shift from Week 2 → Week 3 is:

> **The LLM does not need to know everything. Instead, my backend first finds the right information and then gives only that relevant information to the LLM.**

That is the foundation for the RAG architecture I'll build in later weeks.
