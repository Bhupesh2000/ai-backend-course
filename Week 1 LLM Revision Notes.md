# Week 1 — LLM Foundations: Quick Revision

## 1. AI → ML → Deep Learning → GenAI → LLM

- **AI:** Machines performing intelligent tasks.
- **ML:** Learns rules/patterns from data.
- **Deep Learning:** ML using neural networks.
- **Generative AI:** Generates text, images, audio, etc.
- **LLM:** Generative AI focused on language/text.

### LLM mental model
`f(text) → text`

- Stateless
- No persistent memory/database
- Produces plausible text
- Memory, context and personality are built by the surrounding backend.

**Backend owns:** state, context assembly, routing, retries, cache, logging, cost, guardrails, evals.

---

## 2. Tokens

- **Token:** Sub-word chunk consumed/produced by the model.
- Tokens are the unit of:
  - Billing
  - Context limits
  - Roughly latency

### Rough estimates
- English: ~4 chars/token
- Code: ~2–3 chars/token
- JSON: often inefficient
- Non-Latin text: often more tokens

### Cost
`cost = input_tokens × input_price + output_tokens × output_price`

(prices are per million tokens)

### Remember
- Output tokens are generally more expensive.
- Output tokens are generated sequentially → more output = more latency.
- Fixed system prompts create a repeated cost.
- Use a tokenizer for accurate token counts; don't trust the model.

### Prompt caching
Put static content first and changing user input last.

---

## 3. Context Window

- **Context window:** Maximum tokens allowed in one request, including input + output.
- **Not memory.**
- Model remains stateless.

### Chat
Backend stores history and resends relevant history on every request.

Consequences:
- More history → more input tokens
- Higher cost
- Higher latency
- Eventually context limit is reached

### If context doesn't fit
1. **Truncate** — drop old content
2. **Summarize** — compress old content
3. **Retrieve/RAG** — fetch only relevant content
4. **Bigger context** — more expensive

### Lost in the middle
Attention is generally stronger at the **start/end** and weaker in the middle.

→ Put critical instructions at the start and reinforce important constraints at the end.

**Context management = memory hierarchy problem.**

---

## 4. Non-Determinism

Same input can produce different outputs because the model **samples** among possible next tokens.

### Temperature
Controls output variation.

- `0` → near-deterministic
- `0.3` → focused
- `0.5–0.7` → natural variation
- `0.8–1.0` → creative
- Very high → incoherent risk

**Temperature 0 is not a guarantee of identical output.**

### Top-p
Limits sampling to a selected probability mass.

**Tune temperature OR top-p, not both.**

### Testing
Avoid:
`assert response == "exact string"`

Prefer:
- Validate schema
- Validate required fields
- Validate allowed enum values
- Validate semantics

**Test structure/meaning, not exact wording.**

---

## 5. Prompt Engineering

### Mental model
**Prompt = versioned configuration artifact**

Treat it like code/config:
- Version
- Review
- Test
- Stage rollout
- Monitor
- Roll back

### High-value techniques

1. **Role + context**
2. **Few-shot examples**
3. **Explicit output format**
4. **Delimit untrusted user data**
5. **Give reasoning room for difficult tasks**

### Prompt structure

`Role → Rules → Examples → Retrieved context → User input → Output format`

Why?
- Static prefix → cacheable
- Important rules → strong attention
- User input → volatile and placed last

### Prompt injection
Separate **instructions** from **untrusted data** using delimiters.

Think:
**Prompt injection ≈ SQL injection**

---

## 6. Hallucinations

### Definition
Model produces a **plausible but incorrect** answer.

Why?
- Model optimizes for plausibility, not truth.
- Failure can look like a successful HTTP 200 response.

### Common failure modes

| Failure | Mitigation |
|---|---|
| Factual fabrication | RAG / grounding |
| Fake citations | Verify supplied sources |
| Outdated knowledge | Inject current data |
| Instruction drift | Repeat + validate |
| Format violation | Schema enforcement |
| Sycophancy | Avoid biased prompting |
| Reasoning failure | Use tools |
| Refusal | Rephrase / fallback |

### Mitigations
1. **Ground with trusted context/RAG**
2. **Require + verify citations**
3. **Allow “I don't know”**
4. **Constrain output**
5. **Validate downstream**
6. **Use tools for deterministic/computable tasks**

**Confidence in tone ≠ correctness.**

---

# Week 1 — Core Mental Model

```text
LLM
│
├── Stateless
│     └── Backend must manage state/history
│
├── Token-based
│     └── Cost + latency + context capacity
│
├── Context-limited
│     └── Truncate / summarize / retrieve
│
├── Non-deterministic
│     └── Sampled output → test structure, not strings
│
├── Prompt-configured
│     └── Version + test + rollout prompts
│
└── Plausibility-optimized
      └── Can hallucinate → ground + verify
```

## 12 Terms to Recall

1. Token
2. Context window
3. Stateless
4. Input vs output tokens
5. Lost in the middle
6. Prompt caching
7. Temperature
8. Top-p
9. Few-shot prompting
10. Delimiter / injection boundary
11. Grounding
12. Citation verification
