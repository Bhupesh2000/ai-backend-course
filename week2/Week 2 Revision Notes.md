# Week 2 — LLM APIs: Notes

> **Goal:** Understand LLM APIs as backend dependencies, not just how to call a model.
>
> Week 2 covers: API calls → messages/state → parameters/cost → streaming → structured outputs → failure engineering → FastAPI service.

---

# 1. Core Mental Model

An LLM API is basically an **HTTP dependency**.

```text
Your backend
    |
    | POST + JSON request
    v
LLM vendor API
    |
    | JSON response
    v
Your backend
```

The SDK (`OpenAI`, `Anthropic`, etc.) is mainly a convenient wrapper around the HTTP API.

The important difference from a normal dependency is that LLMs have:

- high latency (seconds)
- token-based cost
- RPM + TPM rate limits
- non-deterministic responses
- partial-success cases where HTTP 200 is still not a usable result

---

# 2. Day 8 — First API Call

## Basic request

OpenAI request:

```python
resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "Hello"}
    ],
    max_tokens=200,
)
```

Think:

```text
model + messages + parameters
        ↓
      API
        ↓
response object
```

The response is an **object**, not just a string.

Important fields:

```python
resp.choices[0].message.content
resp.choices[0].finish_reason

resp.usage.prompt_tokens
resp.usage.completion_tokens
```

## `usage`

`usage` tells you how many tokens were consumed.

```text
input/prompt tokens
output/completion tokens
```

Treat token usage as the **invoice for the request**.

Log it in production.

## `finish_reason`

This tells you how generation ended.

| Value | Meaning | Action |
|---|---|---|
| `stop` | Normal completion | Normal path |
| `length` | Hit output token limit | Do NOT serve blindly |
| `tool_calls` | Model wants to call a tool | Handle tool call |
| `content_filter` | Blocked by safety system | Fallback/error path |

### Critical point

HTTP `200` does **not** automatically mean success.

Example:

```text
HTTP 200
answer = "A database index is used to..."
finish_reason = "length"
```

The answer may have been cut off.

So:

> **HTTP status checks transport success. `finish_reason` checks generation completion.**

---

# 3. OpenAI vs Anthropic Shape

The concepts are similar, but the response/request structures differ.

### OpenAI

```python
messages=[
    {"role": "user", "content": question}
]
```

System can be represented as a message.

Usage:

```python
resp.usage.prompt_tokens
resp.usage.completion_tokens
```

### Anthropic

System prompt is handled differently:

```python
system="..."
```

Content comes back as blocks:

```python
resp.content[0].text
```

Usage:

```python
resp.usage.input_tokens
resp.usage.output_tokens
```

`max_tokens` is required.

### Production lesson

Create one internal interface so the rest of your application doesn't care which vendor is being used.

```text
Application
    |
    v
Internal LLM client
   / \
OpenAI Anthropic
```

This is essentially an **Adapter + Facade** idea.

---

# 4. Python Basics That Matter

## Imports

```python
from openai import OpenAI
```

Means: import `OpenAI` from the installed `openai` package.

You do not declare `OpenAI` yourself because the library provides it.

Same for:

```python
from fastapi.responses import StreamingResponse
```

`StreamingResponse` is provided by FastAPI.

## `.env`

```python
load_dotenv()
client = OpenAI()
```

`load_dotenv()` loads values from `.env` into environment variables.

The OpenAI client can then pick up `OPENAI_API_KEY`.

Never hardcode API keys.

Use:

```text
.env
.gitignore
```

## f-strings

```python
f"finish={reason} in={input_tokens}"
```

The `f` means:

> Evaluate `{...}` expressions and insert their values into the string.

Without `f`:

```python
"finish={reason}"
```

Python treats `{reason}` as ordinary text.

So:

```python
f"Hello {name}"
```

means interpolation.

```python
"Hello {name}"
```

means literal text.

---

# 5. Day 9 — Messages Array

The messages array is the **complete input/context the model sees for that request**.

```python
messages = [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    {"role": "user", "content": "..."},
]
```

## Roles

### System

Standing configuration:

- rules
- persona
- output requirements
- constraints

### User

Human input.

Treat it as **untrusted input**.

### Assistant

Previous model responses included so the model can see conversation history.

---

# 6. LLM APIs Are Stateless

The API does not automatically remember your previous request.

If you want a conversation:

```text
Request 1
[system, user]

Request 2
[system, user, assistant, user]

Request 3
[system, user, assistant, user, assistant, user]
```

**You own the state.**

Typically:

```text
session_id
    ↓
Redis/Postgres
    ↓
message history
    ↓
context assembly
    ↓
LLM API
```

---

# 7. History Trimming

You cannot keep sending unlimited history because the context window is finite and token cost grows.

Preferred approach:

> **Trim by token budget, not by number of messages.**

Example:

```python
while approx_tokens(history) > MAX_HISTORY_TOKENS:
    history = history[2:]
```

The important rule:

> **Drop complete user + assistant pairs.**

Don't leave:

```text
user
assistant
user
```

and delete the assistant reply for the last user.

Also:

> **Never evict the system prompt.**

Growth strategies:

1. Sliding window
2. Token-budgeted history
3. Summarise/compact old history
4. Retrieve relevant old history

---

# 8. Python: Lists and Slicing

```python
history.append(x)
```

Adds `x` to the list.

Equivalent mental model:

```text
push_back(x)
```

Slicing:

```python
history[-3:]
```

means:

> last 3 elements.

```python
history[2:]
```

means:

> everything from index 2 onward.

---

# 9. Day 10 — Parameters

Important parameters:

| Parameter | Meaning |
|---|---|
| `model` | Which model to use |
| `max_tokens` | Maximum generated output |
| `temperature` | Sampling randomness |
| `stop` | Stop generation on specified sequences |
| `n` | Generate multiple candidates |
| `seed` | Best-effort reproducibility |
| `response_format` | JSON/schema output |
| `presence_penalty` | Discourage repeated topics |
| `frequency_penalty` | Discourage repeated tokens |

## `max_tokens`

Think of this as both:

```text
cost bound
+
latency/output-size bound
```

Always set a sensible maximum.

Without it, an unexpectedly long generation can increase both cost and latency.

---

# 10. Token Counting

A rough estimate like:

```python
len(text) // 4
```

is useful for learning, but not accurate enough for production.

Use:

```python
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4o-mini")
tokens = len(enc.encode(text))
```

Token counts are needed for:

- pre-flight validation
- history trimming
- cost estimation

---

# 11. Cost Model

Basic formula:

```text
input_cost =
    input_tokens / 1,000,000 × input_price

output_cost =
    output_tokens / 1,000,000 × output_price

total_cost =
    input_cost + output_cost
```

Monthly:

```text
monthly_cost =
    cost_per_request × requests_per_month
```

## Main cost levers

Generally consider:

1. cheaper model
2. shorter output
3. caching
4. prompt caching
5. smaller prompts
6. batching where latency allows
7. routing by difficulty

---

# 12. RPM vs TPM

LLM providers can limit you on two dimensions:

```text
RPM = Requests Per Minute
TPM = Tokens Per Minute
```

You can be below your RPM limit but still get throttled because your requests contain too many tokens.

Example:

```text
RPM limit = 10,000
TPM limit = 2,000,000
```

If each request uses 8,500 tokens:

```text
2,000,000 / 8,500 ≈ 235 requests/min
```

So TPM, not RPM, becomes the real limit.

> **Capacity planning must consider both requests and tokens.**

---

# 13. Day 11 — Streaming

## Why streaming exists

LLM generation happens sequentially.

Suppose:

```text
Total generation = 8 seconds
First token ready = 0.4 seconds
```

Without streaming:

```text
0s ───────────────────── 8s
                         ↓
                    full response
```

With streaming:

```text
0s ── 0.4s ───────────── 8s
      ↓                   ↓
   first token         final token
```

Streaming does **not necessarily reduce total generation time**.

It improves:

> **TTFT = Time To First Token**

TTFT is much closer to what a human perceives as responsiveness.

---

# 14. SSE

Streaming is commonly delivered using:

> **Server-Sent Events (SSE)**

Conceptually:

```text
data: {"delta":"Hello"}

data: {"delta":" world"}

data: {"delta":"!"}

data: [DONE]
```

It is a long-lived HTTP response where the server sends pieces over time.

---

# 15. Streaming Mental Model — Important

This was one of the confusing Python concepts.

Think:

```text
s = one continuous stream

s
│
├── c1
├── c2
├── c3
├── c4
└── ...
```

Where:

- `s` = stream
- `c` = one chunk

The stream is **not literally a Python list** that keeps growing.

It is better thought of as:

> **A live sequence/source of chunks that become available over time.**

Then:

```python
async for chunk in stream:
    ...
```

means:

> "Give me each chunk from this stream as it becomes available."

---

# 16. The Two `async` Operations

This was another important point.

Streaming code often looks like:

```python
stream = await client.chat.completions.create(
    ...,
    stream=True
)

async for chunk in stream:
    ...
```

These do **different jobs**.

### First: `await`

```python
stream = await create(...)
```

Means:

> Wait until the stream has been established / returned to me.

### Second: `async for`

```python
async for chunk in stream:
```

Means:

> Now consume chunks from that already-established stream as they arrive.

Mental model:

```text
await
  ↓
"I have the pipe."

async for
  ↓
"Give me what is coming through the pipe."
```

---

# 17. Streaming `delta`

Each chunk usually contains a **delta**, meaning an increment, not the entire accumulated answer.

Example:

```text
chunk 1 → "Redis"
chunk 2 → " is"
chunk 3 → " an"
chunk 4 → " in-memory"
```

You accumulate them if you need the complete response:

```python
pieces.append(piece)
```

At the end:

```text
"Redis" + " is" + " an" + " in-memory"
```

---

# 18. `include_usage`

For streaming:

```python
stream_options={"include_usage": True}
```

asks the API to include token usage.

Usage may arrive in the final chunk.

Without this, you may not have token usage for your streamed request.

This matters because:

> **Streaming traffic still needs cost accounting.**

---

# 19. Streaming Trade-offs

Streaming is excellent for humans reading prose.

It is not automatically better.

### Problems introduced by streaming

1. **Pre-send validation becomes difficult**
   - You already sent part of the answer.

2. **Retries become difficult**
   - If the stream dies halfway, you generally cannot simply resume from the vendor.
   - Restarting can generate/pay for the answer again.

3. **Caching is harder**
   - Accumulate the response before writing the final result to cache.

4. **Usage arrives late**
   - Usually at the end.

5. **Long-lived connections**
   - More connections stay open.

6. **Mid-stream errors**
   - HTTP 200 may already have been sent.
   - You cannot suddenly turn that into HTTP 500.

So:

```text
Human reading prose → streaming is useful

Parser / strict validation → buffer first
```

---

# 20. `yield`

Python:

```python
def gen():
    yield "hello"
    yield "world"
```

`yield` produces values **one at a time** instead of returning once.

With streaming:

```python
async def gen():
    async for chunk in stream:
        yield chunk
```

The generator produces each piece as it becomes available.

---

# 21. Day 12 — Structured Outputs

The problem:

```text
"Please return JSON"
```

is not reliable enough for backend code.

Possible failures:

```text
```json
{"category": "BILLING"}
```

Sure! Here's the JSON:
{"category": "BILLING",}

{"result": "billing"}
```

At scale, even a small failure percentage becomes many failures.

---

# 22. Structured Output Reliability Ladder

```text
Level 1
"Please return JSON"
        ↓
Level 2
json_object
        ↓
Level 3
json_schema + strict=true
        ↓
Level 4
schema + Pydantic + business validation
```

## JSON mode

Guarantees:

> syntactically valid JSON

Does NOT necessarily guarantee:

- correct fields
- correct types
- correct enum values

## Strict schema mode

Guarantees:

> output conforms to the specified schema.

This is much closer to a typed interface.

---

# 23. Pydantic

Example:

```python
class Ticket(BaseModel):
    category: Literal["BILLING", "TECHNICAL", "SALES", "OTHER"]
    urgency: Literal["LOW", "MEDIUM", "HIGH"]
    summary: str
    requires_human: bool
```

Pydantic acts like a validating Python data model.

`Literal` gives an enum-like restriction:

```python
Literal["HIGH", "LOW"]
```

Only those values are valid.

---

# 24. Most Important Structured Output Rule

> **Structured output guarantees shape, not truth.**

Example:

```json
{
  "order_id": "ORD-9999"
}
```

This may perfectly satisfy the schema.

But perhaps `ORD-9999` does not exist.

So:

```text
LLM
 ↓
structured output
 ↓
schema validation
 ↓
business validation
 ↓
use result
```

Your backend must verify facts against authoritative data.

---

# 25. Refusal and Truncation

Structured output does not eliminate all failure cases.

You still need to handle:

### Refusal

The model declines to answer.

### Truncation

```text
finish_reason == "length"
```

The model may stop in the middle of the JSON.

So:

> **Always check `finish_reason`, even with structured output.**

Also keep `max_tokens` sufficiently high.

---

# 26. Explicit Uncertainty

Do not force the model to guess.

Prefer something like:

```python
confidence: Literal["HIGH", "LOW"]
```

or nullable fields.

The model should have an explicit:

```text
unknown / cannot determine
```

path.

This is safer than pretending every input has a definite answer.

---

# 27. Production Pattern for Structured Data

Strong architecture:

```text
Unstructured user input
        ↓
       LLM
        ↓
Structured typed object
        ↓
Deterministic validation
        ↓
Business logic / decision
```

The LLM produces **data**.

Your normal backend code makes the **decision**.

Do not let free-form model prose directly make important business decisions.

---

# 28. Day 13 — Failure Engineering

LLM failures fall into categories.

| Failure | Retry? |
|---|---|
| 429 rate limit | Yes |
| 500/502/503 server error | Yes |
| 529 overloaded | Yes |
| Timeout | Usually yes, carefully |
| 400 bad request | No |
| 401 auth | No |
| 404 wrong model | No |
| Content filter | No |
| Truncation | No |

Basic rule:

> **Classify the error before deciding to retry.**

---

# 29. Retry Economics

Unlike many database reads:

> **LLM retries cost money.**

If an 8,000-token request fails and you retry it, the retry consumes tokens too.

Naively:

```text
original
 + retry 1
 + retry 2
```

can multiply cost during an outage.

Therefore:

- cap attempts
- use an overall time budget
- consider request size
- sometimes fail over instead of retrying large requests

---

# 30. Exponential Backoff

Instead of:

```text
retry immediately
retry immediately
retry immediately
```

wait progressively longer.

Conceptually:

```text
attempt 1 → wait ~1s
attempt 2 → wait ~2s
attempt 3 → wait ~4s
```

Exact implementation:

```python
min(2 ** attempt, 16)
```

---

# 31. Jitter

This is important.

Without jitter:

```text
10,000 clients fail
       ↓
all wait 2 seconds
       ↓
all retry together
       ↓
service gets hammered
       ↓
all fail again
```

This is a **thundering herd**.

With jitter:

```text
client 1 → 1.4s
client 2 → 2.2s
client 3 → 1.8s
client 4 → 2.7s
...
```

Retries spread out.

Example:

```python
delay = base_delay * random.uniform(0.5, 1.5)
```

> **Jitter is part of the retry strategy, not an optional decoration.**

---

# 32. `retry-after`

If the vendor tells you:

```text
retry-after: 5
```

prefer respecting that value rather than blindly applying your own backoff.

Mental model:

```text
Vendor:
"Try again after 5 seconds."

Your client:
"Okay."
```

---

# 33. Two Timeout Budgets

Use:

### Per-attempt timeout

Example:

```text
30 seconds
```

How long one attempt may take.

### Overall request budget

Example:

```text
60 seconds
```

How long all attempts together may consume.

Why?

Without an overall budget:

```text
attempt 1 → 30s
backoff
attempt 2 → 30s
backoff
attempt 3 → 30s
```

You can blow your upstream SLA.

---

# 34. Circuit Breaker — Mental Model

This was one of the concepts that needed extra clarification.

A circuit breaker protects **your service** from repeatedly calling a dependency that is currently failing.

Think of an electrical circuit.

```text
Normal:
Your service ─── [CLOSED] ─── LLM
                         calls allowed

Failure threshold reached:
Your service ─── [OPEN]  X  LLM
                  calls blocked
```

Instead of:

```text
request
 ↓
LLM fails
 ↓
retry
 ↓
LLM fails
 ↓
retry
 ↓
LLM fails
```

for every incoming request, the breaker eventually says:

> "Stop calling the broken dependency for a while."

---

# 35. Circuit Breaker States

```text
             failures
CLOSED ─────────────────→ OPEN
  ↑                         │
  │                         │ cooldown
  │                         ↓
  └──────── success ─── HALF_OPEN
```

### CLOSED

Normal operation.

Calls are allowed.

Failures are counted.

### OPEN

Too many failures.

Calls fail immediately.

No network call is made.

This is called **fail fast**.

### HALF_OPEN

After a cooldown:

> Allow a small probe request.

If it succeeds:

```text
HALF_OPEN → CLOSED
```

If it fails:

```text
HALF_OPEN → OPEN
```

---

# 36. Why Circuit Breaker Is Different From Retry

Retry says:

> "This request failed. Maybe try again."

Circuit breaker says:

> "This dependency appears broken. Stop sending requests for a while."

So they work together:

```text
Circuit breaker
      ↓
Should I call the vendor?
      ↓
YES
      ↓
Retry policy
      ↓
How should I retry this failed attempt?
```

---

# 37. Important Circuit Breaker Detail

Not every error means the dependency is broken.

For example:

```text
400 → your request is wrong
401 → your credentials are wrong
```

These should not normally be treated as evidence that the LLM service is down.

A `429` means throttling, not necessarily dependency failure.

So:

> **Circuit breaker failure classification matters.**

---

# 38. Idempotency

LLM calls themselves are not naturally deterministic/idempotent.

If:

```text
LLM result
   ↓
send email
```

and a timeout causes a retry, you could accidentally:

```text
send email
send email
```

The idempotency protection should be on the **side effect**:

```text
LLM
 ↓
business decision
 ↓
idempotency-keyed email/payment/etc.
```

Not merely on the LLM call.

---

# 39. Failure Ladder

A useful fallback order:

```text
cheaper model
     ↓
different vendor
     ↓
cached response
     ↓
static response
```

Degrade gracefully where the product allows it.

---

# 40. Day 14 — FastAPI Service

FastAPI is useful here because it is:

- async-native
- Pydantic-based
- automatically documented with OpenAPI

Service shape:

```text
POST /v1/answer
    → buffered JSON

POST /v1/answer/stream
    → SSE streaming

GET /healthz
    → liveness

GET /readyz
    → readiness
```

---

# 41. Why Two Answer Endpoints?

Because streaming and validation have different guarantees.

### Buffered

```text
LLM
 ↓
complete response
 ↓
validate
 ↓
cache / moderate / process
 ↓
send to client
```

### Streaming

```text
LLM
 ↓
first chunk
 ↓
client
 ↓
next chunk
 ↓
client
```

Once streamed, you cannot "unsend" content.

So:

> Human reading prose → streaming  
> Parser / validated workflow → buffered

---

# 42. `StreamingResponse`

This is a FastAPI class:

```python
from fastapi.responses import StreamingResponse
```

You do not declare it yourself.

It takes a generator/async generator and turns its yielded values into an HTTP streaming response.

Example:

```python
return StreamingResponse(
    gen(),
    media_type="text/event-stream"
)
```

Mental model:

```text
generator
   ↓
yield chunk
   ↓
StreamingResponse
   ↓
HTTP client
```

---

# 43. `AsyncOpenAI`

In an async FastAPI endpoint:

```python
client = AsyncOpenAI()

async def answer(...):
    r = await client.chat.completions.create(...)
```

Use the async SDK.

Do NOT put a blocking synchronous call such as:

```python
OpenAI().chat.completions.create(...)
```

inside an async request path.

A blocking call can block the event loop and hurt concurrency for other requests.

---

# 44. `async def` and `await`

```python
async def answer():
```

defines a coroutine.

```python
await something()
```

means:

> Wait for this asynchronous operation without blocking the event loop.

This is particularly important for LLM APIs because network calls can take seconds.

---

# 45. `async for`

```python
async for chunk in stream:
```

means:

> Repeatedly wait for the next asynchronous item from the stream and process it.

Remember the distinction:

```text
await create()
    ↓
get the stream

async for chunk in stream
    ↓
consume the stream
```

---

# 46. Buffered Endpoint Flow

```text
Client
  ↓
FastAPI
  ↓
Pydantic validates request
  ↓
build messages
  ↓
AsyncOpenAI
  ↓
LLM
  ↓
check finish_reason
  ↓
log tokens/latency
  ↓
Pydantic response
  ↓
Client
```

---

# 47. Streaming Endpoint Flow

```text
Client
  ↓
FastAPI
  ↓
build messages
  ↓
AsyncOpenAI stream
  ↓
async for chunk
  ↓
extract delta
  ↓
yield SSE event
  ↓
client receives piece
  ↓
repeat
```

---

# 48. Health vs Readiness

These are deliberately separate.

## `/healthz`

Answers:

> "Is this application process alive?"

Usually:

```json
{"status": "ok"}
```

It should not depend on the LLM being available.

## `/readyz`

Answers:

> "Is this instance ready to serve traffic?"

It can check the LLM dependency.

This separation prevents a vendor outage from making Kubernetes think your application itself is dead and repeatedly restarting healthy pods.

---

# 49. Configuration From Environment

Instead of hardcoding:

```python
MODEL = "gpt-4o-mini"
```

use:

```python
MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
```

Meaning:

> Use `LLM_MODEL` if configured; otherwise use `"gpt-4o-mini"`.

Similarly:

```python
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "500"))
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
```

Environment values arrive as strings, so convert them:

```python
int(...)
float(...)
```

---

# 50. Request Validation With Pydantic

```python
class AnswerRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    system: str | None = None
```

This means FastAPI/Pydantic validates incoming JSON.

For example:

```json
{
  "question": "Explain indexes"
}
```

is valid.

An empty question is rejected.

The system field can be a string or `None`.

---

# 51. Structured Logging

Instead of random log messages:

```text
something failed
```

emit structured data:

```json
{
  "request_id": "...",
  "model": "gpt-4o-mini",
  "input_tokens": 100,
  "output_tokens": 200,
  "latency_ms": 1200,
  "ttft_ms": 300,
  "finish_reason": "stop"
}
```

Useful fields include:

- request ID
- route
- model
- input tokens
- output tokens
- cost
- latency
- TTFT
- attempts
- finish reason
- cache hit

---

# 52. Request ID

Generate one:

```python
rid = str(uuid.uuid4())
```

Use it in:

- logs
- errors
- client responses

This lets you connect:

```text
user request
    ↕
API logs
    ↕
LLM call
    ↕
failure
```

without exposing internal vendor details.

---

# 53. Never Leak Vendor Errors

Don't simply return the raw OpenAI/Anthropic exception to the user.

Instead:

```text
internal error
     ↓
stable internal mapping
     ↓
client response
```

Example:

```json
{
  "code": "UPSTREAM_UNAVAILABLE",
  "request_id": "..."
}
```

The client should not need to know which vendor exception class occurred.

---

# 54. Week 2 Architecture

Put everything together:

```text
                         Client
                           |
                           v
                       FastAPI
                           |
                 ┌─────────┴─────────┐
                 │                   │
          /v1/answer          /v1/answer/stream
             │                       │
          buffered                  SSE
             │                       │
             └─────────┬─────────────┘
                       ↓
                 Request validation
                       ↓
                Token / cost checks
                       ↓
                  LLM client
                 ┌─────┼─────┐
                 │     │     │
              timeout retry breaker
                 │     │     │
                 └─────┼─────┘
                       ↓
                   Vendor API
                       ↓
                response / chunks
                       ↓
                  validation
                       ↓
                    logging
                       ↓
                    client
```

The important boundary:

> **The application should depend on your internal LLM client, not directly on vendor SDK calls everywhere.**

---

# 55. What You Should Be Able to Explain After Week 2

## API

- What goes into an LLM API request?
- What does the response object contain?
- Why is `usage` important?
- Why is `finish_reason` important?

## Conversation

- Why is the API stateless?
- How do you build multi-turn chat?
- Why trim by tokens rather than messages?
- Why keep the system prompt?
- Why drop complete user/assistant pairs?

## Cost

- Input vs output tokens
- `max_tokens`
- RPM vs TPM
- Monthly cost calculation
- Cost optimisation levers

## Streaming

- Why streaming improves TTFT
- SSE
- chunk/delta
- `yield`
- `StreamingResponse`
- `await` vs `async for`
- why streaming makes validation/retry harder

## Structured output

- JSON mode vs strict schema
- Pydantic
- shape vs truth
- refusal
- truncation
- explicit uncertainty

## Reliability

- retryable vs non-retryable errors
- exponential backoff
- jitter
- retry-after
- per-attempt vs overall timeout
- circuit breaker
- fallback
- idempotency

## FastAPI

- async endpoints
- `AsyncOpenAI`
- request/response Pydantic models
- buffered vs streaming endpoint
- health vs readiness
- structured logging
- environment configuration
- stable error contracts

---

# 56. Your Important "Don't Forget" List

These are the concepts worth revisiting because they were either initially confusing or especially important for backend interviews.

### 1. `f` strings

```python
f"hello {name}"
```

The `f` enables expression interpolation.

---

### 2. Stream mental model

```text
stream = one continuous source of chunks

async for chunk in stream:
    process(chunk)
```

Not literally:

```text
stream = growing Python list
```

---

### 3. `await` vs `async for`

```python
stream = await create(...)
```

= get/establish the stream.

```python
async for chunk in stream:
```

= consume chunks from the stream.

Simple mental model:

```text
await       → "I have the pipe."
async for   → "Give me what's coming through the pipe."
```

---

### 4. `StreamingResponse`

Provided by FastAPI:

```python
from fastapi.responses import StreamingResponse
```

It converts a generator/async generator into a streaming HTTP response.

---

### 5. Circuit breaker

Retry:

> "This request failed; maybe try again."

Circuit breaker:

> "This dependency is failing repeatedly; stop calling it for a while."

States:

```text
CLOSED → OPEN → HALF_OPEN → CLOSED
```

It protects **your service resources**, not the vendor.

---

### 6. HTTP 200 is not enough

Always consider:

```text
HTTP status
+
finish_reason
+
refusal
+
content validity
```

A successful HTTP request can still produce an unusable model response.

---

### 7. Structured output ≠ truth

```text
Schema → shape
Database/business validation → truth
```

Never trust an extracted identifier merely because it passed the schema.

---

### 8. Streaming ≠ faster generation

Streaming mainly improves:

```text
TTFT
```

not necessarily:

```text
total generation time
```

---

### 9. Retry ≠ free

Retries consume tokens and money.

LLM retry policy should consider:

```text
error type
+
request size
+
retry cost
+
overall time budget
```

---

### 10. Async all the way down

For FastAPI streaming/concurrent requests:

```text
FastAPI async
    ↓
AsyncOpenAI
    ↓
await network call
```

A synchronous blocking call inside the async request path can damage concurrency.

---

# 57. Week 2 Interview Cheat Sheet

### Q: Why check `finish_reason`?

Because HTTP 200 can still contain a truncated response.

### Q: Why log token usage?

To understand and attribute cost and build cost/usage dashboards.

### Q: Why token-based trimming?

Message count doesn't represent actual context size.

### Q: Why streaming?

Reduce TTFT and improve perceived responsiveness.

### Q: Why is streaming harder?

It makes validation, retries, caching and error handling harder because data has already been sent.

### Q: JSON mode vs strict schema?

JSON mode → valid JSON.

Strict schema → schema-conforming JSON.

Neither guarantees factual truth.

### Q: Why jitter?

To avoid synchronized retry storms.

### Q: Why circuit breaker?

To stop repeatedly calling a failing dependency and protect your own resources.

### Q: Why two timeout budgets?

Per-attempt timeout controls one call; overall budget controls the entire retry sequence.

### Q: Why AsyncOpenAI in FastAPI?

Because synchronous network calls block the event loop.

### Q: Why `/healthz` and `/readyz` separately?

Liveness asks whether the process is alive; readiness asks whether it can serve traffic.

### Q: Where should deterministic decisions happen?

In normal backend code after the LLM produces structured data.

---

# 58. One-Screen Week 2 Summary

```text
LLM API
  ↓
Request/response objects
  ↓
usage + finish_reason
  ↓
messages = complete model context
  ↓
token budgeting
  ↓
cost + RPM/TPM
  ↓
streaming when humans need progressive output
  ↓
structured output when backend needs data
  ↓
validation: shape ≠ truth
  ↓
timeouts + retries + jitter
  ↓
circuit breaker + fallback
  ↓
FastAPI
  ├── buffered endpoint
  ├── streaming endpoint
  ├── health
  └── readiness
  ↓
structured logs + stable errors
```

## The main principle

> **Treat the LLM like an expensive, slow, probabilistic external dependency.**
>
> Build a normal backend reliability layer around it, but account for the things that make LLMs different: tokens, cost, non-determinism, partial success, streaming, and output validation.
