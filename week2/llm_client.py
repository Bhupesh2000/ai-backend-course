import time, random, logging
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APITimeoutError, APIStatusError

load_dotenv()
log = logging.getLogger("llm")

MAX_ATTEMPTS = 3
OVERALL_BUDGET_S = 60
PER_ATTEMPT_TIMEOUT_S = 30

@dataclass
class LLMResult:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    attempts: int

class LLMError(Exception): ...
class LLMTruncated(LLMError): ...

class CircuitBreaker:
    def __init__(self, threshold=5, cooldown_s=30):
        self.threshold, self.cooldown_s = threshold, cooldown_s
        self.failures, self.opened_at = 0, None

    def allow(self) -> bool:
        if self.opened_at is None:
            return True
        if time.time() - self.opened_at > self.cooldown_s:
            self.opened_at, self.failures = None, 0   # HALF_OPEN probe
            return True
        return False

    def record(self, ok: bool):
        if ok:
            self.failures, self.opened_at = 0, None
        else:
            self.failures += 1
            if self.failures >= self.threshold:
                self.opened_at = time.time()

_breaker = CircuitBreaker()
_client = OpenAI(timeout=PER_ATTEMPT_TIMEOUT_S)

def _sleep_for(attempt: int, retry_after: Optional[float]) -> float:
    if retry_after:
        return retry_after
    return min(2 ** attempt, 16) * random.uniform(0.5, 1.5)   # jitter

def complete(messages, model="gpt-4o-mini", max_tokens=500,
             temperature=0.3) -> LLMResult:
    if not _breaker.allow():
        raise LLMError("circuit open")

    started = time.perf_counter()
    last_err = None

    for attempt in range(MAX_ATTEMPTS):
        if time.perf_counter() - started > OVERALL_BUDGET_S:
            break
        try:
            r = _client.chat.completions.create(
                model=model, messages=messages,
                max_tokens=max_tokens, temperature=temperature,
            )
            choice = r.choices[0]
            if choice.finish_reason == "length":
                _breaker.record(True)          # not a dependency failure
                raise LLMTruncated("output truncated; raise max_tokens")

            _breaker.record(True)
            return LLMResult(
                text=choice.message.content,
                model=r.model,
                input_tokens=r.usage.prompt_tokens,
                output_tokens=r.usage.completion_tokens,
                latency_s=time.perf_counter() - started,
                attempts=attempt + 1,
            )

        except (RateLimitError, APITimeoutError) as e:
            last_err = e
            _breaker.record(False)
            retry_after = getattr(getattr(e, "response", None), "headers", {}) \
                            .get("retry-after")
            delay = _sleep_for(attempt, float(retry_after) if retry_after else None)
            log.warning("retryable %s attempt=%d sleep=%.1f", type(e).__name__,
                        attempt + 1, delay)
            time.sleep(delay)

        except APIStatusError as e:
            if e.status_code in (500, 502, 503, 529):
                last_err = e
                _breaker.record(False)
                time.sleep(_sleep_for(attempt, None))
                continue
            _breaker.record(False)
            raise LLMError(f"non-retryable {e.status_code}: {e}") from e

    raise LLMError(f"exhausted after {MAX_ATTEMPTS} attempts") from last_err

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    res = complete([{"role": "user", "content": "Define idempotency in one sentence."}])
    print(res.text)
    print(res)