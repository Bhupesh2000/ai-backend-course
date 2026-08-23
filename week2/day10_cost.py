import tiktoken
from dataclasses import dataclass

# pip install tiktoken

PRICING = {                       # USD per 1M tokens — verify against vendor page
    "gpt-4o-mini": {"in": 0.15,  "out": 0.60},
    "gpt-4o":      {"in": 2.50,  "out": 10.00},
}

@dataclass
class CostEstimate:
    input_tokens: int
    output_tokens: int
    cost_usd: float

def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

def count_messages(messages, model: str = "gpt-4o-mini") -> int:
    # ~4 tokens of framing overhead per message, +3 for the reply priming
    return sum(count_tokens(m["content"], model) + 4 for m in messages) + 3

def estimate(messages, expected_output: int, model="gpt-4o-mini") -> CostEstimate:
    in_tok = count_messages(messages, model)
    p = PRICING[model]
    cost = (in_tok / 1e6) * p["in"] + (expected_output / 1e6) * p["out"]
    return CostEstimate(in_tok, expected_output, cost)

def monthly(est: CostEstimate, requests_per_month: int) -> float:
    return est.cost_usd * requests_per_month

if __name__ == "__main__":
    msgs = [
        {"role": "system", "content": "You are a support assistant. " * 40},
        {"role": "user",   "content": "My payment failed three times today."},
    ]
    e = estimate(msgs, expected_output=400)
    print(f"input tokens : {e.input_tokens}")
    print(f"per request  : ${e.cost_usd:.6f}")
    print(f"at 3M/month  : ${monthly(e, 3_000_000):,.2f}")