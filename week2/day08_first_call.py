import os
from dotenv import load_dotenv
from openai import OpenAI 
import anthropic

load_dotenv()

def ask_openai(question: str) -> dict:
    client = OpenAI()
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": question}],
        max_tokens=200
    )
    return {
        "vendor": "openai",
        "text": resp.choices[0].message.content,
        "finish_reason": resp.choices[0].finish_reason,
        "input_tokens": resp.usage.prompt_tokens,
        "output_tokens": resp.usage.completion_tokens,
    }

def ask_anthropic(question: str) -> dict:
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,                      # required for Anthropic
        messages=[{"role": "user", "content": question}],
    )
    return {
        "vendor": "anthropic",
        "text": resp.content[0].text,
        "finish_reason": resp.stop_reason,   # "end_turn" | "max_tokens" | ...
        "input_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
    }


if __name__ == "__main__":
    q = "Explain a database index to a backend engineer in two sentences."
    for fn in (ask_openai, ask_anthropic):
        r = fn(q)
        print(f"\n--- {r['vendor']} ---")
        print(r["text"])
        print(f"finish={r['finish_reason']}  "
              f"in={r['input_tokens']} out={r['output_tokens']}")