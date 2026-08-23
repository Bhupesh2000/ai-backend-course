from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

SYSTEM = {"role": "system",
          "content": "You are a terse senior backend engineer. "
                     "Answer in at most 3 sentences."}

MAX_HISTORY_TOKENS = 1500          # deliberately small so you can see trimming

def approx_tokens(messages) -> int:
    """Rough estimate: ~4 chars per token. Replaced with tiktoken on Day 10."""
    return sum(len(m["content"]) for m in messages) // 4

def trim(history):
    """Drop oldest COMPLETE pairs until under budget. System is never in history."""
    h = list(history)
    while approx_tokens(h) > MAX_HISTORY_TOKENS and len(h) >= 2:
        h = h[2:]                   # drop one user+assistant pair
    return h

def chat():
    history = []
    while True:
        user_input = input("\nyou> ").strip()
        if user_input in {"exit", "quit"}:
            break

        history.append({"role": "user", "content": user_input})
        history = trim(history)

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[SYSTEM] + history,
            max_tokens=300,
            temperature=0.3,
        )
        reply = resp.choices[0].message.content
        history.append({"role": "assistant", "content": reply})

        print(f"\nbot> {reply}")
        print(f"     [turns={len(history)//2} "
              f"in={resp.usage.prompt_tokens} "
              f"out={resp.usage.completion_tokens} "
              f"finish={resp.choices[0].finish_reason}]")


if __name__ == "__main__":
    chat()
