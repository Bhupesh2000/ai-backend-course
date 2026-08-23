import time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

def stream_answer(question: str):
    """Yields text pieces. Prints TTFT, TPS and usage at the end."""
    start = time.perf_counter()
    ttft = None
    pieces = []

    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": question}],
        max_tokens=500,
        temperature=0.3,
        stream=True,
        stream_options={"include_usage": True},   # ← without this: no token counts
    )

    usage = None
    for chunk in stream:
        if chunk.usage:                  # final chunk carries usage
            usage = chunk.usage
        if not chunk.choices:
            continue
        piece = chunk.choices[0].delta.content
        if piece:
            if ttft is None:
                ttft = time.perf_counter() - start
            pieces.append(piece)
            yield piece

    total = time.perf_counter() - start
    out_tok = usage.completion_tokens if usage else 0
    tps = out_tok / total if total else 0
    print(f"\n\n[TTFT={ttft:.2f}s  total={total:.2f}s  "
          f"out_tokens={out_tok}  TPS={tps:.1f}]")

if __name__ == "__main__":
    q = "Explain write-ahead logging and why it enables crash recovery. ~300 words."
    for piece in stream_answer(q):
        print(piece, end="", flush=True)