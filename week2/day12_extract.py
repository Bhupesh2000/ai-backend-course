from typing import Literal, Optional, List
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

class Ticket(BaseModel):
    category: Literal["BILLING", "TECHNICAL", "SALES", "OTHER"]
    urgency: Literal["LOW", "MEDIUM", "HIGH"]
    summary: str
    requires_human: bool
    # explicit uncertainty channel — never force a guess
    mentioned_order_ids: List[str]
    confidence: Literal["HIGH", "LOW"]

SYSTEM = (
    "You classify customer support tickets. "
    "Extract order IDs ONLY if they literally appear in the text; "
    "never infer or invent one. "
    "If the ticket is ambiguous, set confidence to LOW."
)

def classify(text: str) -> Optional[Ticket]:
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"<ticket>{text}</ticket>"},
        ],
        response_format=Ticket,
        temperature=0,
        max_tokens=500,
    )
    choice = completion.choices[0]

    if choice.finish_reason == "length":
        raise RuntimeError("truncated — increase max_tokens")
    if getattr(choice.message, "refusal", None):
        return None                      # model declined; handle upstream

    return choice.message.parsed

def verify(ticket: Ticket, known_order_ids: set) -> Ticket:
    """Shape came from the schema. TRUTH comes from here."""
    ticket.mentioned_order_ids = [
        oid for oid in ticket.mentioned_order_ids if oid in known_order_ids
    ]
    if ticket.confidence == "LOW":
        ticket.requires_human = True
    return ticket

if __name__ == "__main__":
    known = {"ORD-1001", "ORD-1002"}
    samples = [
        "My card was charged twice for order ORD-1001 this morning!",
        "How do I change my profile picture?",
        "hey",                                # ambiguous → LOW confidence
        "I was charged for order ORD-9999",   # nonexistent → must be filtered
    ]
    for s in samples:
        t = classify(s)
        if t is None:
            print(f"{s!r} -> REFUSED")
            continue
        t = verify(t, known)
        print(f"{s!r}\n  -> {t.model_dump()}\n")