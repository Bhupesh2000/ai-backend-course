import re
from dataclasses import dataclass
from typing import List

import tiktoken


# Tokenizer used to count tokens.
_enc = tiktoken.get_encoding("cl100k_base")


def n_tok(s: str) -> int:
    """Return the number of tokens in a string."""
    return len(_enc.encode(s))


@dataclass
class Chunk:
    """
    Represents one chunk produced from a document.

    `text` is the text that will actually be embedded.
    `raw_text` is the original text without the contextual header.
    """

    text: str
    raw_text: str

    # Metadata about where this chunk came from
    doc_id: str
    section_path: str

    # Position of the chunk in the original document
    start_char: int
    end_char: int

    # Useful for tracking chunk size
    token_count: int

    # Important when we later migrate embedding models
    embed_model: str = "text-embedding-3-small"


# Target size of each chunk.
# This is a target, not a strict requirement.
TARGET_TOKENS = 400

# When creating the next chunk, keep some content
# from the previous chunk to preserve context across boundaries.
OVERLAP_TOKENS = 60


# Regex for detecting Markdown headings:
#
# # Heading       -> level 1
# ## Heading      -> level 2
# ### Heading     -> level 3
# ...
#
# The MULTILINE flag makes ^ work at the beginning of every line.
HEADING = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


def split_by_headings(md: str):
    """
    Split a Markdown document into sections based on headings.

    Returns:
        (section_path, body, start_char)

    Example:

        # Payments
        ## Refunds
        Refund information...

    becomes approximately:

        ("Payments > Refunds", "Refund information...", position)
    """

    # Find all Markdown headings in the document.
    matches = list(HEADING.finditer(md))

    # If there are no headings, treat the entire document as one section.
    if not matches:
        yield ("", md, 0)
        return

    # Keeps track of the current heading hierarchy.
    #
    # Example:
    # # Payments
    # ## Refunds
    # ### Partial Refunds
    #
    # stack becomes:
    # ["Payments", "Refunds", "Partial Refunds"]
    stack: List[str] = []

    for i, m in enumerate(matches):

        # Number of # characters tells us the heading level.
        level = len(m.group(1))

        # Actual heading text.
        title = m.group(2).strip()

        # Remove headings that are deeper than the current level
        # and add the current heading.
        #
        # Example:
        # ["Payments", "Refunds", "Partial Refunds"]
        # encountering "## Troubleshooting"
        # becomes:
        # ["Payments", "Troubleshooting"]
        stack = stack[: level - 1] + [title]

        # Content starts immediately after this heading.
        body_start = m.end()

        # Content ends just before the next heading.
        body_end = (
            matches[i + 1].start()
            if i + 1 < len(matches)
            else len(md)
        )

        body = md[body_start:body_end].strip()

        # Convert the heading hierarchy into a useful path.
        section_path = " > ".join(stack)

        yield (section_path, body, body_start)


def split_long(body: str, start_char: int):
    """
    Split a section if it is larger than TARGET_TOKENS.

    Strategy:
        1. Try to keep paragraphs together.
        2. When a chunk becomes too large, finalize it.
        3. Carry some paragraphs from the end of the previous
           chunk into the next chunk as overlap.
    """

    # If the section is already small enough, don't split it.
    if n_tok(body) <= TARGET_TOKENS:
        return [(body, start_char)]

    # First try to split at paragraph boundaries.
    #
    # A blank line generally separates Markdown paragraphs.
    paragraphs = [
        p
        for p in re.split(r"\n\s*\n", body)
        if p.strip()
    ]

    out = []

    # Current chunk being constructed.
    cur = []

    # Number of tokens currently in `cur`.
    cur_tok = 0

    # Character position where the current chunk starts.
    cur_start = start_char

    for p in paragraphs:

        pt = n_tok(p)

        # If adding this paragraph would exceed our target,
        # finalize the current chunk first.
        if cur_tok + pt > TARGET_TOKENS and cur:

            out.append(("\n\n".join(cur), cur_start))

            # -----------------------------
            # Create overlap
            # -----------------------------
            #
            # Take paragraphs from the END of the previous chunk
            # until we have roughly OVERLAP_TOKENS.
            tail = []
            tail_tok = 0

            for prev in reversed(cur):

                if tail_tok >= OVERLAP_TOKENS:
                    break

                # Insert at the beginning because we are
                # traversing the previous paragraphs backwards.
                tail.insert(0, prev)
                tail_tok += n_tok(prev)

            # Start the next chunk with the overlap.
            cur = list(tail)
            cur_tok = tail_tok

            # NOTE:
            # This simplified implementation keeps the same
            # section start position. In production, you'd usually
            # calculate the exact character offset of the new chunk.
            cur_start = start_char

        # Add the current paragraph to the chunk.
        cur.append(p)
        cur_tok += pt

    # Don't forget the final chunk.
    if cur:
        out.append(("\n\n".join(cur), cur_start))

    return out


def chunk_markdown(md: str, doc_id: str, doc_title: str) -> List[Chunk]:
    """
    Main function.

    Converts a Markdown document into a list of Chunk objects.

    Flow:

        Markdown
           ↓
        Find sections using headings
           ↓
        Split oversized sections
           ↓
        Add contextual header
           ↓
        Create Chunk objects
    """

    chunks: List[Chunk] = []

    # First divide the document using its Markdown structure.
    for section_path, body, start in split_by_headings(md):

        # Ignore empty sections.
        if not body:
            continue

        # Add document title to the heading hierarchy.
        #
        # Example:
        # Payments Guide > Refunds > Partial Refunds
        full_path = (
            f"{doc_title} > {section_path}"
            if section_path
            else doc_title
        )

        # A section might still be too large,
        # so split it further if necessary.
        for piece, piece_start in split_long(body, start):

            # ⭐ IMPORTANT:
            #
            # The contextual header is included in the text
            # that gets embedded.
            #
            # Instead of embedding:
            #
            #   "It can be issued for up to 90 days."
            #
            # we embed:
            #
            #   "[Payments Guide > Refunds > Partial Refunds]"
            #   "It can be issued for up to 90 days."
            #
            # This gives the embedding useful context.
            embedded = f"[{full_path}]\n{piece}"

            # Store the chunk along with useful metadata.
            chunks.append(
                Chunk(
                    text=embedded,
                    raw_text=piece,
                    doc_id=doc_id,
                    section_path=full_path,
                    start_char=piece_start,
                    end_char=piece_start + len(piece),
                    token_count=n_tok(embedded),
                )
            )

    return chunks


# ---------------------------------------------------------
# Example
# ---------------------------------------------------------

if __name__ == "__main__":

    sample = """
# Payments Guide

## Refunds
Refunds are processed within 5 business days.

### Partial Refunds
It can be issued for up to 90 days after the original transaction.
Partial refunds require the original order ID.

## Troubleshooting
If a payment fails, check the card expiry first.
"""

    chunks = chunk_markdown(
        sample,
        doc_id="payments-guide",
        doc_title="Payments Guide",
    )

    # Print the chunks so we can inspect the result.
    for c in chunks:
        print(f"[{c.token_count:3d} tok] {c.section_path}")
        print(f"    {c.raw_text[:70]!r}")
        print(f"    embedded as: {c.text.splitlines()[0]}")
        print()


### Things to remember
# split_by_headings()
#         ↓
# Find the document's natural sections

# split_long()
#         ↓
# Break sections that are too large
# while maintaining some overlap

# chunk_markdown()
#         ↓
# Combine everything + add contextual headers
# and create Chunk objects