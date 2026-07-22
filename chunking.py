"""
Sentence-aware text chunking — shared between bert_agent.py (chunking is a
necessity there, to stay under the model's hard context limit) and
llm_backend.py (chunking is a quality choice there, to keep each
quasi-identifier detection call's scope narrow — see llm_backend.py's
_build_prompt for why a smaller, focused span catches things a full-document
sweep can miss).
"""

import re

# Titles/abbreviations whose trailing period must not be read as a sentence
# end (mirrors the set of things that legitimately precede a name — see
# bert_agent's own _is_mergeable_gap for the "Dr. Namn" merge case).
_ABBREVIATIONS = {"dr", "prof", "fil", "med", "jur", "ex", "etc", "kl", "nr", "sid"}


def split_into_units(text: str) -> list[tuple[int, int]]:
    """
    Split text into small units that are always safe to cut between: never
    inside a line, and never inside a sentence (skipping periods that look
    like abbreviations rather than sentence ends). Chunking groups these
    back up to a size budget without ever splitting one in half, so a chunk
    boundary can't land mid-entity or strip the context immediately around one.
    """
    units: list[tuple[int, int]] = []
    for line_match in re.finditer(r"[^\n]*\n?", text):
        line_start, line_end = line_match.start(), line_match.end()
        if line_start == line_end:
            continue
        line = text[line_start:line_end]

        sent_start = 0
        for m in re.finditer(r"[.!?]+(?=\s|$)", line):
            end = m.end()
            preceding = re.search(r"(\w+)\.?$", line[sent_start:m.start()])
            word = preceding.group(1).lower() if preceding else ""
            if word in _ABBREVIATIONS:
                continue  # likely "Dr." etc — not a real sentence end
            units.append((line_start + sent_start, line_start + end))
            sent_start = end

        if sent_start < len(line):
            units.append((line_start + sent_start, line_end))

    return [u for u in units if u[1] > u[0]]


def chunk_by_sentences(text: str, max_chunk_chars: int) -> list[tuple[int, int]]:
    """Group line/sentence units into chunks up to max_chunk_chars, never splitting a unit."""
    chunks: list[tuple[int, int]] = []
    chunk_start = None
    chunk_end = None
    for start, end in split_into_units(text):
        if chunk_start is None:
            chunk_start, chunk_end = start, end
        elif end - chunk_start <= max_chunk_chars:
            chunk_end = end
        else:
            chunks.append((chunk_start, chunk_end))
            chunk_start, chunk_end = start, end
    if chunk_start is not None:
        chunks.append((chunk_start, chunk_end))
    return chunks
