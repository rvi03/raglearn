"""Token-aware recursive text splitting.

The shared fallback for both chunking strategies: when a structure- or
semantic-derived chunk overruns the embedder's token budget, it must be broken
into pieces that fit, because the embedder silently truncates anything past its
limit (a dropped figure in a filing is a correctness bug, not a formatting one).

The split descends a ladder of separators from coarse to fine -- paragraph,
line, sentence, word, and finally the character -- cutting at the coarsest
boundary that brings every piece within budget. The character rung is the
termination guarantee: a run with no whitespace or punctuation (a long base64
blob, a malformed table) still terminates because single characters always fit.

Sizing is measured in tokens, not characters, via an injected counter, so the
budget matches what the embedder actually sees. Injecting the counter keeps this
module free of any tokenizer or model dependency and makes it testable with a
trivial stand-in.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

# Separators tried in order, coarsest first. The trailing empty string is the
# character rung -- the terminal fallback that guarantees termination when no
# textual boundary exists in an oversized run.
_DEFAULT_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", " ", "")

TokenCounter = Callable[[str], int]


def recursive_split(
    text: str,
    *,
    max_tokens: int,
    overlap_tokens: int = 0,
    count_tokens: TokenCounter,
    separators: Sequence[str] = _DEFAULT_SEPARATORS,
) -> list[str]:
    """Split text into chunks that each fit within a token budget.

    Args:
      text: The text to split.
      max_tokens: The largest a returned chunk may be, in tokens. Must be >= 1.
      overlap_tokens: Tokens of trailing context to repeat at the start of each
        chunk after the first, so a boundary does not sever a fact from the
        context that explains it. Must be < ``max_tokens``.
      count_tokens: Returns the token count of a string, aligned to the
        embedder's tokenizer.
      separators: Boundary ladder, coarsest first; the last should be ``""``
        (the character rung) to guarantee termination on unbreakable runs.

    Returns:
      The chunks in document order. Empty or whitespace-only input yields an
      empty list. Every chunk fits ``max_tokens`` whenever the ladder ends in
      the character rung.

    Raises:
      ValueError: If ``max_tokens`` < 1 or ``overlap_tokens`` is out of range.
    """
    if max_tokens < 1:
        raise ValueError(f"max_tokens must be >= 1, got {max_tokens}")
    if not 0 <= overlap_tokens < max_tokens:
        raise ValueError(
            f"overlap_tokens must be in [0, max_tokens), got {overlap_tokens} (max {max_tokens})"
        )
    if not text.strip():
        return []
    atoms = _atomize(text, tuple(separators), max_tokens, count_tokens)
    return _pack(atoms, max_tokens, overlap_tokens, count_tokens)


def _atomize(
    text: str,
    separators: tuple[str, ...],
    max_tokens: int,
    count_tokens: TokenCounter,
) -> list[str]:
    """Break text into ordered pieces that each fit ``max_tokens``.

    Cuts on the first separator present in the text; any piece still over budget
    recurses onto the remaining, finer separators. The character rung
    (``separators`` ending in ``""``) guarantees every piece eventually fits.
    """
    if count_tokens(text) <= max_tokens:
        stripped = text.strip()
        return [stripped] if stripped else []

    separator, finer = _pick_separator(text, separators)
    pieces = list(text) if separator == "" else text.split(separator)

    atoms: list[str] = []
    for piece in pieces:
        if not piece.strip():
            continue
        if count_tokens(piece) <= max_tokens:
            atoms.append(piece.strip())
        else:
            atoms.extend(_atomize(piece, finer, max_tokens, count_tokens))
    return atoms


def _pick_separator(text: str, separators: tuple[str, ...]) -> tuple[str, tuple[str, ...]]:
    """Return the coarsest separator present in ``text`` and the finer ones left.

    The character rung (``""``) is always considered present, so this is total:
    an oversized run with no textual boundary still resolves to a splittable
    separator.
    """
    for index, separator in enumerate(separators):
        if separator == "" or separator in text:
            return separator, separators[index + 1 :]
    # No ladder ending in the character rung -- fall back to it so callers that
    # pass a custom ladder without one still terminate.
    return "", ()


def _pack(
    atoms: list[str],
    max_tokens: int,
    overlap_tokens: int,
    count_tokens: TokenCounter,
) -> list[str]:
    """Greedily merge ordered atoms into chunks, repeating overlap between them.

    Atoms are joined with a single space: the recursive descent already cut at
    the most meaningful boundary available, so the fallback's job is only to
    keep pieces whole and within budget, not to preserve the original
    whitespace. Each chunk after the first is seeded with the trailing atoms of
    the previous chunk whose tokens sum to at most ``overlap_tokens``.
    """
    chunks: list[str] = []
    current: list[str] = []
    for atom in atoms:
        candidate = [*current, atom]
        if current and count_tokens(" ".join(candidate)) > max_tokens:
            chunks.append(" ".join(current))
            current = [*_overlap_tail(current, overlap_tokens, count_tokens), atom]
        else:
            current = candidate
    if current:
        chunks.append(" ".join(current))
    return chunks


def _overlap_tail(
    atoms: list[str],
    overlap_tokens: int,
    count_tokens: TokenCounter,
) -> list[str]:
    """Return the trailing atoms whose token sum stays within ``overlap_tokens``."""
    if overlap_tokens <= 0:
        return []
    tail: list[str] = []
    total = 0
    for atom in reversed(atoms):
        total += count_tokens(atom)
        if total > overlap_tokens:
            break
        tail.insert(0, atom)
    return tail
