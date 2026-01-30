import re
from typing import Iterable


def chunk_string(
        s: str,
        breaks: str | re.Pattern | Iterable[str | re.Pattern] | None
):
    if not breaks:
        return [s], []

    # Normalize breaks into regex strings
    if isinstance(breaks, (str, re.Pattern)):
        breaks = [breaks]

    patterns = []
    for b in breaks:
        if isinstance(b, re.Pattern):
            patterns.append(b.pattern)
        else:
            patterns.append(re.escape(b))

    pattern = f"({'|'.join(patterns)})"
    parts = re.split(pattern, s)

    chunks = parts[::2]
    seps = parts[1::2]
    return chunks, seps

def merge_chunks(chunks: Iterable[str], seps: Iterable[str]):
    out = []
    for i, c in enumerate(chunks):
        out.append(c)
        if i < len(seps):
            out.append(seps[i])

    return "".join(out)


def alternate(groups: str | re.Pattern | Iterable[str | re.Pattern] | None):
    if not groups:
        return ""
    if isinstance(groups, re.Pattern):
        return groups.pattern
    if isinstance(groups, str):
        return groups
    return "|".join(p.pattern if isinstance(p, re.Pattern) else p for p in groups)

def word(pat: str | re.Pattern):
    if not pat:
        return ""
    return fr"\b{pat.pattern if isinstance(pat, re.Pattern) else pat}\b"

def any_word(groups: str | re.Pattern | Iterable[str | re.Pattern] | None):
    return word(alternate(groups))

def group(p: str, capturing=True):
    return f"({p})" if capturing else f"(?:{p})"