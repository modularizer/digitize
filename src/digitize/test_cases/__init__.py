from .decimal import DECIMAL_TALES
from .edge import EDGE_CASE_CHRONICLES
from .examples import EXAMPLES
from .formatting import FORMATTING_TALES
from .math import MATH
from .more import MORE
from .ordinal_time import ORDINAL_AND_TIME_TALES
from .roman import ROMAN_TALES
from .sci import SCI
from .signed import SIGNED_TALES
from .stories import STORIES
from .suffix import SUFFIX_SAGAS
from .ths import THS

TESTS = [
    *EXAMPLES,
    *STORIES,
    *FORMATTING_TALES,
    *EDGE_CASE_CHRONICLES,
    *SUFFIX_SAGAS,
    *ORDINAL_AND_TIME_TALES,
    *ROMAN_TALES,
    *SIGNED_TALES,
    *DECIMAL_TALES,
    *THS,
    *MATH,
    *SCI,
    *MORE
]

def get_suite(name: str):
    name = name.lower()
    if name in {"tests", "test", "all"}:
        return TESTS
    if name in {"examples", "example"}:
        return EXAMPLES
    if name in {"stories", "story"}:
        return STORIES
    if name in {"formatting", "formatting_tales"}:
        return FORMATTING_TALES
    if name in {"edges", "edge", "edge_cases", "edge_case_chronicles"}:
        return EDGE_CASE_CHRONICLES
    if name in {"suffix", "suffixes", "suffix_sagas"}:
        return SUFFIX_SAGAS
    if name in {"ordinal", "ordinals", "ordinal_and_time", "ordinal_and_time_tales"}:
        return ORDINAL_AND_TIME_TALES
    if name in {"roman", "romans", "roman_tales"}:
        return ROMAN_TALES
    if name in {"signed", "signs", "signed_tales"}:
        return SIGNED_TALES
    if name in {"decimal", "decimals", "decimal_tales"}:
        return DECIMAL_TALES
    if name in {"ths"}:
        return THS
    if name in {"math"}:
        return MATH
    if name in {"sci"}:
        return SCI
    if name in {"more"}:
        return MORE
    raise ValueError(f"unknown suite: {name}")