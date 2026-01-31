import time
from dataclasses import dataclass, field
from typing import Any, Literal

from digitize.attrstr import attrstr
from digitize.sentinel import _sentinel

StageName = Literal[
    "input", "split by breaks", "repeat_sentinel_preprocessed",
    "attempt_to_differentiate_seconds", "replace_multipliers",
    "main loop", "re-sub seconds", "unit sub", "a set(s) of",
    "re-sub repeat map", "negative", "positive", "+- space",
    "divided by", "power-a", "power-b", "power-c", "power-d",
    "square root", "cubed root", "nth root",
    "sci-a", "sci-b", "sci-c",
    "and-a", "and-b", "of", "and-c",
    "simple-evals", "times", "simple-evals-2",
    "hanging-set", "merge-units", "merge-units-2",
    "simple-evals-3", "merge-units-b", "iter", "unit_merge", "full"
]

class Stage:
    INPUT: StageName = "input"
    SPLIT_BY_BREAKS: StageName = "split by breaks"
    REPEAT_SENTINEL_PREPROCESSED: StageName = "repeat_sentinel_preprocessed"
    ATTEMPT_TO_DIFFERENTIATE_SECONDS: StageName = "attempt_to_differentiate_seconds"
    REPLACE_MULTIPLIERS: StageName = "replace_multipliers"
    MAIN_LOOP: StageName = "main loop"
    RESUB_SECONDS: StageName = "re-sub seconds"
    UNIT_SUB: StageName = "unit sub"
    A_SET_OF: StageName = "a set(s) of"
    RESUB_REPEAT_MAP: StageName = "re-sub repeat map"
    NEGATIVE: StageName = "negative"
    POSITIVE: StageName = "positive"
    PLUS_MINUS_SPACE: StageName = "+- space"
    DIVIDED_BY: StageName = "divided by"
    POWER_A: StageName = "power-a"
    POWER_B: StageName = "power-b"
    POWER_C: StageName = "power-c"
    POWER_D: StageName = "power-d"
    SQUARE_ROOT: StageName = "square root"
    CUBED_ROOT: StageName = "cubed root"
    NTH_ROOT: StageName = "nth root"
    SCI_A: StageName = "sci-a"
    SCI_B: StageName = "sci-b"
    SCI_C: StageName = "sci-c"
    AND_A: StageName = "and-a"
    AND_B: StageName = "and-b"
    OF: StageName = "of"
    AND_C: StageName = "and-c"
    SIMPLE_EVALS: StageName = "simple-evals"
    TIMES: StageName = "times"
    SIMPLE_EVALS_2: StageName = "simple-evals-2"
    HANGING_SET: StageName = "hanging-set"
    MERGE_UNITS: StageName = "merge-units"
    MERGE_UNITS_2: StageName = "merge-units-2"
    SIMPLE_EVALS_3: StageName = "simple-evals-3"
    MERGE_UNITS_B: StageName = "merge-units-b"
    ITER: StageName = "iter"
    UNIT_MERGE: StageName = "unit_merge"
    FULL: StageName = "full"




@dataclass
class StageResult:
    stage: StageName
    new: str | None = None
    changed: bool = True
    prev: "StageResult | None | str" = None
    ctx: Any = None
    skipped: bool = False
    call_level: int | None = 0
    stages: list["StageResult"] = field(default_factory=list)
    started_at: float | None = field(default_factory=time.perf_counter)
    finished_at: float | None = None
    log: bool = False
    log_context: bool = False

    def __post_init__(self):
        if isinstance(self.prev, str):
            self.prev = StageResult(stage=Stage.INPUT, new=self.prev)

    @property
    def elapsed(self) -> float | None:
        if self.finished_at is None or self.started_at is None:
            return None
        return self.finished_at - self.started_at

    @property
    def ms(self):
        return self.elapsed * 1000 if self.elapsed is not None else None

    @property
    def us(self):
        return self.elapsed * 1e6 if self.elapsed is not None else None

    def get_explanation(self):
        ctx = (self.stage + ((f"({self.ctx!r})" if self.ctx else "") if self.log_context is True or (self.log_context and self.stage in self.log_context) else ""))
        if ctx:
            ctx += ": "
        n = f"'{self.new}'" if isinstance(self.new, str) else self.new
        prefix = "\t" * self.call_level
        return f"{prefix}{ctx}'{self.old}' -> {n}"

    @property
    def content(self):
        return self.new if isinstance(self.new, str) else self.old

    @property
    def old(self):
        return self.prev.content if isinstance(self.prev, StageResult) else ""

    def __str__(self):
        return self.content

    def __repr__(self):
        return f"StageResult(stage={self.stage!r}, new={self.new!r}, changed={self.changed}, ...)"


    def precursors(self, max_levels: int | None = None):
        prevs = [self.prev] if self.prev is not None else []
        if isinstance(self.prev, StageResult) and (max_levels is None or max_levels > 1):
            prevs.extend(self.prev.precursors(max_levels - 1 if max_levels is not None else None))
        return prevs

    @property
    def history(self):
        return list(reversed(self.precursors())) + [self]

    @property
    def changes(self):
        return [x for x in self.stages if x.changed]

    def __iter__(self):
        return iter(self.history)

    def __index__(self, n: int | slice | None = None):
        # self[-1] is self.prev
        # self[-2] is self.prev.prev, etc.
        if n is None:
            return self.precursors()
        if isinstance(n, int) and n < 0:
            return self.precursors(-1 * n)[0]
        return self.history[n]

    def __call__(self, *args, **kwargs):
        return self.finish(*args, **kwargs)

    def finish(self, new: str | None = _sentinel, ctx=_sentinel, **kwargs):
        self.finished_at = time.perf_counter()
        if new is not _sentinel:
            self.new = new
        if ctx is not _sentinel:
            self.ctx = ctx
        self.changed = self.new != self.old
        self.__dict__.update(kwargs)
        if self.log:
            print(self.get_explanation())
        return self.result

    def skip(self, **kwargs):
        self.skipped = True
        self.finished_at = time.perf_counter()
        self.changed = False
        self.__dict__.update(kwargs)
        return self.result

    @property
    def result(self):
        return attrstr(self.content, result=self)
