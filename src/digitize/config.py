import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal
from digitize.sentinel import _sentinel
from digitize import units
from digitize.stages import StageName
from digitize.units import Unit, UnitGroup, UnitMode, unit_modes

SIMPLE_REP_PATTERN = r"(?:time|occurence|instance)s?"
COMPLEX_REP_PATTERN = r"(?:time|occurence|instance|attempt|try|tries)s?(?: (?:at|of))?(?: (?:which|when))?"


DigitizeMode = Literal["default", "token", "strip", "num", "norm"]

@dataclass
class DigitizeParams:
    description: str = _sentinel
    config: DigitizeMode | any = _sentinel
    use_commas: bool = _sentinel
    fmt: str = _sentinel

    replace_multipliers: bool = _sentinel
    fmt_multipliers: str | None = _sentinel

    # Ordinals:
    support_ordinals: bool = _sentinel
    fmt_ordinal: str | None  = _sentinel     # one hundred seventy-second -> 172nd

    # reps / "time(s)":
    rep_signifiers: str | re.Pattern | Iterable[str | re.Pattern] = _sentinel
    support_reps: bool = _sentinel
    fmt_rep: str | None = _sentinel     # default "%nx"    -> 3x   (for "3 times", "twice")
    fmt_nth_time: str | None  = _sentinel    # default "%n%ox"  -> 500th time (for "500th time")
    rep_fmt: str = _sentinel
    rep_fmt_plural: bool = _sentinel

    attempt_to_differentiate_seconds: bool = _sentinel

    literal_fmt: bool | None  = _sentinel

    support_roman: bool = _sentinel

    parse_signs: bool = _sentinel

    power: str = _sentinel
    mult: str = _sentinel
    div: str = _sentinel

    combine_add: bool = _sentinel
    res: int | None = _sentinel
    do_simple_evals: bool = _sentinel
    do_fraction_evals: bool = _sentinel


    breaks: str | re.Pattern | Iterable[str | re.Pattern] = _sentinel
    units: Unit | UnitGroup | Iterable[Unit] = _sentinel
    unit_mode: UnitMode = _sentinel
    unit_max_cascade: Literal["base"] | int | None = _sentinel
    int_cascade_mode: bool = _sentinel

    skip_stages: Iterable[StageName] = _sentinel,
    log_stages: bool | Iterable[StageName] = _sentinel,
    log_context: bool | Iterable[StageName]  = _sentinel,

    def non_sentinels(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not _sentinel}

    def replace(self, **kwargs):
        return DigitizeParams(**{**self.__dict__, **kwargs})

    def __repr__(self):
        d = "\n".join(f"\t{k}={v if not isinstance(v, DigitizeParams) else v.config}," for k, v in self.non_sentinels().items())
        return f"DigitizeParams(\n{d}\n)"

    def finalize(self):
        self.breaks = [self.breaks] if isinstance(self.breaks, str) else list(self.breaks)
        if not self.literal_fmt:
            self.fmt = re.sub(r"\d+", "%n", self.fmt)
        if self.fmt_multipliers is None:
            self.fmt_multipliers = self.fmt
        if not self.literal_fmt:
            self.fmt_multipliers = re.sub(r"\d+", "%n", self.fmt_multipliers)
        if self.fmt_ordinal is None:
            self.fmt_ordinal = self.fmt.replace("%n", "%n%o")
        if not self.literal_fmt:
            self.fmt_ordinal = re.sub(r"\d+", "%n", self.fmt_ordinal)
        if self.fmt_rep is None:
            if self.rep_fmt == "x":
                self.fmt_rep = self.fmt.replace("%n", "%nx")
            else:
                self.fmt_rep = self.fmt.replace("%n", f"%n {self.rep_fmt}%s" if self.rep_fmt_plural else f"%n {self.rep_fmt}")
        if not self.literal_fmt:
            self.fmt_rep = re.sub(r"\d+", "%n", self.fmt_rep)
        if self.fmt_nth_time is None:
            self.fmt_nth_time = re.sub("%n(%o)?", rf"%n\1 {self.rep_fmt}", self.fmt_ordinal)
        if not self.literal_fmt:
            self.fmt_nth_time = re.sub(r"\d+", "%n", self.fmt_nth_time)

        if self.parse_signs:
            if not self.literal_fmt:
                self.fmt = self.fmt.replace("%n", "%p%n")
                self.fmt_multipliers = self.fmt_multipliers.replace("%n", "%p%n")
                self.fmt_ordinal = self.fmt_ordinal.replace("%n", "%p%n")
                self.fmt_rep = self.fmt_rep.replace("%n", "%p%n")
                self.fmt_nth_time = self.fmt_nth_time.replace("%n", "%p%n")
        
        if self.unit_mode == unit_modes.CLOSEST:
            self.unit_max_cascade = 1
        return self


default = DigitizeParams(
    description="Tries to respect human language. Pretty and semi-parseable",
    config="default",
    use_commas= False,
    fmt= "%n",
    replace_multipliers = True,
    fmt_multipliers=None,

    # Ordinals:
    support_ordinals = True,
    fmt_ordinal=None,     # one hundred seventy-second -> 172nd

    # reps / "time(s)":
    rep_signifiers = COMPLEX_REP_PATTERN,
    support_reps= True,
    fmt_rep=None,      # default "%nx"    -> 3x   (for "3 times", "twice")
    fmt_nth_time =None,    # default "%n%ox"  -> 500th time (for "500th time")
    rep_fmt= "time",
    rep_fmt_plural = True,

    attempt_to_differentiate_seconds = True,

    literal_fmt = False,

    support_roman = False,

    parse_signs = True,
    power="**",
    mult="*",
    div="/",
    combine_add=True,
    res=None,
    do_simple_evals=True,
    do_fraction_evals=True,
    breaks=(),
    units=(
        units.dozens,
        units.bakers_dozens,
        units.pi,
        units.pairs,
    ),
    unit_mode=unit_modes.BASE,
    unit_max_cascade="base",
    int_cascade_mode=True,
    skip_stages = (),
    log_stages = True,
    log_context = True,
)


units = default.replace(
    units = (
        units.dozens,
        units.bakers_dozens,
        units.pi,
        units.pairs,
        units.meters,
        units.seconds,
        units.feet,
        units.grams,
        units.hz,
        units.inches,
        units.yards,
        units.months,
    )
)

nomath = default.replace(
    combine_add=False,
    do_simple_evals=False,
    do_fraction_evals=False
)

simplemath = default.replace(
    combine_add=True,
    do_simple_evals=True,
    res=None,
    do_fraction_evals=False
)

token = default.replace(
    description="ugly but parseable",
    config="token",
    fmt="[NUM=%n,OG=%i]",
    fmt_multipliers="[NUM=%n,MULT=%m,OG=%i]",
    fmt_ordinal="[NUM=%n,ORD=%o,OG=%i]",
    fmt_rep="[NUM=%n,REP=%r,OG=%i]",
    fmt_nth_time="[NUM=%n,ORD=%o,REP=%r,OG=%i]",
)

strip = default.replace(
    description="simplifies the string a lot but very lossy of n-th n-th times, etc",
    config="strip",
    rep_signifiers=COMPLEX_REP_PATTERN,
    fmt_ordinal="%n",
    fmt_rep="%n",
    fmt_nth_time="%n",
)

nums = default.replace(
    description="do not even look for once, n times, etc.",
    config="num",
    support_reps=False,
    attempt_to_differentiate_seconds=False,
)

norm = default.replace(
    description="Not grammatically correct but more parseable. e.g. 1-th, 2-th, 3-th time, etc",
    config="norm",
    fmt_ordinal="%n-th",
    fmt_rep="%n-th time"
)



