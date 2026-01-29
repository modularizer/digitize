#!/usr/bin/env python3
import argparse
import ast
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction
from typing import Tuple, Literal, Union, TypedDict, Optional
from decimal import Decimal, localcontext, ROUND_HALF_UP


class Sentinel:
    def __bool__(self):
        return False
    def __repr__(self):
        return ""
    def __str__(self):
        return ""

_sentinel = Sentinel()

SIMPLE_REP_PATTERN = r"(?:time|occurence|instance)s?"
COMPLEX_REP_PATTERN = r"(?:time|occurence|instance|attempt|try|tries)s?(?: (?:at|of))?(?: (?:which|when))?"



def guess_plural(word: str, known: dict[str, str] = None, skip: list[str] = None) -> str:
    if not word.strip() or not word[-1].strip():
        return word
    know_mapping = {
        "child": "children",
        "person": "people",
        "man": "men",
        "woman": "women",
        "Hz": "Hz",
        **(known or {}),
        **({x: x for x in (skip or [])})
    }
    if know_mapping.get(word, None) is not None:
        return know_mapping[word]
    if word.isupper():
        return word
    if len(word) == 1:
        return word
    if re.search(r"[aeiou]y$", word):
        return word + "s"          # day → days
    if re.search(r"y$", word):
        return word[:-1] + "ies"   # city → cities
    if re.search(r"(s|x|z|ch|sh)$", word):
        return word + "es"         # box → boxes
    if re.search(r"fe$", word):
        return word[:-2] + "ves"   # knife → knives
    if re.search(r"f$", word):
        return word[:-1] + "ves"   # leaf → leaves (best guess)
    return word + "s"

@dataclass
class UnitGrammar:
    key: str
    s: Optional[bool] = _sentinel # means the plural is just + 's'
    plural: Optional[str] = _sentinel

    def __post_init__(self):
        if not isinstance(self.key, str):
            raise TypeError("key must be a string")
        if self.plural is _sentinel or not self.plural:
            if self.s is _sentinel:
                self.plural = guess_plural(self.key)
            elif self.s:
                self.plural = self.key + "s"
            else:
                self.plural = self.key

    def __repr__(self):
        return f"UnitGrammar(key={self.key}, plural={self.plural})"

    def __str__(self):
        return self.key



@dataclass
class UnitValue:
    value: int | str = 1
    new_unit: Optional["str | UnitGrammar | Unit"]  = "" # converts to a new unit
    new_unit_plural: Optional[str] = ""
    recurse: bool = False # adds new_unit to list of units
    replacement: Optional[str] = "set(s) of %n%u"

    def __repr__(self):
        return f"UnitValue({self.value}, new_unit={self.new_unit})"

    def __str__(self):
        if self.value in ("1", "one", 1, "a", "the"):
            p = self.new_unit.children[0].key if isinstance(self.new_unit, Unit) else self.new_unit
        else:
            p = self.new_unit.children[0].plural if isinstance(self.new_unit, Unit) else self.new_unit
        u = f" {p}"
        return f"{self.value} {u}"


@dataclass
class Unit(UnitValue, UnitGrammar):
    key: str | Iterable[str]
    value: int | str | UnitValue = 1
    base: bool = False
    new_unit: Optional["str | UnitGrammar | Unit"]  = _sentinel # converts to a new unit
    new_unit_plural: Optional[str] = _sentinel
    recurse: bool = _sentinel # adds new_unit to list of units
    replacement: Optional[str] = _sentinel
    s: Optional[bool] = _sentinel # means the plural is just + 's'
    plural: Optional[str] = _sentinel
    pattern: Optional[str | re.Pattern] = _sentinel
    full_pattern: Optional[str | re.Pattern] = _sentinel
    children: Iterable["Unit"] = _sentinel

    @property
    def is_group(self):
        return not isinstance(self.key, str)

    def __repr__(self):
        return f"Unit(key={self.key}, value={self.value}, ...)"

    def __str__(self):
        return str(self.key)


    def __post_init__(self):
        if self.children is _sentinel:
            self.children = [self]
        self.children = list(self.children)
        if not self.key:
            return
        if isinstance(self.key, str):
            if not isinstance(self.key, str):
                raise TypeError("key must be a string")
            if self.plural is _sentinel or not self.plural:
                if self.s is _sentinel:
                    self.plural = guess_plural(self.key)
                elif self.s:
                    self.plural = self.key + "s"
                else:
                    self.plural = self.key
        if not isinstance(self.value, UnitValue):
            self.value = UnitValue(
                value=self.value,
                new_unit=self.new_unit or "",
                new_unit_plural=self.new_unit_plural or "",
                recurse=self.recurse or False,
                replacement=self.replacement if self.replacement is not _sentinel else "set(s) of %n%u"
            )
        if self.new_unit is _sentinel:
            self.new_unit = self.value.new_unit
        if self.new_unit_plural is _sentinel:
            self.new_unit_plural = self.new_unit.plural if isinstance(self.new_unit, Unit) else ""
        if self.recurse is _sentinel:
            self.recurse = self.value.recurse or False
        if self.replacement is _sentinel:
            self.replacement = self.value.replacement
        self.value = self.value.value

        if isinstance(self.new_unit, str):
            self.new_unit = UnitGrammar(key=self.new_unit, plural=self.new_unit_plural)

        if not self.is_group:
            if self.pattern is _sentinel or not self.pattern:
                if self.plural == self.key:
                    self.pattern  = re.escape(self.key)
                else:
                    shared = ""
                    for i, s in enumerate(self.key):
                        if self.plural[i] == s:
                            shared += s
                    key_end = self.key[len(shared):]
                    plural_end = self.plural[len(shared):]
                    parts = [ v for v in [key_end, plural_end] if v]
                    if parts:
                        opts = "|".join(re.escape(p) for p in parts)
                        q = '?' if len(parts) == 1 else ''
                        self.pattern = rf"{re.escape(shared)}(?:{opts}){q}"
                    else:
                        self.pattern = rf"{re.escape(shared)}"

            if isinstance(self.pattern, str):
                self.pattern = re.compile(self.pattern.replace("%k", self.key))

            if self.full_pattern is _sentinel or not self.full_pattern:
                p = self.pattern.pattern if isinstance(self.pattern, re.Pattern) else self.pattern
                f = self.pattern.flags if isinstance(self.pattern, re.Pattern) else 0
                self.full_pattern = re.compile(
                    rf"(?:(\s)|(?<![A-Za-z])){p}(?![A-Za-z])",
                    flags=f
                )


            if isinstance(self.full_pattern, str):
                self.full_pattern = re.compile(self.full_pattern.replace("%k", self.key))

            self.children = [self]
        else:
            children = []
            for _k in self.key:
                kw = {**self.__dict__, "children": (), "key": _k}
                children.extend(Unit(**kw).children)
            self.children = children

            opts = [self.key] if isinstance(self.key, str) else list(self.key)
            opts += [guess_plural(k) for k in opts]
            opts = set(opts)
            p = "|".join(re.escape(v) for v in opts)
            self.pattern=p


        self.children: list[Unit]
        if self.recurse and isinstance(self.new_unit, Unit):
            self.children.extend(self.new_unit.children)

        if self.base:
            self.full_pattern = None

        if self.replacement is _sentinel:
            self.replacement = "set of %n%u"

    def suffix(self):
        if self.new_unit and self.new_unit is not _sentinel:
            nu = self.new_unit.children[0] if isinstance(self.new_unit, Unit) else self.new_unit
            if self.value in ("1", "one", 1, "a", "the"):
                p = nu.key
            else:
                p = nu.plural
            return p
        else:
            return ""

    @property
    def full_replacement(self):
        if not self.replacement:
            return None
        if self.base:
            p = self.key if isinstance(self.key, str) else self.key[0]
            return rf"\1{p}" if p else ""
        if self.new_unit and self.new_unit is not _sentinel:
            p = self.suffix()
            u = rf"\1{p}" if p else ""
        else:
            u = ""
        n = str(self.value)
        p = r"\1" + self.replacement.replace("%n", n).replace("%u", u)
        return p

    def sub(self, s, all_units: list["Unit"] = None):
        all_units = all_units or ()
        if not self.key:
            return s
        if not s:
            return
        if self.is_group:
            for child in self.children:
                s = child.sub(s, all_units)
            return s


        pat = self.pattern.pattern if isinstance(self.pattern, re.Pattern) else self.pattern
        if x := re.match( rf"(.*)(?:(\s)|(?<![A-Za-z]))(an?|\d) {pat} and (?:an? )?(\d+(?:\/|\.)\d+)(.*)", s):
            # this ONLY is a match if the thing that follows is NOT a unit (unless it it the same unit
            start = x.group(1)
            s = x.group(2) or ""
            n = 1 if x.group(3).startswith("a") else int(x.group(1))
            f = x.group(4)
            rest = x.group(5)
            # print(f"{n=}, {f=}, {rest=}")
            rm = re.match(r"(\S+)", rest.strip())
            fw = rm.group(1) if rm else ""
            rest2 = self.sub(rest, all_units)
            rm2 = re.match(r"(\S+)", rest2.strip())
            fw2 = rm2.group(1) if rm2 else ""
            if fw == fw2:
                if "/" in f:
                    nums, dens = f.split("/")
                    num = int(nums.strip())
                    den = int(dens.strip())
                    f = f"{(n*den + num)}/{den}"
                elif "." in f:
                    starts, ends = f.split(".")
                    start = int(starts.strip())
                    end = int(ends.strip())
                    f = f"{start + n}.{end}"
                else:
                    f = str(n + int(f))
                rep = self.full_replacement.replace(r"\1", " ")
                if self.base and f != "1":
                    k = self.key if isinstance(self.key, str) else self.key[0]
                    p = self.plural if self.plural else guess_plural(k)
                    rep = f" {p}"
                s = start + s + f + rep + rest2
        if not self.base:
            s = re.sub(self.full_pattern, self.full_replacement, s)
        return s


    def base_merge2(self, s):
        if not self.key or not self.base or not self.pattern:
            return s
        pat = self.pattern.pattern if isinstance(self.pattern, re.Pattern) else self.pattern
        n = r"(?:-?\d+(?:\.\d+)?|-?\d+/\d+|an?)"


        p1 = rf"({n})\s?({pat})\s?(?:less|fewer) (?:than|then) ({n})\s?({pat})"
        p2 = rf"({n})\s?({pat})\s?(?:more|greater) (?:than|then) ({n})\s?({pat})"
        s = re.sub(p1, r"(\3 - \1) \2", s)
        s = re.sub(p2, r"(\3 + \1) \2", s)
        return s

    def base_merge(self, s):
        if not self.key or not self.base or not self.pattern:
            return s
        pat = self.pattern.pattern if isinstance(self.pattern, re.Pattern) else self.pattern
        n = r"(?:(?:-?\d+(?:\.|\/\d+)?)|(?:an?))"

        # print(self.key, pat)
        p = rf"({n})(\s?{pat})\s?(?:and )?({n})\s?({pat})"

        # print("P", p, "S", s)
        def repl(m):
            def norm(x):
                return "1" if x in ("a", "an") else x
            left = norm(m.group(1))
            right = norm(m.group(3))
            unit = m.group(2)
            return f"({left} + {right}){unit}"
        # print(self.key, p, s)
        return re.sub(p, repl, s)





PI_DECIMAL = "3.14159265358979323846264338327950288419716939937510"


@dataclass
class UnitGroup:
    base: Unit | str | Iterable[str]
    names: dict[int, str|Iterable[str]]
    children: Iterable[Unit] = ()

    def __post_init__(self):
        base_keys = []
        if isinstance(self.base, str):
            base_keys = [self.base]
        elif isinstance(self.base, Iterable):
            base_keys = [b.children.key if hasattr(b, "children") else b for b in self.base]
        elif isinstance(self.base, Unit):
            base_keys = [b.key for b in self.base.children]
        else:
            print(f"base:{self.base}")
            raise TypeError("invalid base")

        base_key = base_keys[0]
        self.base = Unit(key=base_key, base=True)

        children = []
        children.extend([Unit(key=k, value=1, new_unit=self.base) for k in base_keys[1:]])
        for v, k in self.names.items():
            children.append(Unit(key=k, value=v, new_unit=self.base))


        self.children = children

    def __repr__(self):
        return f"UnitGroup({self.base})"

MetricPrefix = Literal["p", "n", "u", "m", "k", "M", "G", "T"]
small_metric = "pnum"
big_metric = "kMGT"
all_metric = small_metric + big_metric
def build_metric_prefixes(group: Iterable[str], prefixes: Iterable[MetricPrefix] = all_metric, extra: dict[str | int | float, str | Iterable[str]] = None) -> dict[int, Iterable[str]]:
    group = list(group)
    mapping = {
        "p": [0.000000000001, ("p", "pico")],
        "n": [0.000000001, ("n", "nano")],
        "u": [0.000001, ("μ", "micro")],
        "m": [0.001, ("m", "milli")],
        "k": [1000, ("k", "kilo")],
        "M": [10**6, ("M", "mega")],
        "G": [10**9, ("G", "giga")],
        "T": [10**12, ("T", "tera")]
    }
    o = {
        mapping[p][0]: tuple(p2+group[i2] for i2, p2 in enumerate(mapping[p][1]))
        for i, p in enumerate(prefixes)
    }
    if extra:
        for k, v in extra.items():
            if isinstance(k, str):
                k = mapping[k][0]
            old = list(o.get(k, []))
            o[k] = old + ([v] if isinstance(v, str) else list(v))
            # print(f"o[{k}] = {o[k]}")
    return o


def metric(
        base: Iterable[str],
        prefixes: Iterable[MetricPrefix] = all_metric,
        extra: dict[str | int | float, str | Iterable[str]] = None
):
    return UnitGroup(
        base,
        build_metric_prefixes(base, prefixes, extra)
    )


class units:
    pi = Unit(key=("pi", "PI", "math.pi", "π"), value=PI_DECIMAL)
    dozens = Unit(key="dozen", value=12)
    bakers_dozens = Unit(key=("baker's dozen", "bakers dozen"), value=13)
    pairs = Unit(key="pair", value=2, pattern="pairs?(?: of)?")
    grams = metric(("g", "gram"), all_metric, {1000: "kilo"})
    meters = metric(("m", "meter"))
    hz = metric(("Hz", "hz"), big_metric)
    seconds = metric(("s", "second"), small_metric,  {
            60: ("m", "min", "minute"),
            60*60: ("h", "hr", "hour"),
            24*60*60: ("d", "day"),
            7*24*60*60: ("w", "wk", "week")
        }
    )
    minutes = UnitGroup(("min", "minute"), {
        60: ("h", "hour", "hour"),
        24*60: ("d", "day"),
        7*24*60: ("w", "wk", "week")
    })
    hours= UnitGroup(("h", "hr", "hour"), {
        24: ("d", "day"),
        7*24: ("w", "wk", "week")
    })
    days = UnitGroup(("d", "day"), {
        7*24: ("w", "wk", "week")
    })
    months = UnitGroup(("mo", "month"), {3: "season", 12: ("y", "yr", "year"), 120: "decade", 1200: "century", 12000: "eon"})
    years = UnitGroup(("y", "yr", "year"), {10: "decade", 100: "century", 1000: "eon"})
    inches = UnitGroup("inch", {
        12: ("ft", "foot"),
        12*3: ("yd", "yard"),
        12*5280: ("mi", "mile")
    })
    feet = UnitGroup(("ft", "foot"), {
        3: ("yd", "yard"),
        5280: ("mi", "mile")
    })
    yards = UnitGroup(("yd", "yard"), {
        1760: ("mi", "mile")
    })



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


    breaks: Iterable[str] = _sentinel
    units: Unit | UnitGroup | Iterable[Unit] = _sentinel

    def non_sentinels(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not _sentinel}

    def replace(self, **kwargs):
        return DigitizeParams(**{**self.__dict__, **kwargs})

    def __repr__(self):
        d = "\n".join(f"\t{k}={v if not isinstance(v, DigitizeParams) else v.config}," for k, v in self.non_sentinels().items())
        return f"DigitizeParams(\n{d}\n)"


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
        )
    )



class modes:
    default = default
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




def digitize(
    s: str,
        *,
    config: DigitizeMode  | DigitizeParams = default,
    use_commas: bool = _sentinel,
    fmt: str = _sentinel,
    replace_multipliers: bool = _sentinel,
    fmt_multipliers: str = _sentinel,

    # Ordinals:
    support_ordinals: bool = _sentinel,
    fmt_ordinal: str | None  = _sentinel,     # one hundred seventy-second -> 172nd

    # reps / "time(s)":
    rep_signifiers: str | re.Pattern | Iterable[str | re.Pattern] = _sentinel,
    support_reps: bool = _sentinel,
    fmt_rep: str | None = _sentinel,     # default "%nx"    -> 3x   (for "3 times", "twice")
    fmt_nth_time: str | None  = _sentinel,    # default "%n%ox"  -> 500th time (for "500th time")
    rep_fmt: str = _sentinel,
    rep_fmt_plural: bool = _sentinel,

    attempt_to_differentiate_seconds: bool = _sentinel,

    literal_fmt: bool | None  = _sentinel,
    
    support_roman: bool = _sentinel,
    
    parse_signs: bool = _sentinel,
        power: str = _sentinel,
    mult: str = _sentinel,
    div: str = _sentinel,
        combine_add: bool = _sentinel,
        res: int = _sentinel,
        do_simple_evals: bool = _sentinel,
        do_fraction_evals: bool = _sentinel,
        breaks: str | Iterable[str] = _sentinel,
        units: Unit | UnitGroup | Iterable[Unit] = _sentinel,
        _iter: bool = True

) -> str:
    if not s.strip():
        return s
    params = DigitizeParams(
        use_commas=use_commas,
        fmt=fmt,
        replace_multipliers=replace_multipliers,
        support_ordinals=support_ordinals,
        fmt_ordinal=fmt_ordinal,
        rep_signifiers=rep_signifiers,
        support_reps=support_reps,
        fmt_rep=fmt_rep,
        fmt_nth_time=fmt_nth_time,
        rep_fmt=rep_fmt,
        rep_fmt_plural=rep_fmt_plural,
        attempt_to_differentiate_seconds=attempt_to_differentiate_seconds,
        literal_fmt=literal_fmt,
        support_roman=support_roman,
        parse_signs=parse_signs,
        power=power,
        mult=mult,
        div=div,
        combine_add=combine_add,
        res=res,
        do_simple_evals=do_simple_evals,
        do_fraction_evals=do_fraction_evals,
        breaks=breaks,
        units=units
    )
    defaults = config if isinstance(config, DigitizeParams) else getattr(modes, config)
    config = DigitizeParams(**{**defaults.non_sentinels(), **params.non_sentinels()})
    description=config.description
    use_commas=config.use_commas
    fmt=config.fmt
    replace_multipliers = config.replace_multipliers
    fmt_multipliers = config.fmt_multipliers
    support_ordinals = config.support_ordinals
    fmt_ordinal=config.fmt_ordinal     # one hundred seventy-second -> 172nd
    rep_signifiers = config.rep_signifiers
    support_reps= config.support_reps
    fmt_rep=config.fmt_rep      # default "%nx"    -> 3x   (for "3 times", "twice")
    fmt_nth_time =config.fmt_nth_time    # default "%n%ox"  -> 500th time (for "500th time")
    rep_fmt= config.rep_fmt
    rep_fmt_plural = config.rep_fmt_plural
    attempt_to_differentiate_seconds = config.attempt_to_differentiate_seconds
    literal_fmt = config.literal_fmt
    support_roman = config.support_roman
    parse_signs = config.parse_signs
    mult = config.mult
    div = config.div
    power = config.power
    combine_add = config.combine_add
    res = config.res
    do_simple_evals = config.do_simple_evals
    do_fraction_evals = config.do_fraction_evals
    breaks = config.breaks
    units=config.units
    #__________________________________________________
    if isinstance(breaks, str):
        breaks = (breaks,)

    if not breaks:
        chunks = [s]
        seps = []
    else:
        # split and keep delimiters
        pattern = f"({'|'.join(map(re.escape, breaks))})"
        parts = re.split(pattern, s)

        chunks = parts[::2]   # text
        seps   = parts[1::2]  # separators

    if len(chunks) > 1:
        processed = [digitize(c, config=config) for c in chunks]

        out = []
        for i, c in enumerate(processed):
            out.append(c)
            if i < len(seps):
                out.append(seps[i])

        return "".join(out)

    # _________________________________________
    if not literal_fmt:
        fmt = re.sub(r"\d+", "%n", fmt)
    if fmt_multipliers is None:
        fmt_multipliers = fmt
    if not literal_fmt:
        fmt_multipliers = re.sub(r"\d+", "%n", fmt_multipliers)
    if fmt_ordinal is None:
        fmt_ordinal = fmt.replace("%n", "%n%o")
    if not literal_fmt:
        fmt_ordinal = re.sub(r"\d+", "%n", fmt_ordinal)
    if fmt_rep is None:
        if rep_fmt == "x":
            fmt_rep = fmt.replace("%n", "%nx")
        else:
            fmt_rep = fmt.replace("%n", f"%n {rep_fmt}%s" if rep_fmt_plural else f"%n {rep_fmt}")
    if not literal_fmt:
        fmt_rep = re.sub(r"\d+", "%n", fmt_rep)
    if fmt_nth_time is None:
        fmt_nth_time = re.sub("%n(%o)?", rf"%n\1 {rep_fmt}", fmt_ordinal)
    if not literal_fmt:
        fmt_nth_time = re.sub(r"\d+", "%n", fmt_nth_time)
        
    if parse_signs:
        if not literal_fmt:
            fmt = fmt.replace("%n", "%p%n")
            fmt_multipliers = fmt_multipliers.replace("%n", "%p%n")
            fmt_ordinal = fmt_ordinal.replace("%n", "%p%n")
            fmt_rep = fmt_rep.replace("%n", "%p%n")
            fmt_nth_time = fmt_nth_time.replace("%n", "%p%n")

    _SEC__sentinel = "__DIGITIZE_ECOND_UNIT__"
    _REPEAT_PREFIX = "__repeat__"
    _repeat_map: dict[str, str] = {}

    if support_reps and rep_signifiers:
        # Make a single alternation regex. Each signifier is treated as a full match.
        # Use a capturing group so we can store the exact matched text.
        if isinstance(rep_signifiers, str):
            rep_rx = re.compile(
                rf"(?i)\b({rep_signifiers})\b"
            )
        elif isinstance(rep_signifiers, re.Pattern):
            rep_rx = rep_signifiers
        elif isinstance(rep_signifiers, Iterable):
            pats = [p.pattern if isinstance(p, re.Pattern) else p for p in rep_signifiers]
            rep_rx = re.compile(
                r"(?i)\b(" + "|".join(pats) + r")\b"
            )
        else:
            raise TypeError(f"rep_signifiers must be a string, pattern, or iterable of strings or patterns, not {type(rep_signifiers)}")

        def _rep_repl(m: re.Match) -> str:
            key = f"{_REPEAT_PREFIX}{len(_repeat_map)}_"
            _repeat_map[key] = m.group(0)  # store original, with original casing/spaces
            return key

        s = rep_rx.sub(_rep_repl, s)



    if attempt_to_differentiate_seconds:
        # (a|one|per|each) second
        # Keep the left word intact and replace only "second" with a _sentinel.
        s = re.sub(
            r"\b(a|one|per|each)\s+(second)\b",
            lambda m: f"{m.group(1)} {_SEC__sentinel}",
            s,
            flags=re.IGNORECASE,
        )

        # the second (after|before|between|when)
        s = re.sub(
            r"\b(the)\s+(second)\s+(after|before|between|when)\b",
            lambda m: f"{m.group(1)} {_SEC__sentinel} {m.group(3)}",
            s,
            flags=re.IGNORECASE,
        )

    # --- expand capital suffixes BEFORE lowercasing (capital-only by regex) ---
    if replace_multipliers:
        suffix_multipliers = {"k": 10**3, "K": 10**3, "M": 10**6, "G": 10**9, "T": 10**12, "P": 10**15}

        def _expand_suffix(m):
            n = float(m.group(1))
            suf = m.group(2)
            value = int(n * suffix_multipliers[suf])
            n_fmt = f"{value:,}" if use_commas else str(value)
            m_fmt = suf
            return fmt_multipliers.replace("%n", n_fmt).replace("%m", m_fmt).replace("%i", f"{m.group(1)}{m.group(2)}")

        s = re.sub(
            r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?)([kKMGTP])(?=[^A-Za-z0-9]|$)",
            _expand_suffix,
            s,
        )




    n019 = [
        "zero","one","two","three","four","five","six","seven",
        "eight","nine","ten","eleven","twelve","thirteen","fourteen",
        "fifteen","sixteen","seventeen","eighteen","nineteen"
    ]
    digit_word_to_digit = {
        "o": "0", "oh": "0",
        "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
        "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    }

    tens_words = ["twenty","thirty","forty","fifty","sixty","seventy","eighty","ninety"]

    word_to_num = {w: i for i, w in enumerate(n019)}
    word_to_num.update({w: (i + 2) * 10 for i, w in enumerate(tens_words)})

    magnitude_value = {
        "thousand": 10**3,
        "million": 10**6,
        "billion": 10**9,
        "trillion": 10**12,
        "quadrillion": 10**15,
    }

    # --- Ordinals ---
    ordinal_word_to_num = {}
    ordinal_magnitude_exact = {}
    # --- Fractions (building "num/den", NOT evaluating) ---
    fraction_den_word = {
        "half": 2, "halves": 2,
        "third": 3, "thirds": 3,

        # "hundredth(s)" etc
        "hundredth": 100, "hundredths": 100,
        "thousandth": 10**3, "thousandths": 10**3,
        "millionth": 10**6, "millionths": 10**6,
        "billionth": 10**9, "billionths": 10**9,
        "trillionth": 10**12, "trillionths": 10**12,
        "quadrillionth": 10**15, "quadrillionths": 10**15,
    }

    if support_ordinals:
        ordinal_word_to_num.update({
            "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
            "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
            "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14,
            "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18,
            "nineteenth": 19,
            # common typo
            "fifthe": 5,
        })
        ordinal_word_to_num.update({
            "twentieth": 20, "thirtieth": 30, "fortieth": 40, "fiftieth": 50,
            "sixtieth": 60, "seventieth": 70, "eightieth": 80, "ninetieth": 90,
        })
        ordinal_magnitude_exact = {
            "hundredth": 100,
            "thousandth": 10**3,
            "millionth": 10**6,
            "billionth": 10**9,
            "trillionth": 10**12,
            "quadrillionth": 10**15,
        }

    # --- rep words ---
    repeat_word_to_num = {"once": 1, "twice": 2, "thrice": 3}
    
    # --- Roman Numerals ---
    _ROMAN_PATTERN = re.compile(r"^(?=[MDCLXVI])M*(C[MD]|D?C{0,3})(X[CL]|L?X{0,3})(I[XV]|V?I{0,3})$", re.IGNORECASE)
    
    def roman_to_int(s: str) -> int:
        roman = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
        s = s.upper()
        num = 0
        for i in range(len(s) - 1):
            if roman[s[i]] < roman[s[i + 1]]:
                num -= roman[s[i]]
            else:
                num += roman[s[i]]
        num += roman[s[-1]]
        return num

    def ordinal_suffix(n: int) -> str:
        n_abs = abs(n)
        last_two = n_abs % 100
        if 11 <= last_two <= 13:
            return "th"
        last = n_abs % 10
        if last == 1:
            return "st"
        if last == 2:
            return "nd"
        if last == 3:
            return "rd"
        return "th"


    def is_numeric_atom(tok: str) -> bool:
        t = tok.lower()

        # decimals
        if t == "point":
            return True

        # allow digit-spelling zero tokens to stay inside phrases
        if t in {"oh", "o"}:
            return True

        # keep fraction denominators inside the numeric phrase
        if t in fraction_den_word:
            return True


        if support_reps and t in repeat_word_to_num:
            return True

        if support_ordinals and re.fullmatch(r"\d+(st|nd|rd|th)", t):
            return True

        if t.isdigit() or t in word_to_num or t == "hundred" or t in magnitude_value:
            return True

        if support_ordinals and (t in ordinal_word_to_num or t in ordinal_magnitude_exact):
            return True

        if support_reps and re.fullmatch(rf"{_REPEAT_PREFIX}\d+_", tok, flags=re.IGNORECASE):
            return True

        if support_roman and _ROMAN_PATTERN.fullmatch(tok):
            return True

        return False


    def allows_and_after(prev_norm: str | None) -> bool:
        if prev_norm is None:
            return False
        prev_norm = prev_norm.lower()
        return prev_norm == "hundred" or prev_norm in magnitude_value

    def _is_repeat_tail(w: str) -> bool:
        return re.fullmatch(rf"{_REPEAT_PREFIX}\d+_", w, flags=re.IGNORECASE) is not None



    # returns (value, is_ordinal, is_time_phrase)
    def parse_number(norm_words: list[str]) -> Tuple[object, bool, bool] | None:
        if norm_words and norm_words[-1] == "and":
            return None

        # handle once/twice/thrice as standalone (or with trailing time/times)
        if support_reps and norm_words:
            w0 = norm_words[0]
            if w0 in repeat_word_to_num:
                if len(norm_words) == 1:
                    return (repeat_word_to_num[w0], False, True)
                if _is_repeat_tail(norm_words[-1]):
                    return (repeat_word_to_num[w0], False, True)

        is_time_phrase = False
        if support_reps and norm_words and _is_repeat_tail(norm_words[-1]):
            is_time_phrase = True
            core = norm_words[:-1]
        else:
            core = norm_words

        if not core:
            return None



        # -------------------------
        # FRACTIONS: "<numerator> <denominator>"
        #   "one third" -> 1/3
        #   "two hundredths" -> 2/100
        #   "five point 8 millionth" -> 5.8/1000000
        #   "2 halves" -> 2/2
        #
        # Safety rules:
        #   - denominator must be LAST token in phrase
        #   - fraction only if denom is plural OR numerator is (a|one|1) OR numerator is numeric (e.g. 5.8)
        # -------------------------
        denom_tok = core[-1]

        denom_val: int | None = None
        denom_is_plural = False

        if not is_time_phrase:
            # word denominators
            if denom_tok in fraction_den_word:
                denom_val = fraction_den_word[denom_tok]
                denom_is_plural = denom_tok.endswith("s") or denom_tok.endswith("ves")

            # numeric ordinals like "100th" / "100ths"
            if denom_val is None:
                mo = re.fullmatch(r"(\d+)(st|nd|rd|th)(s)?", denom_tok)
                if mo and mo.group(2) == "th":
                    denom_val = int(mo.group(1))
                    denom_is_plural = mo.group(3) is not None

            if denom_val is not None:
                numer_words = core[:-1]

                # If there's no numerator, DO NOT abort parsing — let ordinal/time logic handle it.
                # Except: allow bare "half" -> 1/2 (per your tests).
                if not numer_words:
                    if denom_tok in {"half", "halves"}:
                        return (("FRAC", "1", denom_val), False, False)
                    # fall through to normal parsing (so "third time" still becomes 3rd time)
                    denom_val = None

                if denom_val is not None:

                    # numerator: allow "a" => 1
                    if len(numer_words) == 1 and numer_words[0] == "a":
                        numer_str = "1"
                        numer_is_oneish = True
                        numer_is_numeric = True
                    else:
                        # reuse your decimal parsing style for numerator-only
                        def _parse_numeric_string(words: list[str]) -> str | None:
                            # decimal numerator: "<int> point <digits...>"
                            if "point" in words:
                                p2 = words.index("point")
                                left2 = words[:p2]
                                right2 = words[p2 + 1:]
                                if not right2:
                                    return None
                                digs: list[str] = []
                                for w in right2:
                                    if w == "and":
                                        return None
                                    if w.isdigit():
                                        digs.append(w)
                                    elif w in digit_word_to_digit:
                                        digs.append(digit_word_to_digit[w])
                                    else:
                                        return None
                                frac = "".join(digs)
                                if frac == "":
                                    return None

                                # parse left2 as integer (allow empty => 0)
                                if not left2:
                                    ip = 0
                                else:
                                    total2 = 0
                                    current2 = 0
                                    saw2 = False
                                    for w in left2:
                                        if w == "and":
                                            continue
                                        if w in {"oh", "o"}:
                                            current2 += 0
                                            saw2 = True
                                            continue
                                        if w.isdigit():
                                            current2 += int(w); saw2 = True; continue
                                        if w in word_to_num:
                                            current2 += word_to_num[w]; saw2 = True; continue
                                        if w == "hundred":
                                            if not saw2: return None
                                            current2 *= 100; continue
                                        if w in magnitude_value:
                                            if not saw2: return None
                                            total2 += current2 * magnitude_value[w]
                                            current2 = 0
                                            continue
                                        return None
                                    if not saw2:
                                        return None
                                    ip = total2 + current2

                                return f"{ip}.{frac}"

                            # integer numerator
                            # (use your existing integer loop behavior, but return string)
                            total2 = 0
                            current2 = 0
                            saw2 = False
                            for w in words:
                                if w == "and":
                                    continue
                                if w in {"oh", "o"}:
                                    current2 += 0
                                    saw2 = True
                                    continue
                                if w.isdigit():
                                    current2 += int(w); saw2 = True; continue
                                if w in word_to_num:
                                    current2 += word_to_num[w]; saw2 = True; continue
                                if w == "hundred":
                                    if not saw2: return None
                                    current2 *= 100; continue
                                if w in magnitude_value:
                                    if not saw2: return None
                                    total2 += current2 * magnitude_value[w]
                                    current2 = 0
                                    continue
                                return None
                            if not saw2:
                                return None
                            return str(total2 + current2)

                        numer_str = _parse_numeric_string(numer_words)
                        if numer_str is None:
                            return None

                        numer_is_oneish = numer_str in {"1", "1.0"}
                        numer_is_numeric = True  # if we parsed it, it’s numeric

                    # gating rule so we don't break ordinals in normal prose
                    if not (denom_is_plural or numer_is_oneish or numer_is_numeric):
                        return None

                    return (("FRAC", numer_str, denom_val), False, False)


        # -------------------------
        # DECIMALS: "<int> point <digits...>"
        #   "zero point five" -> 0.5
        #   "point five" -> 0.5
        #   "two point zero five" -> 2.05
        # -------------------------
        if "point" in core:
            # only support the first "point" for now
            p = core.index("point")
            left = core[:p]
            right = core[p + 1 :]

            # require at least one digit after point
            if not right:
                return None

            # right side must be ONLY digit-words (0-9) or digit tokens
            digits: list[str] = []
            for w in right:
                if w == "and":
                    return None
                if w.isdigit():
                    digits.append(w)  # preserve "05" if it appears
                    continue
                if w in digit_word_to_digit:
                    digits.append(digit_word_to_digit[w])
                    continue
                return None

            frac_digits = "".join(digits)
            if frac_digits == "":
                return None

            # parse left as an integer using the existing logic (but reject ordinals)
            # allow empty left => 0
            if not left:
                int_part = 0
            else:
                total = 0
                current = 0
                saw_any = False
                is_ord_left = False

                for w in left:
                    if w == "and":
                        continue

                    # allow "oh"/"o" as 0 on the LEFT side of "point"
                    if w in {"oh", "o"}:
                        current += 0
                        saw_any = True
                        continue

                    # reject ordinals for decimals in stage 1
                    if support_ordinals:
                        if re.fullmatch(r"\d+(st|nd|rd|th)", w):
                            return None
                        if w in ordinal_word_to_num or w in ordinal_magnitude_exact:
                            return None

                    if w.isdigit():
                        current += int(w)
                        saw_any = True
                        continue

                    if w in word_to_num:
                        current += word_to_num[w]
                        saw_any = True
                        continue

                    if w == "hundred":
                        if not saw_any:
                            return None
                        current *= 100
                        continue

                    if w in magnitude_value:
                        if not saw_any:
                            return None
                        total += current * magnitude_value[w]
                        current = 0
                        continue

                    return None

                if not saw_any:
                    return None

                int_part = total + current

            return (("DEC", int_part, frac_digits), False, False)

        total = 0
        current = 0
        saw_any = False
        is_ord = False

        for w in core:
            if w == "and":
                continue

            if support_ordinals:
                mo = re.fullmatch(r"(\d+)(st|nd|rd|th)", w)
                if mo:
                    current += int(mo.group(1))
                    saw_any = True
                    is_ord = True
                    continue

            if w.isdigit():
                current += int(w)
                saw_any = True
                continue

            if w in word_to_num:
                current += word_to_num[w]
                saw_any = True
                continue

            if support_ordinals and w in ordinal_word_to_num:
                current += ordinal_word_to_num[w]
                saw_any = True
                is_ord = True
                continue

            if w == "hundred":
                if not saw_any:
                    return None
                current *= 100
                continue

            if support_ordinals and w in ordinal_magnitude_exact:
                if not saw_any:
                    # bare "hundredth" => 1 * 100
                    current = ordinal_magnitude_exact[w]
                    saw_any = True
                else:
                    current *= ordinal_magnitude_exact[w]
                is_ord = True
                continue

            if w in magnitude_value:
                if not saw_any:
                    return None
                total += current * magnitude_value[w]
                current = 0
                continue
                
            if support_roman and _ROMAN_PATTERN.fullmatch(w):
                current += roman_to_int(w)
                saw_any = True
                continue

            return None

        return (total + current, is_ord, is_time_phrase) if saw_any else None

    # tokenization that preserves whitespace and punctuation, and keeps numeric ordinals like "1st"
    tokens = re.findall(
        rf"{_REPEAT_PREFIX}\d+_|\d+(?:st|nd|rd|th)|[A-Za-z]+|\d+|\s+|[^A-Za-z\d\s]+",
        s,
        flags=re.IGNORECASE,
    )


    out: list[str] = []
    raw: list[str] = []
    norm: list[str] = []
    pending_ws: str = ""

    def _flush_phrase():
        nonlocal raw, norm, pending_ws
        if not norm:
            return

        # Decide whether this phrase is worth attempting to parse
        has_convertible = False
        for w in norm:
            if w == "and":
                continue

            if w in fraction_den_word:
                has_convertible = True
                break

            if w.isdigit():
                continue
            if w in word_to_num or w == "hundred" or w in magnitude_value:
                has_convertible = True
                break
            if support_ordinals and (w in ordinal_word_to_num or w in ordinal_magnitude_exact or re.fullmatch(r"\d+(st|nd|rd|th)", w)):
                has_convertible = True
                break
            if support_reps and re.fullmatch(rf"{_REPEAT_PREFIX}\d+_", w, flags=re.IGNORECASE):
                has_convertible = True
                break
            if support_reps and w in repeat_word_to_num:
                has_convertible = True
                break
            if support_roman and _ROMAN_PATTERN.fullmatch(w):
                has_convertible = True
                break


        if not has_convertible:
            out.append("".join(raw))
            raw = []
            norm = []
            if pending_ws:
                out.append(pending_ws)
                pending_ws = ""
            return

        parsed = parse_number(norm)
        if parsed is None:
            out.append("".join(raw))
        else:
            n, is_ord, is_time = parsed


            # fractions
            if isinstance(n, tuple) and len(n) == 3 and n[0] == "FRAC":
                numer_str, denom_val = n[1], n[2]
                num = f"{numer_str}/{denom_val}"
                out.append(
                    fmt
                    .replace("%n", num)
                    .replace("%s", "s")
                    .replace("%r", "x")
                    .replace("%i", "".join(raw))
                )


            # decimals
            elif isinstance(n, tuple) and len(n) == 3 and n[0] == "DEC":
                int_part, frac_digits = n[1], n[2]
                int_part_str = f"{int_part:,}" if use_commas else str(int_part)
                num = f"{int_part_str}.{frac_digits}"
                # decimals are never ordinals/reps in stage 1
                out.append(
                    fmt
                    .replace("%n", num)
                    .replace("%s", "s")  # irrelevant but keep pipeline stable
                    .replace("%r", "x")
                    .replace("%i", "".join(raw))
                )
            else:
                # existing integer behavior
                num = f"{n:,}" if use_commas else str(n)
                plural_s = "s" if abs(n) != 1 else ""
                if _repeat_map:
                    x = "".join(raw)
                    pat = re.compile(rf"^(.*)?({_REPEAT_PREFIX}\d+_)(.*)$")
                    m = re.match(pat, x)
                    r = _repeat_map[m.group(2)] if m else "x"
                    i = re.sub(pat, rf"\1{r}", x)
                else:
                    r = "x"
                    i = "".join(raw)


                if is_time and support_reps:
                    if is_ord and support_ordinals:
                        suf = ordinal_suffix(n)
                        out.append(
                            fmt_nth_time
                            .replace("%n", num)
                            .replace("%o", suf)
                            .replace("%s", plural_s)
                            .replace("%r",  r)
                            .replace("%i", i)
                        )
                    else:
                        out.append(
                            fmt_rep
                            .replace("%n", num)
                            .replace("%s", plural_s)
                            .replace("%r",  r)
                            .replace("%i", i)
                        )
                else:
                    if is_ord and support_ordinals:
                        suf = ordinal_suffix(n)
                        out.append(
                            fmt_ordinal
                            .replace("%n", num)
                            .replace("%o", suf)
                            .replace("%s", plural_s)
                            .replace("%r",  r)
                            .replace("%i", i)
                        )
                    else:
                        out.append(
                            fmt
                            .replace("%n", num)
                            .replace("%s", plural_s)
                            .replace("%r",  r)
                            .replace("%i", i)
                        )



        raw = []
        norm = []
        if pending_ws:
            out.append(pending_ws)
            pending_ws = ""

    def _next_nonspace(j: int) -> str | None:
        k = j + 1
        while k < len(tokens) and tokens[k].isspace():
            k += 1
        return tokens[k] if k < len(tokens) else None

    def _peek_decimal_start(i: int) -> bool:
        """
        True if tokens after index i look like: <ws>* 'point' <ws>* <digit-or-digitword>
        """
        nxt = _next_nonspace(i)
        if nxt is None or nxt.lower() != "point":
            return False
        # find token after 'point'
        j = i + 1
        while j < len(tokens) and tokens[j].isspace():
            j += 1
        if j >= len(tokens) or tokens[j].lower() != "point":
            return False
        k = j + 1
        while k < len(tokens) and tokens[k].isspace():
            k += 1
        if k >= len(tokens):
            return False
        t = tokens[k].lower()
        return t.isdigit() or t in digit_word_to_digit


    def _commit_pending_ws_into_phrase():
        nonlocal pending_ws
        if pending_ws:
            raw.append(pending_ws)
            pending_ws = ""

    def _hyphen_is_internal(prev_norm: str | None, next_tok: str | None) -> bool:
        if prev_norm is None or next_tok is None:
            return False
        p = prev_norm.lower()
        nxt = next_tok.lower()

        if p in tens_words:
            if nxt.isdigit() or nxt in word_to_num:
                return True
            if support_ordinals and nxt in ordinal_word_to_num:
                return True
        return False

    for i, t in enumerate(tokens):
        if t.isspace():
            if norm:
                pending_ws += t
            else:
                out.append(t)
            continue

        if re.fullmatch(rf"{_REPEAT_PREFIX}\d+_|[A-Za-z]+|\d+|\d+(?:st|nd|rd|th)", t, flags=re.IGNORECASE):
            tl = t.lower()

            # allow "a" as numerator ONLY when it starts a fraction phrase: "a third", "a half", etc.
            if tl == "a":
                nxt = _next_nonspace(i)
                if nxt is not None and nxt.lower() in fraction_den_word:
                    if norm:
                        _commit_pending_ws_into_phrase()
                    raw.append(t)        # keep "a"
                    norm.append("a")     # normalize
                    continue

            # special handling for "time/times":
            # only include it as part of phrase if the phrase so far parses as a number (cardinal/ordinal/repeat word)
            # special handling for "time/times" sentinels
            if support_reps and re.fullmatch(rf"{_REPEAT_PREFIX}\d+_", t, flags=re.IGNORECASE):
                nxt = _next_nonspace(i)

                # If another numeric atom follows, this is NOT repetition; it's "A times B"
                # So: flush the left number, emit the original word ("times"), and continue.
                if nxt is not None and is_numeric_atom(nxt):
                    _flush_phrase()
                    out.append(_repeat_map.get(t, t))  # emit original signifier (e.g., "times")
                    continue

                # otherwise keep existing repetition behavior
                if norm:
                    if parse_number(norm) is not None:
                        _commit_pending_ws_into_phrase()
                        raw.append(t)
                        norm.append(t.lower())
                    else:
                        _flush_phrase()
                        out.append(_repeat_map.get(t, t))
                else:
                    out.append(_repeat_map.get(t, t))
                continue


            if tl == "and":
                if norm:
                    nxt = _next_nonspace(i)
                    prev_norm = next((w for w in reversed(norm) if w != "and"), None)
                    if allows_and_after(prev_norm) and (nxt is not None and is_numeric_atom(nxt)):
                        _commit_pending_ws_into_phrase()
                        raw.append(t)
                        norm.append("and")
                    else:
                        _flush_phrase()
                        out.append(t)
                else:
                    out.append(t)
                continue

            # special-case: "oh point ..." / "o point ..." should behave like "zero point ..."
            if tl in {"oh", "o"} and _peek_decimal_start(i):
                if norm:
                    _commit_pending_ws_into_phrase()
                raw.append(t)          # keep original text (oh/o)
                norm.append("zero")    # treat as numeric 0 for parsing
                continue


            if is_numeric_atom(t):
                if norm:
                    _commit_pending_ws_into_phrase()
                raw.append(t)
                norm.append(tl)
            else:
                _flush_phrase()
                out.append(t)
            continue

        if t == "-":
            if norm:
                nxt = _next_nonspace(i)
                prev_norm = next((w for w in reversed(norm) if w != "and"), None)
                if _hyphen_is_internal(prev_norm, nxt):
                    _commit_pending_ws_into_phrase()
                    raw.append("-")
                    continue
            _flush_phrase()
            out.append("-")
            continue

        _flush_phrase()
        out.append(t)

    _flush_phrase()
    if pending_ws:
        out.append(pending_ws)

    s = "".join(out).replace("%p", "")

    if attempt_to_differentiate_seconds:
        s = re.sub(_SEC__sentinel, "second", s)

    all_units = []
    if isinstance(units, UnitGroup | Unit):
        all_units = units.children
    else:
        for u in units:
            all_units.extend(u.children)

    all_units = list(sorted(all_units, key=lambda u: len(u.key), reverse=True))


    # print("pre sub", s)
    for u in all_units:
        # print(f"checking unit {u.key}, {u.full_pattern}, {u.full_replacement}, {s=}")
        s = u.sub(s, all_units)
        # print(f"{s=}")
    # print("post sub", s)

    if _repeat_map:
        s = re.sub(
            rf"{_REPEAT_PREFIX}\d+_",
            lambda m: _repeat_map.get(m.group(0), m.group(0)),
            s,
        )

    s = s.replace("%p", "")

    if parse_signs:
        num_rx = r"(\d+(?:\.\d+)?(?:/\d+)?(?:e[+-]?\d+)?)"
        s = re.sub(rf"\b(neg|negative|minus)\s+{num_rx}\b", r"-\2", s, flags=re.IGNORECASE)
        s = re.sub(rf"\b(pos|positive|plus)\s+{num_rx}\b", r"+\2", s, flags=re.IGNORECASE)
        s = re.sub(rf"\+\s+{num_rx}\b", r"+\1", s, flags=re.IGNORECASE)
        s = re.sub(rf"-\s+{num_rx}\b", r"-\1", s, flags=re.IGNORECASE)

    # x over y  /  x divided by y  ->  x/y
    s = re.sub(
        r"\b([+-]?\d+(?:\.\d+)?)\s+(?:over|divided\s+by)\s+(\d+(?:\.\d+)?)\b",
        rf"\1{div}\2",
        s,
        flags=re.IGNORECASE,
    )
    # x over y  /  x divided by y  ->  x/y
    s = re.sub(
        r"\b([+-]?\d+(?:\.\d+)?)\s+(?:over|divided\s+by|out of|into|of)\s+(\d+(?:\.\d+)?)\b",
        rf"\1{div}\2",
        s,
        flags=re.IGNORECASE,
    )

    # multiplication: "<num> times <num>" or "<num> multiplied by <num>" -> "<num>*<num>"
    num_atom = r"(?:[+-]?\d+(?:\.\d+)?(?:/\d+)?(?:e[+-]?\d+)?)"



    s = re.sub(
        rf"({num_atom})\s\+({num_atom})",
        r"\1+\2",
        s,
        flags=re.IGNORECASE,
    )
    s = re.sub(
        rf"({num_atom})\s-({num_atom})",
        r"\1-\2",
        s,
        flags=re.IGNORECASE,
    )


    def _frac_exponent_to_ordinal(m: re.Match) -> str:
        base = m.group(1)
        numer = m.group(2)
        denom = int(m.group(3))

        # only if numerator is a plain integer
        if not numer.isdigit():
            return m.group(0)

        # only for "hundredth/thousandth/millionth..." style denominators
        # (i.e. powers of 10 >= 100)
        if denom < 100:
            return m.group(0)
        t = denom
        while t % 10 == 0:
            t //= 10
        if t != 1:
            return m.group(0)

        exp = int(numer) * denom
        return f"{base} to the {exp} power"
    s = re.sub(
        r"\b([+-]?\d+(?:\.\d+)?(?:/\d+)?)\s+to\s+the\s+(\d+)/(\d+)\s+power\b",
        _frac_exponent_to_ordinal,
        s,
        flags=re.IGNORECASE,
    )
    s = re.sub(
        rf"\b({num_atom})\s(?:raised )?(?:to the power of|to the)\s({num_atom})(?:rd|st|th)?(?: (?:power|exponent|degree))?\b",
        rf"\1{power}\2",
        s,
        flags=re.IGNORECASE,
    )
    s = re.sub(
        rf"\b({num_atom})\s(?:raised )?(?:to the power of|to the)\s({num_atom})(?:rd|st|th)\b",
        rf"\1{power}\2",
        s,
        flags=re.IGNORECASE,
    )
    powers = {"squared": 2, "cubed": 3}
    for k, v in powers.items():
        s = re.sub(
            rf"\b({num_atom})\s(?:raised )?(?:to the power of|to the)?\s({k})(?: power)?\b",
            rf"\1{power}{v}",
            s,
            flags=re.IGNORECASE,
        )

    # helpers
    ws = r"\s+"
    num_atom = r"[+-]?\d+(?:\.\d+)?(?:/\d+)?"

    # square root of x  ->  sqrt(x)
    s = re.sub(
        rf"\b(square)\s+root(?:\s+of)?{ws}({num_atom})\b",
        rf"\2{power}(1/2)",
        s,
        flags=re.IGNORECASE,
    )

    # cube root of x  ->  cbrt(x)
    s = re.sub(
        rf"\b(cube)\s+root(?:\s+of)?{ws}({num_atom})\b",
        rf"\2{power}(1/3)",
        s,
        flags=re.IGNORECASE,
    )

    # nth root of x  ->  root(n,x)
    # "the 5th root of 32", "5th root of 32", "5 root of 32" (if you want)
    s = re.sub(
        rf"\b(?:the\s+)?({num_atom})(?:st|nd|rd|th)?\s+root(?:\s+of)?{ws}({num_atom})\b",
        rf"\2{power}(1/\1)",
        s,
        flags=re.IGNORECASE,
    )


    # --- Scientific notation normalization (place right before return) ---
    num_atom_sci = r"[+-]?\d+(?:\.\d+)?(?:/\d+)?(?:e[+-]?\d+)?"

    # 1) "6 e -5" -> "6e-5"
    s = re.sub(
        rf"\b({num_atom_sci})\s*e\s*([+-]?\d+)\b",
        r"\1e\2",
        s,
        flags=re.IGNORECASE,
    )

    # 2) "6e-5" -> "6{mult}10{power}-5"
    s = re.sub(
        rf"\b([+-]?\d+(?:\.\d+)?)(?:e([+-]?\d+))\b",
        lambda m: f"{m.group(1)}{mult}10{power}{m.group(2)}",
        s,
        flags=re.IGNORECASE,
    )

    # 3) "6*10^5", "6×10**(5)", "6 x 10^5" -> "6{mult}10{power}5"
    ten_pow = rf"10(?:\s*)?(?:\^|\*\*)(?:\s*)?(\(?[+-]?\d+\)?)"
    s = re.sub(
        rf"\b({num_atom_sci})\s*(?:\*|x|×)\s*{ten_pow}\b",
        lambda m: f"{m.group(1)}{mult}(10{power}{m.group(2)})",
        s,
        flags=re.IGNORECASE,
    )

    # --- ACTUAL_MATH: mixed numbers like "1 and 1/2" ---
    # supports: "1 and 1/2", "1 and 2/3", etc.
    if combine_add is not False:  # default True
        _res = 3 if res is _sentinel else int(res) if res is not None else res
    else:
        _res = None


    def _mixed_repl(m: re.Match | str, _res) -> str:
        whole_s = m.group(1)
        num_s = m.group(2)
        den_s = m.group(3)

        try:
            whole = Fraction(whole_s)
            num = Fraction(num_s)
            den = Fraction(den_s)
            if den == 0:
                return m.group(0)

            val = whole + (num / den)

            # EXACT-ONLY MODE
            if _res is None:
                # only emit if decimal terminates
                d = val.denominator
                while d % 2 == 0:
                    d //= 2
                while d % 5 == 0:
                    d //= 5
                if d != 1:
                    return m.group(0)

                return _fraction_to_exact_str(val)

            # ROUNDED MODE
            return _fraction_to_exact_str(
                val.limit_denominator()  # already exact; formatting handles rounding elsewhere
            )

        except Exception:
            return m.group(0)



    # Only collapse when the RHS is a fraction token your pipeline produced.
    # This avoids touching phrases like "rock and roll".
    s = re.sub(
        r"\b([+-]?\d+(?:\.\d+)?)\s+and\s+([+-]?\d+(?:\.\d+)?)/(\d+)\b",
        lambda f: _mixed_repl(f, res),
        s,
        flags=re.IGNORECASE,
    )

    # --- unit mixed-number: (a|N) <unit> and a/b  ->  (N + a/b) <unit> ---
    # Assumes "half" etc is already replaced into "1/2" earlier.

    def _unit_and_frac(m: re.Match) -> str:
        whole_txt = m.group(1).lower()   # "a" or digits
        unit = m.group(2)               # e.g. "day", "days"
        a = int(m.group(3))
        b = int(m.group(4))

        whole = 1 if whole_txt == "a" else int(whole_txt)
        val = whole + (a / b)

        # if you have combine_add/res, use them; otherwise just default to 3 decimals
        if "combine_add" in locals() and combine_add is False:
            whole_out = "1" if whole_txt == "a" else m.group(1)
            return f"{whole_out} {unit} and {a}/{b}"

        places = res if ("res" in locals() and isinstance(res, int)) else 3
        out = f"{val:.{places}f}".rstrip("0").rstrip(".")

        plural_unit = unit
        if unit.endswith("ay"):
            plural_unit = unit + "s"
        elif unit.endswith("ry"):
            plural_unit = unit[:-2] + "ries"
        elif not unit.endswith("s"):
            plural_unit = unit + "s"
        return f"{out} {plural_unit}"

    s = re.sub(
        r"\b(a|\d+)\s+(\S+)\s+and\s+(\d+)/(\d+)$",
        _unit_and_frac,
        s,
        flags=re.IGNORECASE,
    )
    # print("testing", s)

    s = re.sub(
        r"(\d+(?:\/|\.)\d+) of (?:an?\s?set(?:\(s\))?s? of )?(\d+)",
        rf"(\1){mult}\2",
        s,
        flags=re.IGNORECASE,
    )
    s = re.sub(
        r"(\d+) and (\d+(?:\.|\/)\d+)",
        rf"\1 + \2",
        s,
        flags=re.IGNORECASE,
    )


    if do_simple_evals:
        # print("se", s)
        s = simple_eval(s, power=power, mult=mult, div=div, eval_fractions=do_fraction_evals,res=res)
        # print("postse", s)

    s = re.sub(
        rf"({num_atom})\s?(?:time|multiplied|timesed|occurence|instance|attempt|multiply|multiple|set)(?:\(s\))?s?(?: (?:by|of))?\s+({num_atom})",
        rf"\1{mult}\2",
        s,
        flags=re.IGNORECASE,
    )

    if do_simple_evals:
        # print("se", s)
        s = simple_eval(s, power=power, mult=mult, div=div, eval_fractions=do_fraction_evals,res=res)

    # replace hanging
    s = re.sub(rf"\b(?:an? )?set(?:\(s\))?s? of (\d)", r"\1", s, flags=re.IGNORECASE)
    def merge_units(s, u):
        if isinstance(u, UnitGroup):
            s = u.base.base_merge(s)
        elif isinstance(u, Unit):
            s = u.base_merge(s)
        else:
            for u2 in u:
                s = merge_units(s, u2)
        return s
    def merge_units2(s, u):
        if isinstance(u, UnitGroup):
            s = u.base.base_merge2(s)
        elif isinstance(u, Unit):
            s = u.base_merge2(s)
        else:
            for u2 in u:
                s = merge_units2(s, u2)
        return s
    base_units = [u for u in all_units if u.base]
    known_bu_keys = set()
    for bu in base_units:
        if isinstance(bu.key, str):
            known_bu_keys.add(bu.key)
        else:
            for buk in bu.key:
                known_bu_keys.add(buk)


    new_units = set()
    for u in all_units:
        nu = u.new_unit if isinstance(u.new_unit, str) else u.new_unit.key if hasattr(u.new_unit, "key") else ""
        if nu:
            if isinstance(nu, str):
                new_units.add(nu)
            else:
                for nuk in nu:
                    new_units.add(nuk)
    # u.new_unit if isinstance(u.new_unit, str) else u.new_unit.key if hasattr(u.new_unit, "key") else "" for u in all_units)
    new_units = [nu for nu in new_units if nu not in known_bu_keys]
    more_bases = [Unit(k, base=True) for k in new_units]
    # print("more bases", new_units)

    s = merge_units(s, base_units)
    s = merge_units(s, more_bases)
    # print("pre-merg2", s)
    s = merge_units2(s, base_units)
    s = merge_units2(s, more_bases)
    # print("post-merg2", s)
    if do_simple_evals:
        # print("se", s)
        s = simple_eval(s, power=power, mult=mult, div=div, eval_fractions=do_fraction_evals,res=res)

    s = merge_units(s, base_units)
    s = merge_units(s, more_bases)

    if not _iter:
        return s
    first = s
    s2 = digitize(s, config=config, _iter=False)
    # print("ITER 0", s)
    # print("ITER 1", s2)
    i = 0
    prevs = [s]
    while i < 100:
        if s2 == s:
            return s
        if s2 in prevs:
            return first
        if len(s2) > len(s):
            return s
        prevs.append(s2)
        s = s2
        s2 = digitize(s, config=config, _iter=False)
        i += 1
    raise StopIteration(f"hit {i}")






def _pi_for_res(res: int | None) -> str:
    # Use guard digits so later rounding is stable.
    # If res is None, default to ~50 digits.
    guard = 25
    if res is None:
        return PI_DECIMAL
    # +2 because "3." counts
    digits = min(len(PI_DECIMAL), res + guard + 2)
    return PI_DECIMAL[:digits]


def _has_real_math(expr: str) -> bool:
    """
    True iff the expression contains at least TWO numbers and at least one binary operator.
    This rejects: +10, -10, (10), -(10), + (10)
    Accepts: 2+3, 2*-3, (2+3), -(2+3), 10/4, 2**3
    """
    try:
        node = ast.parse(expr, mode="eval")
    except SyntaxError:
        return False

    nums = 0
    has_binop = False

    for n in ast.walk(node):
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
            has_binop = True
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            nums += 1
        # optional: Python <3.8
        if hasattr(ast, "Constant") and isinstance(n, ast.Constant):  # type: ignore[attr-defined]
            nums += 1

    return has_binop and nums >= 2


_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
_ALLOWED_UNARYOPS = (ast.UAdd, ast.USub)

_DEC_RX = re.compile(r"(?<![\w.])(\d+\.\d+)(?![\w.])")  # decimal literals only, no sign

def _decimal_literal_to_fraction_str(lit: str) -> str:
    # "5.8" -> "Fraction(58, 10)"
    a, b = lit.split(".", 1)
    num = int(a + b)
    den = 10 ** len(b)
    return f"Fraction({num},{den})"

def _rewrite_decimal_literals(expr: str) -> str:
    return _DEC_RX.sub(lambda m: _decimal_literal_to_fraction_str(m.group(1)), expr)

def _fraction_to_exact_str(x: Fraction) -> str:
    """
    If denominator has only 2s and 5s -> exact terminating decimal string.
    Else -> "n/d" (reduced).
    """
    n, d = x.numerator, x.denominator
    dd = d
    while dd % 2 == 0: dd //= 2
    while dd % 5 == 0: dd //= 5
    if dd != 1:
        return f"{n}/{d}"

    # terminating decimal
    sign = "-" if n < 0 else ""
    n = abs(n)

    # scale to integer / 10^k
    k = 0
    dd = d
    while dd % 2 == 0: dd //= 2; k += 1
    while dd % 5 == 0: dd //= 5; k += 1

    # compute exact decimal digits
    scaled = n * (10**k) // d
    s = str(scaled)
    if k == 0:
        return sign + s
    if len(s) <= k:
        s = "0" * (k - len(s) + 1) + s
    out = s[:-k] + "." + s[-k:]
    out = out.rstrip("0").rstrip(".")
    return sign + out

from fractions import Fraction
from math import isqrt

def _int_nth_root_exact(n: int, k: int) -> int | None:
    """Return exact integer r such that r**k == n, else None. n>=0, k>=1."""
    if n < 0:
        return None
    if k == 1:
        return n
    if n in (0, 1):
        return n
    if k == 2:
        r = isqrt(n)
        return r if r * r == n else None

    # binary search
    lo, hi = 0, 1
    while hi**k < n:
        hi *= 2
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        p = mid**k
        if p == n:
            return mid
        if p < n:
            lo = mid
        else:
            hi = mid
    return None

def _pow_fraction_exact(base: Fraction, exp: Fraction) -> Fraction | None:
    """
    Return exact Fraction for base**exp iff it stays rational.
    Otherwise return None (meaning: leave expression untouched).
    """
    if exp == 0:
        return Fraction(1, 1)

    # integer exponent => always rational
    if exp.denominator == 1:
        return base ** exp.numerator

    # rational exponent p/q in lowest terms
    p, q = exp.numerator, exp.denominator
    if base == 0:
        return Fraction(0, 1) if p > 0 else None  # 0**neg or 0**fraction undefined-ish

    # handle sign: negative base with non-integer exponent is not rational in general
    if base < 0:
        return None

    # reduce base
    a = base.numerator
    b = base.denominator

    ra = _int_nth_root_exact(a, q)
    rb = _int_nth_root_exact(b, q)
    if ra is None or rb is None:
        return None  # would be irrational -> do not evaluate

    rooted = Fraction(ra, rb)

    # now raise rooted to integer p
    if p >= 0:
        return rooted ** p
    else:
        return Fraction(1, 1) / (rooted ** (-p))



def _dec_pow_rational_exp(base: Fraction, exp: Fraction, res: int) -> Fraction:
    """
    Approximate base**exp (base>0, exp rational) to `res` decimal places,
    returned as a Fraction that exactly equals that decimal.
    """
    if base <= 0:
        raise ValueError("decimal approx only for positive base")

    p, q = exp.numerator, exp.denominator
    if q <= 0:
        raise ValueError("bad exponent")

    # work precision: res + guard digits
    prec = max(50, res + 25)

    with localcontext() as ctx:
        ctx.prec = prec

        B = Decimal(base.numerator) / Decimal(base.denominator)

        # compute q-th root with Newton: y_{n+1} = ((q-1)*y + B / y^(q-1)) / q
        def nth_root(x: Decimal, n: int) -> Decimal:
            if n == 1:
                return x
            if n == 2:
                return x.sqrt()
            # initial guess
            y = x if x < 1 else x / Decimal(n)
            for _ in range(60):
                y_prev = y
                y = ((Decimal(n - 1) * y) + (x / (y ** (n - 1)))) / Decimal(n)
                if y == y_prev:
                    break
            return y

        # handle negative exponent via reciprocal at the end
        neg = p < 0
        p = abs(p)

        root = nth_root(B, q)
        val = (root ** p)
        if neg:
            val = Decimal(1) / val

        # round to `res` decimals (half-up), then convert to exact Fraction
        if res <= 0:
            quant = Decimal("1")
        else:
            quant = Decimal("1").scaleb(-res)  # 10^-res

        rounded = val.quantize(quant, rounding=ROUND_HALF_UP)

        # Fraction(str(Decimal)) gives an exact rational equal to that decimal text
        return Fraction(str(rounded))


def _safe_eval_expr(expr: str, eval_fractions: bool = False, res: int = 3) -> Fraction:
    """
    Safely evaluate expression with only +,-,*,/,** and parentheses,
    returning an EXACT Fraction. Non-integer exponents are rejected
    (so we don't accidentally introduce irrationals / floats).
    """

    expr2 = _rewrite_decimal_literals(expr)
    node = ast.parse(expr2, mode="eval")

    def eval_node(n) -> Fraction:
        if isinstance(n, ast.Expression):
            return eval_node(n.body)

        if isinstance(n, ast.Constant):
            if isinstance(n.value, int):
                return Fraction(n.value, 1)
            if isinstance(n.value, float):
                # should not happen after rewrite; keep safest fallback
                return Fraction(str(n.value))
            raise ValueError

        if isinstance(n, ast.UnaryOp) and isinstance(n.op, _ALLOWED_UNARYOPS):
            v = eval_node(n.operand)
            return v if isinstance(n.op, ast.UAdd) else -v

        if isinstance(n, ast.BinOp) and isinstance(n.op, _ALLOWED_BINOPS):
            l = eval_node(n.left)
            r = eval_node(n.right)

            if isinstance(n.op, ast.Add):  return l + r
            if isinstance(n.op, ast.Sub):  return l - r
            if isinstance(n.op, ast.Mult): return l * r
            if isinstance(n.op, ast.Div):  return l / r

            if isinstance(n.op, ast.Pow):
                if r.denominator == 1:
                    return l ** r.numerator

                # fractional exponent:
                got = _pow_fraction_exact(l, r)
                if got is not None:
                    return got

                if eval_fractions:
                    # APPROXIMATE irrational power to `res` decimals
                    return _dec_pow_rational_exp(l, r, res)

                raise ValueError("non-integer exponent")

        # allow only Fraction(...) calls we injected
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "Fraction":
            if len(n.args) != 2:
                raise ValueError
            a = eval_node(n.args[0])
            b = eval_node(n.args[1])
            if a.denominator != 1 or b.denominator != 1:
                raise ValueError
            return Fraction(a.numerator, b.numerator)

        raise ValueError(f"Disallowed expression: {expr!r}")

    return eval_node(node)

def _is_terminating_decimal(d: int) -> bool:
    d = abs(d)
    if d == 0:
        return False
    while d % 2 == 0:
        d //= 2
    while d % 5 == 0:
        d //= 5
    return d == 1

def _pi_decimal_str(res: int | None) -> str:
    # res=None => full PI_DECIMAL constant
    if res is None:
        return PI_DECIMAL
    if res <= 0:
        return "3"
    with localcontext() as ctx:
        ctx.prec = max(60, res + 25)
        q = Decimal("1").scaleb(-res)  # 10^-res
        d = Decimal(PI_DECIMAL).quantize(q, rounding=ROUND_HALF_UP)
        s = format(d, "f").rstrip("0").rstrip(".")
        return s


def simple_eval(
    s: str,
    power="**",
    mult="*",
    div="/",
    eval_fractions: bool = False,
    res: int = 3,
    *,
    max_decimal_digits: int = 2000,   # safety: don't expand gigantic terminating decimals
) -> str:
    s = s.replace(power, "**").replace(mult, "*").replace(div, "/")

    expr_rx = re.compile(r"""
        [0-9\(\)\.pPiI]                # first non-space char
        [0-9\(\)\.\s\+\-\*/pPiI]*      # middle (spaces allowed)
        [0-9\)\.pPiI]                  # last non-space char
    """, re.VERBOSE)
    for m in reversed(list(expr_rx.finditer(s))):
        expr = m.group(0)
        if not _has_real_math(expr):
            continue

        core = expr.strip()
        try:
            val: Fraction = _safe_eval_expr(core, eval_fractions=eval_fractions, res=res)
        except Exception:
            continue

        out = _fraction_to_exact_str(val)

        if eval_fractions and "/" in out:
            n, d = val.numerator, val.denominator
            if d != 0:
                abs_d = abs(d)
                abs_n = abs(n)

                if res is None:
                    if _is_terminating_decimal(d):
                        out = _fraction_to_exact_str(val)

                elif res <= max_decimal_digits:
                    neg = (n < 0) ^ (d < 0)
                    n = abs_n
                    d = abs_d

                    q, r = divmod(n, d)
                    if res == 0:
                        if 2 * r >= d:
                            q += 1
                        out = f"-{q}" if neg and q != 0 else str(q)
                    else:
                        scale = 10 ** res
                        scaled, rem = divmod(r * scale, d)
                        if 2 * rem >= d:
                            scaled += 1
                            if scaled == scale:
                                q += 1
                                scaled = 0

                        frac = str(scaled).rjust(res, "0")
                        out = f"{q}.{frac}".rstrip("0").rstrip(".")
                        if neg and out != "0":
                            out = "-" + out

        # Put whitespace back exactly as it was in the matched span
        s = s[:m.start()] + out + s[m.end():]


    s = s.replace("**", power).replace("*", mult).replace("/", div)
    return s



EXAMPLES: list[tuple[str, dict, str]] = [
    ("i have one hundred and one dalmatians", {}, 'i have 101 dalmatians'),
    ("one million and two hundred and thirty four", {"use_commas": True}, '1,000,234'),
    ("Traffic hit 10K requests per second.", {}, 'Traffic hit 10000 requests per second.'),
    ("one hundred and first place", {}, '101st place'),
    ("we shipped on the 11th hour", {}, 'we shipped on the 11th hour'),
    ("she pinged me twice", {}, 'she pinged me 2 times'),
    ("he tried two times and failed", {"fmt_rep": "%nx", "rep_signifiers": r"times?"}, 'he tried 2x and failed'),
    ("third time was the charm", {"fmt_nth_time": "%n-thx", "rep_signifiers": r"times?"}, '3-thx was the charm'),
    ("the second attempt at which it worked", {"fmt_rep": "%nx", "rep_signifiers": COMPLEX_REP_PATTERN}, 'the 2nd time it worked'),
    ("first place, two times, and 3M users", {"config": "token"}, '[NUM=1,ORD=st,OG=first] place, [NUM=2,REP=times,OG=two times], and [NUM=3000000,MULT=M,OG=3M] users'),
]
STORIES = [
    ("one hundred and thirty-five", {}, "135"),
    ("one thousand and one", {}, "1001"),
    ("one thousand and one nights", {}, "1001 nights"),
    ("i have one hundred and one dalmatians", {}, "i have 101 dalmatians"),
    ("she counted one, and then two, and then three", {}, "she counted 1, and then 2, and then 3"),
    ("one hundred and twenty three thousand and four", {}, "123004"),
    ("one million and two hundred and thirty four", {}, "1000234"),
    ("one hundred and", {}, "100 and"),
    ("and one hundred", {}, "and 100"),
    ("he whispered one hundred and thirty five times", {"fmt_rep": "%nx"}, "he whispered 135x"),

    ("one hundred 33 dalmatians", {}, "133 dalmatians"),
    ("one thousand 2 nights", {}, "1002 nights"),
    ("one million two hundred and thirty four", {}, "1000234"),
    ("one hundred and 33", {}, "133"),
    ("one thousand and 2", {}, "1002"),

    ("one hundred and thirty five, exactly", {}, "135, exactly"),
    ("he said one hundred and one.", {}, "he said 101."),
    ("i have (one hundred and one) dalmatians", {}, "i have (101) dalmatians"),
    ("one hundred and one\nnights", {}, "101\nnights"),
    ("one hundred and one  nights", {}, "101  nights"),

    ("one hundred and thirty-five-six", {}, "129"),
    ("one hundred and thirty-five and six and seven", {}, "135 and 6 and 7"),
    ("one and two hundred", {}, "1 and 200"),
    ("one hundred and one and two", {}, "101 and 2"),
]


FORMATTING_TALES = [
    ("one hundred and thirty five", {"use_commas": True}, "135"),
    ("one hundred and thirty five thousand", {"use_commas": True}, "135,000"),
    ("one hundred and thirty five pigs", {"fmt": "[%n]"}, "[135] pigs"),
    ("he saw one hundred and thirty five birds", {"fmt": ""}, "he saw  birds"),

    ("one million and two hundred and thirty four", {"use_commas": True}, "1,000,234"),
    ("balance: 0,000 and one", {}, "balance: 0,000 and 1"),
    ("he saw one hundred and one birds", {"fmt": "<%n>"}, "he saw <101> birds"),
    ("he saw one hundred and one birds", {"fmt": ""}, "he saw  birds"),
]


EDGE_CASE_CHRONICLES = [
    ("someone said one hundred and nothing else", {}, "someone said 100 and nothing else"),
    ("stone and one hundred and one", {}, "stone and 101"),
    ("one and one is two", {}, "1 and 1 is 2"),
    ("one hundred and thirty-five and six", {}, "135 and 6"),
    ("one hundred and thirty five-six", {}, "129"),
    (" ", {}, " "),
    ("", {}, ""),
    ("one... and then one hundred and one!!!", {}, "1... and then 101!!!")
]


# ---------------------------------------------------------------------------
# Suffix sagas: capitalized K/M/G/T/P abbreviations should expand;
# lowercase should not. (Now preserves original case outside numeric changes.)
# ---------------------------------------------------------------------------

SUFFIX_SAGAS = [
    ("We raised 3M dollars.", {}, "We raised 3000000 dollars."),
    ("Traffic hit 10K requests per second.", {}, "Traffic hit 10000 requests per second."),
    ("Storage is 2G and climbing.", {}, "Storage is 2000000000 and climbing."),
    ("Budget: 2.5M for phase one.", {}, "Budget: 2500000 for phase 1."),
    ("Big money: 1T reasons to care.", {}, "Big money: 1000000000000 reasons to care."),
    ("Rare air: 1P possibilities.", {}, "Rare air: 1000000000000000 possibilities."),

    ("We raised 3m dollars.", {}, "We raised 3m dollars."),
    ("Traffic hit 10k requests per second.", {}, "Traffic hit 10000 requests per second."),
    ("He wrote 2g on the napkin.", {}, "He wrote 2g on the napkin."),

    ("I have 10K and one dreams.", {}, "I have 10000 and 1 dreams."),
    ("Deploy to 3M users and keep two backups.", {}, "Deploy to 3000000 users and keep 2 backups."),
    ("Worth 10K, maybe 2.5M, never 3m.", {}, "Worth 10000, maybe 2500000, never 3m."),

    ("Edge: 10K.", {"use_commas": True}, "Edge: 10,000."),

    ("Edge: 10K, 2.5M, and 1G.", {}, "Edge: 10000, 2500000, and 1000000000."),
    ("Edge: 10K.", {"use_commas": False}, "Edge: 10000."),
    ("Edge: 10K.", {"use_commas": True}, "Edge: 10,000."),
    ("not a suffix: 10Km and 2.5Ms", {}, "not a suffix: 10Km and 2.5Ms"),
    ("we raised 3M dollars and spent 2K", {"use_commas": True}, "we raised 3,000,000 dollars and spent 2,000"),

    ("I have $10K and one dreams.", {}, "I have $10000 and 1 dreams."),
]


# ---------------------------------------------------------------------------
# Ordinal & repetition tales: first/second, once/twice, and nth-time phrases
# ---------------------------------------------------------------------------

ORDINAL_AND_TIME_TALES = [
    # --- basic ordinals ---
    ("first place", {}, "1st place"),
    ("second attempt", {}, "2nd time"),
    ("third try", {}, "3rd time"),
    ("fourth wall", {}, "4th wall"),
    ("twenty-first century", {}, "21st century"),
    ("one hundred seventy-second", {}, "172nd"),
    ("one hundred and first", {}, "101st"),
    ("11th hour", {}, "11th hour"),

    # --- ordinals with formatting ---
    ("first prize", {"fmt_ordinal": "[%n%o]"}, "[1st] prize"),
    ("second prize", {"fmt_ordinal": ""}, " prize"),

    # --- once / twice / thrice ---
    ("once", {}, "1 time"),
    ("twice", {"fmt_rep": "%nx"}, "2x"),
    ("thrice", {"fmt_rep": "%nx"}, "3x"),

    # --- explicit times ---
    ("one time", {}, "1 time"),
    ("two times", {"fmt_rep": "%nx"}, "2x"),
    ("1 time", {"fmt_rep": "%nx"}, "1x"),
    ("3 times", {"fmt_rep": "%nx"}, "3x"),
    ("he tried two times", {"fmt_rep": "%nx"}, "he tried 2x"),

    # --- nth time ---
    ("first time", {}, "1st time"),
    ("second time", {}, "2nd time"),
    ("twenty-first time", {"fmt_nth_time": "%n%ox"}, "21stx"),
    ("one hundredth time", {"fmt_nth_time": "%n%ox"}, "100thx"),
    ("five hundredth time", {"fmt_nth_time": "%n%o occurence"}, "500th occurence"),

    # --- custom repetition formatting ---
    ("twice", {"fmt_rep": "%n×"}, "2×"),
    ("third time", {"fmt_nth_time": "<%n%o x>"}, "<3rd x>"),

    # --- boundaries ---
    ("first and second time", {}, "1st and 2nd time"),
    ("once and twice", {"fmt_rep": "%nx"}, "1x and 2x"),
    ("one time and two", {"fmt_rep": "%nx"}, "1x and 2"),
]

ROMAN_TALES = [
    ("Chapter IV", {"support_roman": True}, "Chapter 4"),
    ("Henry VIII", {"support_roman": True}, "Henry 8"),
    ("Year MMXXIII", {"support_roman": True}, "Year 2023"),
    ("Section IX", {"support_roman": True}, "Section 9"),
    ("Part iii", {"support_roman": True}, "Part 3"),
    ("Volume XL", {"support_roman": True}, "Volume 40"),
    ("Not a roman numeral: MMXIXI", {"support_roman": True}, "Not a roman numeral: MMXIXI"),
    ("Mixed: Chapter IV and Section X", {"support_roman": True}, "Mixed: Chapter 4 and Section 10"),
    ("Without support: Chapter IV", {"support_roman": False}, "Without support: Chapter IV"),
]

SIGNED_TALES = [
    ("negative five", {}, "-5"),
    ("positive ten", {}, "+10"),
    ("neg 3", {}, "-3"),
    ("pos 7", {}, "+7"),
    ("- 100", {}, "-100"),
    ("+ 50", {}, "+50"),
    ("negative one hundred", {}, "-100"),
    ("positive two thousand", {}, "+2000"),
    ("it was negative five degrees", {}, "it was -5 degrees"),
    ("score: + 10", {}, "score: +10"),
    ("no sign 5", {}, "no sign 5"),
    ("negative-five", {}, "negative-5"),
    ("positive-six", {}, "positive-6"),
    ("negative five and positive six", {}, "-5 and +6"),
    ("neg 5 and pos 6", {}, "-5 and +6"),
    ("value is - 5", {}, "value is -5"),
    ("value is + 5", {}, "value is +5"),
    ("negative one hundred and one", {}, "-101"),
    ("positive one hundred and one", {}, "+101"),
    ("negative 5 and 6", {}, "-5 and 6"), # "negative" only applies to 5
    ("negative 5 and negative 6", {}, "-5 and -6"),
    ("disabled: negative five", {"parse_signs": False}, "disabled: negative 5"),
]
DECIMAL_TALES = [
    ("zero point five", {}, "0.5"),
    ("point five", {}, "0.5"),
    ("two point zero five", {}, "2.05"),
    ("one hundred and one point two", {}, "101.2"),
    ("negative zero point five", {}, "-0.5"),
    ("we shipped zero point five days early", {}, "we shipped 0.5 days early"),
    ("we shipped oh point five days early", {}, "we shipped 0.5 days early"),
    ("we shipped oh point oh five seven days early", {}, "we shipped 0.057 days early"),
    ("we shipped five thousand point two days early", {}, "we shipped 5000.2 days early"),
]

THS = [
    ("the hundredth time", {}, "the 100th time"),
    ("the fifth time", {}, "the 5th time"),
    ("the fourth amendment", {}, "the 4th amendment"),
    ("one hundredth", {"do_simple_evals": False}, "1/100"),
    ("one hundredth", {}, "0.01"),
    ("1 hundredth", {"do_simple_evals": False}, "1/100"),
    ("1 hundredth", {}, "0.01"),
    ("1 100th", {}, "0.01"),
    ("two hundredths", {}, "0.02"),
    ("five thousandths", {}, "0.005"),
    ("five point 8 millionth", {}, "0.0000058"),
    ("one third", {}, "1/3"),
    ("a third", {}, "1/3"),
    ("two thirds", {}, "2/3"),
    ("negative 5 thirds", {}, "-5/3"),
    ("two halves", {}, "1"),
    ("half", {"do_simple_evals": False}, "1/2"),
    ("half", {}, "0.5"),
    ("one over", {}, "1 over"),
    ("one over five", {"do_simple_evals": False}, "1/5"),
    ("one over five", {}, "0.2"),
    ("two divided by three", {}, "2/3")
]

MATH = [
    ("five times", {}, "5 times"),
    ("five times ten", {"do_simple_evals": False}, "5*10"),
    ("five times ten", {}, "50"),
    ("two point eight multiplied by twenty two point 7", {"do_simple_evals": False}, "2.8*22.7"),
    ("two point eight multiplied by twenty two point 7", {}, "63.56"),
    ("five occurences of 10", {}, "50"),
    ("seven of nine", {}, "7/9"),
    ("6 into 11", {}, "6/11"),
    ("seven point five plus 8", {}, "15.5"),
    ("two to the power of 3", {"do_simple_evals": False}, "2**3"),
    ("two to the power of 3", {"power": "^", "do_simple_evals": False}, "2^3"),
    ("two to the third power", {"power": "^"}, "8"),
    ("two to the third", {"power": "^"}, "8"),
    ("two to the five hundredth power", {"power": "^", "do_simple_evals": False}, "2^500"),
    ("two to the one hundred twenty-seventh power", {"power": "^", "do_simple_evals": False}, "2^127"),
    ("two to the one hundred twenty-seventh power", {"power": "^"}, "170141183460469231731687303715884105728"),
    ("two to the one thousand twenty-seventh power", {"power": "^"}, "1438154507889852726183444152631219786894381583153845258187440649261861406444007705061667818579260288168960911038971146861270318150515332979942779445115792995022143147398923882210417756809968752955624663616680046150705205458739703051791304884326617897306804085476690385919577967507837730438682850636993793097728"),
    ("square root of 5", {"power": "^"}, "5^(1/2)"),
    ("square root of 5", {"power": "^", "do_fraction_evals": True, "res": 3}, "2.236"),
    ("square root of 5", {"power": "^", "do_fraction_evals": True, "res": 10}, "2.2360679775"),
    ("5th root of 32", {}, "32**(1/5)"),
]

SCI = [
  ("six e five", {}, "600000"),
  ("6 e -5", {}, "0.00006"),
  ("six times ten to the fifth", {"power":"^", "mult": " x "}, "600000"),   # via 6*10^5 -> 6e5
  ("6*10^5", {}, "600000"),
  ("6*10**(5)", {"do_simple_evals": False}, "6*(10**(5))"),
]


ACTUAL_MATH = [
    ("one and a half", {}, "1.5"),
    ("one and a third", {}, "4/3"),
    ("one and a third", {"res": 3}, "1.333"),
    ("one and two thirds", {"res": 3}, "1.667"),
    ("one and two thirds", {"res": 4}, "1.6667"),
    ("one point five", {"res": 4}, "1.5"),
    ("one and a half", {"res": 4}, "1.5"),

    ("a day and a half", {}, "1.5 days"),# right now is 'a day and 1/2'
    ("a day and a half", {}, "1.5 days"),# right now is 'a day and a half'
    ("a day and a half an hour", {"do_simple_evals": False}, "a day and 1/2 an hour"),
    ("five days and a half", {}, "5.5 days"),
]

MORE = [
    ("a dozen eggs", {}, "12 eggs"),
    ("five dozen donuts", {"do_simple_evals": False}, "5*12 donuts"),
    ("five dozen donuts", {}, "60 donuts"),
    ("8 sets of 3 cds", {"do_simple_evals": False}, "8*3 cds"),
    ("8 sets of 3 cds", {}, "24 cds"),
    ("8 hours", {"config": "units"}, "28800 s"),
    ("8hours", {"config": "units"}, "28800s"),
    ("8hr", {"config": "units"}, "28800s"),
    ("8hr and 5min", {"config": "units"}, "29100s"),
    ("8hr5min", {"config": "units"}, "29100s"),
    ("half of an hour after sunrise", {"units": Unit("hour",value=60, new_unit="minute")}, "30 minutes after sunrise"),
    ("an hour and a half after sunset", {}, "an hour and 0.5 after sunset"),
    ("an hour and a half after sunset", {"units": Unit("hour", base=True)}, "1.5 hours after sunset"),
    ("an hour and a half after sunset", {"units": Unit("hour",value=60, new_unit="minute")}, "90 minutes after sunset"),
    ("five and two thirds hours after noon", {}, "17/3 hours after noon" ),
    ("five and two thirds hours after noon", {"units": Unit("hour", value=60, new_unit="minute")}, "340 minutes after noon" ),
    ("five minutes and two thirds hours after noon", {"units": Unit("hour", value=60, new_unit="minute")}, "45 minutes after noon" ),
    ("an hour and 22 minutes after noon", {"units": Unit("hour", value=60, new_unit="minute")}, "82 minutes after noon" ),
    ("an hour and 22 minutes and 43 seconds after noon", {"units": units.seconds}, "4963 s after noon" ),
    ("an hour and 23 minutes minus 17 seconds after noon", {"units": units.seconds}, "4963 s after noon" ),
    ("5 seconds less than a minute", {"units": units.seconds}, "55 s" ),
    ("an hour and a half less than two hours",{"units": Unit("hour", base=True)}, "0.5 hours" ),
    ("5.5 minutes less than two hours",{"units": Unit("hour", value=60, new_unit="minute")}, "114.5 minutes" ),
    ("5 minutes and 35 seconds less than two hours",{"units": units.seconds}, "6865 s" ),
]

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
    *ACTUAL_MATH,
    *MORE
]

def demo(text, raise_exc=False, **kwargs):
    k = ", ".join(f"{k2}={v2}" for k2, v2 in kwargs.items())
    cs = f", {k}" if k else ""
    print(f"digitize('{text}'{cs}) => ")
    try:
        out = digitize(text, **kwargs)
        print(f"\t'{out}'")
        return out
    except Exception as e:
        print(f"\t{type(e)}: {e}")
        if raise_exc:
            raise
def test_digitize(text, kwargs, expected):
    assert demo(text, raise_exc=True, **kwargs) == expected


def test_many(suite):
    for i, (text, kwargs, x) in enumerate(suite):
        print(f"{i}. ", end="")
        test_digitize(text, kwargs, x)

def loop(raise_exc=False, **kwargs):
    try:
        while True:
            demo(input(), raise_exc=raise_exc, **kwargs)
    except KeyboardInterrupt:
        exit(1)

def _get_suite(name: str):
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
    if name in {"actual_math"}:
        return ACTUAL_MATH
    if name in {"more"}:
        return MORE
    raise ValueError(f"unknown suite: {name}")

def _parse_kwargs(pairs: list[str]) -> dict:
    """
    Parse CLI kwargs like:
      --kw config=token --kw use_commas=true --kw res=10 --kw power=^
    """
    out: dict = {}
    for item in pairs:
        if "=" not in item:
            raise ValueError(f"bad --kw {item!r}; expected key=value")
        k, v = item.split("=", 1)
        k = k.strip()
        v = v.strip()

        vl = v.lower()
        if vl in {"true", "false"}:
            out[k] = (vl == "true")
            continue
        if vl in {"none", "null"}:
            out[k] = None
            continue

        # int?
        try:
            if re.fullmatch(r"[+-]?\d+", v):
                out[k] = int(v)
                continue
        except Exception:
            pass

        # float?
        try:
            if re.fullmatch(r"[+-]?\d+\.\d+", v):
                out[k] = float(v)
                continue
        except Exception:
            pass

        out[k] = v
    return out

def demo_loop(suite="test", **kwargs):
    suite = _get_suite(suite)
    test_many(suite)
    loop(**kwargs)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv

    ap = argparse.ArgumentParser(
        prog="digitize",
        description="Convert number-words to digits and (optionally) evaluate simple math.",
    )

    ap.add_argument(
        "text",
        nargs="*",
        help="Input text. If omitted, reads from stdin (single pass).",
    )

    ap.add_argument(
        "-m",
        "--mode",
        choices=["single", "loop", "tests", "examples", "demo"],
        default="single",
        help="single: digitize once (default). loop: REPL. tests/examples: run suite. demo: run tests then REPL.",
    )

    ap.add_argument(
        "--suite",
        default="tests",
        help="Suite name for mode=tests/examples/demo (default: tests).",
    )

    ap.add_argument(
        "--kw",
        action="append",
        default=[],
        metavar="KEY=VAL",
        help="Pass digitize() kwargs (repeatable). Example: --kw config=token --kw res=10 --kw power=^",
    )

    ap.add_argument(
        "--newline",
        action="store_true",
        help="When reading stdin (single mode), process line-by-line instead of whole blob.",
    )

    args = ap.parse_args(argv)
    kwargs = _parse_kwargs(args.kw)

    def _read_stdin() -> str:
        return sys.stdin.read()

    def _run_single_text(text: str) -> None:
        out = digitize(text, **kwargs)
        sys.stdout.write(out)
        if not out.endswith("\n"):
            sys.stdout.write("\n")

    # ---- dispatch ----
    if args.mode in {"tests", "examples"}:
        suite = _get_suite(args.suite if args.mode == "tests" else "examples")
        test_many(suite)
        return 0

    if args.mode == "demo":
        demo_loop(args.suite, **kwargs)
        return 0

    if args.mode == "loop":
        loop(**kwargs)
        return 0

    # single (default)
    if args.text:
        _run_single_text(" ".join(args.text))
        return 0

    # stdin path
    data = _read_stdin()
    if args.newline:
        for line in data.splitlines(True):
            if line.endswith("\n"):
                print(digitize(line[:-1], **kwargs))
            else:
                print(digitize(line, **kwargs))
    else:
        _run_single_text(data.rstrip("\n"))

    return 0


def build_md_table(suite):
    results = []
    def limit_width(s, n = 60):
        if len(s) > n:
            return s[:n - 3] + "..."
        return s
    for prompt, params, _ in suite:
        if "\n" in prompt:
            continue
        out = digitize(prompt, **params)
        results.append({
            "prompt": limit_width(str(prompt)),
            "params": limit_width(str(params)),
            "output": limit_width(str(out)),
        })

    headers = ["prompt", "output", "params"]

    # compute column widths
    widths = {
        h: max(len(h), max(len(row[h]) for row in results))
        for h in headers
    }

    def row(values):
        return "| " + " | ".join(
            values[h].ljust(widths[h]) for h in headers
        ) + " |"

    lines = []
    lines.append(row({h: h for h in headers}))
    lines.append("| " + " | ".join("-" * widths[h] for h in headers) + " |")

    for r in results:
        lines.append(row(r))

    return "\n".join(lines)



if __name__ == "__main__":
    print(build_md_table(TESTS))
    # # pass
    # # loop(config="units", raise_exc=True)
    # # loop(raise_exc=True)
    # # print("simple_eval", simple_eval("5*12 donuts"))
    # if not sys.argv[1:]:
    #     raise SystemExit(main(["", "-m", "demo"]))
    # raise SystemExit(main())