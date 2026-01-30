import re
from dataclasses import dataclass
from typing import Optional, Literal, Iterable

from digitize.consts import PI_DECIMAL
from digitize.sentinel import _sentinel


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
        if n == '1':
            return u
        p = r"\1" + self.replacement.replace("%n", n).replace("%u", u)
        # print(f"{p=}")
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
        # print(f"{s=}")
        if x := re.match( rf"(.*)(?:(\s)|(?<![A-Za-z]))(an?|\d) {pat} and (?:an? )?(\d+(?:\/|\.)\d+)(.*)", s):
            # this ONLY is a match if the thing that follows is NOT a unit (unless it it the same unit
            start_s = x.group(1)
            _s = x.group(2) or ""
            n = 1 if x.group(3).startswith("a") else int(x.group(3))
            f = x.group(4)
            rest = x.group(5)
            rm = re.match(r"(\S+)", rest.strip())
            fw = rm.group(1) if rm else ""
            known_units = set()
            z = [*all_units] + [u.new_unit for u in all_units]
            for u in z:
                if isinstance(u.key, str):
                    known_units.add(u.key)
                    known_units.add(u.plural)
                else:
                    for uk in u.key:
                        known_units.add(uk)
                        known_units.add(guess_plural(uk))
            # print(f"{fw=}, {known_units=}")
            if fw not in known_units:
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
                # print(f"{rep=}")
                s = start_s + _s + f + rep + rest
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

        digit_or_fraction = r"(?:(?:\.|\/)\d+)"
        num_rx = fr"(?:-?\d+{digit_or_fraction}?)"
        n = rf"(?:{num_rx}|(?:an?))"

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



@dataclass
class UnitGroup:
    base: Unit | str | Iterable[str]
    names: dict[int, str|Iterable[str]]
    children: Iterable[Unit] = ()

    def __post_init__(self):
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

def base(*keys):
    return [Unit(k, base=True) for k in keys]


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


UnitMode = Literal["base", "closest", "cascade"]
class unit_modes:
    BASE: UnitMode = "base"
    CLOSEST: UnitMode = "closest"
    CASCADE: UnitMode = "cascade"



def flatten_units(units: Unit | UnitGroup | Iterable[Unit | UnitGroup]):
    # print("flatten_units", units)
    all_units = []
    if isinstance(units, UnitGroup | Unit):
        all_units = units.children
    else:
        for u in units:

            all_units.extend(u.children)
    # print("all_units", all_units)
    return list(sorted(all_units, key=lambda u: len(u.key), reverse=True))


class AllUnits(list):
    # def __new__(cls, *args):
    #     units = args if len(args) > 1 else args[0]
    #     return super().__new__(cls, flatten_units(units))

    def specified_base_units(self):
        return [u for u in self if u.base]

    def known_bu_keys(self):
        known_bu_keys = set()
        for bu in self.specified_base_units():
            if isinstance(bu.key, str):
                known_bu_keys.add(bu.key)
            else:
                for buk in bu.key:
                    known_bu_keys.add(buk)
        return known_bu_keys

    def new_base_units(self):
        known_bu_keys = self.known_bu_keys()


        new_units = set()
        for u in self:
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
        return more_bases

    def all_base_units(self):
        return self.specified_base_units() + self.new_base_units()