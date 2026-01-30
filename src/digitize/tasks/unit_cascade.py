import re
from fractions import Fraction
from typing import Literal

from digitize.tasks.fractions import frac_to_exact_str, frac_to_decimal_str, parse_num_to_fraction, level_to_frac


def get_level_merger(levels, names, *, unit_max_cascade: int | Literal["base"] = None):
    # Freeze exact Fractions for ALL levels (no float leakage)
    L = [level_to_frac(x) for x in levels]   # DESC
    N = list(names)

    def repl(m: re.Match, res: int | None = 1, *, int_cascade_mode: bool = False) -> str:
        num_s = m.group(1)
        n = parse_num_to_fraction(num_s)  # this must also avoid float internally

        if n == 0:
            return f"0{N[-1]}"

        start = next((i for i, lvl in enumerate(L) if n >= lvl), len(L) - 1)

        if unit_max_cascade is None:
            max_units = len(L) - start
        elif unit_max_cascade == "base":
            # allow cascading down to (and including) 1 second, but not below
            base = Fraction(1, 1)
            max_units = sum(1 for x in L[start:] if base <= x <= L[start])
            max_units = max(1, max_units)
        else:
            max_units = max(1, int(unit_max_cascade))

        end = min(len(L), start + max_units)
        single_level = (end - start) == 1

        out: list[str] = []
        remaining = n

        for i in range(start, end):
            lvl = L[i]
            unit = N[i]

            if int_cascade_mode or not single_level:
                q = remaining // lvl
                if q:
                    out.append(f"{q}{unit}")
                    remaining -= q * lvl
                continue

            # single-level: allow fractional/decimal
            v = remaining / lvl
            v_str = frac_to_exact_str(v) if res is None else frac_to_decimal_str(v, res)
            out.append(f"{v_str}{unit}")
            break

        return "".join(out) if out else f"0{N[-1]}"

    return repl