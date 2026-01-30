from decimal import getcontext, Decimal, ROUND_HALF_UP
from fractions import Fraction

from digitize.tasks.itd import is_terminating_decimal


def parse_num_to_fraction(num: str) -> Fraction:
    if "/" in num:
        a, b = num.split("/", 1)
        return Fraction(int(a), int(b))
    if "." in num:
        d = Decimal(num)
        # exact conversion: Decimal -> Fraction
        n, den = d.as_integer_ratio()
        return Fraction(n, den)
    return Fraction(int(num), 1)


def frac_to_decimal_str(fr: Fraction, places: int) -> str:
    # exact Decimal division, then quantize
    getcontext().prec = max(50, places + 20)
    d = (Decimal(fr.numerator) / Decimal(fr.denominator)).quantize(
        Decimal("1." + "0" * places),
        rounding=ROUND_HALF_UP,
    )
    # keep fixed digits (matches your examples like 5.29380)
    return format(d, "f")

def frac_to_exact_str(fr: Fraction) -> str:
    if fr.denominator == 1:
        return str(fr.numerator)
    # allow decimals ONLY if exact terminating (no rounding error)
    if is_terminating_decimal(fr):
        getcontext().prec = 80
        d = Decimal(fr.numerator) / Decimal(fr.denominator)
        return format(d.normalize(), "f")
    return f"{fr.numerator}/{fr.denominator}"

def level_to_frac(x) -> Fraction:
    # NEVER do Fraction(float). Use Decimal(str(float)) so 0.001 -> exactly 1/1000.
    if isinstance(x, Fraction):
        return x
    if isinstance(x, int):
        return Fraction(x, 1)
    if isinstance(x, Decimal):
        n, d = x.as_integer_ratio()
        return Fraction(n, d)
    if isinstance(x, float):
        # critical: str(x) gives the human decimal, not the full binary repr
        d = Decimal(str(x))
        n, den = d.as_integer_ratio()
        return Fraction(n, den)
    if isinstance(x, str):
        # allow "0.001", "1e-6"
        d = Decimal(x)
        n, den = d.as_integer_ratio()
        return Fraction(n, den)
    raise TypeError(f"Unsupported level type: {type(x)}")