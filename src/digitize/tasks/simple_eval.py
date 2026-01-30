import ast
import re
from decimal import Decimal, localcontext, ROUND_HALF_UP
from fractions import Fraction
from math import isqrt

from digitize.tasks.itd import is_terminating_decimal

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
                    if is_terminating_decimal(d):
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