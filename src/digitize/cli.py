import argparse
import re
import sys


from digitize.run import test_many, demo_loop, loop
from digitize.bulk import digitize
from digitize.test_cases import get_suite


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
        suite = get_suite(args.suite if args.mode == "tests" else "examples")
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

if __name__ == "__main__":
    if not sys.argv[1:]:
        raise SystemExit(main(["", "-m", "demo"]))
    raise SystemExit(main())