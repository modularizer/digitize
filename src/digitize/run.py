from digitize.bulk import digitize
from digitize.test_cases import get_suite


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


def demo_loop(suite="test", **kwargs):
    suite = get_suite(suite)
    test_many(suite)
    loop(**kwargs)
