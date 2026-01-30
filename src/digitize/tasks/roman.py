import re

ROMAN_PATTERN = re.compile(r"^(?=[MDCLXVI])M*(C[MD]|D?C{0,3})(X[CL]|L?X{0,3})(I[XV]|V?I{0,3})$", re.IGNORECASE)
ROMAN_VALUES = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}

def roman_to_int(s: str) -> int:
    s = s.upper()
    num = 0
    for i in range(len(s) - 1):
        if ROMAN_VALUES[s[i]] < ROMAN_VALUES[s[i + 1]]:
            num -= ROMAN_VALUES[s[i]]
        else:
            num += ROMAN_VALUES[s[i]]
    num += ROMAN_VALUES[s[-1]]
    return num