ORD_N019 = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
    "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14,
    "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18,
    "nineteenth": 19,
}
ORD_TENS = {
    "twentieth": 20, "thirtieth": 30, "fortieth": 40, "fiftieth": 50,
    "sixtieth": 60, "seventieth": 70, "eightieth": 80, "ninetieth": 90,
}
ORD_ORDERS = {
    "hundredth": 100,
    "thousandth": 10**3,
    "millionth": 10**6,
    "billionth": 10**9,
    "trillionth": 10**12,
    "quadrillionth": 10**15,
}

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