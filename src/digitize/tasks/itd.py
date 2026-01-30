def is_terminating_decimal(d: int) -> bool:
    d = abs(d)
    if d == 0:
        return False
    while d % 2 == 0:
        d //= 2
    while d % 5 == 0:
        d //= 5
    return d == 1