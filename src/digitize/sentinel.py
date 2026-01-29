class Sentinel:
    def __bool__(self):
        return False
    def __repr__(self):
        return ""
    def __str__(self):
        return ""

_sentinel = Sentinel()