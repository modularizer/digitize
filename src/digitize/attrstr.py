class attrstr(str):
    __slots__ = ("__dict__",)

    def __new__(cls, value, **kwargs):
        x = super().__new__(cls, value)
        x.__dict__.update(kwargs)
        return x

if __name__ == "__main__":
    a = attrstr("hi", test=5)
    b = attrstr("yo")
