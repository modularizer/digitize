DECIMAL_TALES = [
    ("zero point five", {}, "0.5"),
    ("point five", {}, "0.5"),
    ("two point zero five", {}, "2.05"),
    ("one hundred and one point two", {}, "101.2"),
    ("negative zero point five", {}, "-0.5"),
    ("we shipped zero point five days early", {}, "we shipped 0.5 days early"),
    ("we shipped oh point five days early", {}, "we shipped 0.5 days early"),
    ("we shipped oh point oh five seven days early", {}, "we shipped 0.057 days early"),
    ("we shipped five thousand point two days early", {}, "we shipped 5000.2 days early"),
]