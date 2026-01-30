ROMAN_TALES = [
    ("Chapter IV", {"support_roman": True}, "Chapter 4"),
    ("Henry VIII", {"support_roman": True}, "Henry 8"),
    ("Year MMXXIII", {"support_roman": True}, "Year 2023"),
    ("Section IX", {"support_roman": True}, "Section 9"),
    ("Part iii", {"support_roman": True}, "Part 3"),
    ("Volume XL", {"support_roman": True}, "Volume 40"),
    ("Not a roman numeral: MMXIXI", {"support_roman": True}, "Not a roman numeral: MMXIXI"),
    ("Mixed: Chapter IV and Section X", {"support_roman": True}, "Mixed: Chapter 4 and Section 10"),
    ("Without support: Chapter IV", {"support_roman": False}, "Without support: Chapter IV"),
]