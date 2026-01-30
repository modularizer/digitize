EDGE_CASE_CHRONICLES = [
    ("someone said one hundred and nothing else", {}, "someone said 100 and nothing else"),
    ("stone and one hundred and one", {}, "stone and 101"),
    ("one and one is two", {}, "1 and 1 is 2"),
    ("one hundred and thirty-five and six", {}, "135 and 6"),
    ("one hundred and thirty five-six", {}, "129"),
    (" ", {}, " "),
    ("", {}, ""),
    ("one... and then one hundred and one!!!", {}, "1... and then 101!!!")
]