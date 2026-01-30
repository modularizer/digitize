STORIES = [
    ("one hundred and thirty-five", {}, "135"),
    ("one thousand and one", {}, "1001"),
    ("one thousand and one nights", {}, "1001 nights"),
    ("i have one hundred and one dalmatians", {}, "i have 101 dalmatians"),
    ("she counted one, and then two, and then three", {}, "she counted 1, and then 2, and then 3"),
    ("one hundred and twenty three thousand and four", {}, "123004"),
    ("one million and two hundred and thirty four", {}, "1000234"),
    ("one hundred and", {}, "100 and"),
    ("and one hundred", {}, "and 100"),
    ("he whispered one hundred and thirty five times", {"fmt_rep": "%nx"}, "he whispered 135x"),

    ("one hundred 33 dalmatians", {}, "133 dalmatians"),
    ("one thousand 2 nights", {}, "1002 nights"),
    ("one million two hundred and thirty four", {}, "1000234"),
    ("one hundred and 33", {}, "133"),
    ("one thousand and 2", {}, "1002"),

    ("one hundred and thirty five, exactly", {}, "135, exactly"),
    ("he said one hundred and one.", {}, "he said 101."),
    ("i have (one hundred and one) dalmatians", {}, "i have (101) dalmatians"),
    ("one hundred and one\nnights", {}, "101\nnights"),
    ("one hundred and one  nights", {}, "101  nights"),

    ("one hundred and thirty-five-six", {}, "129"),
    ("one hundred and thirty-five and six and seven", {}, "135 and 6 and 7"),
    ("one and two hundred", {}, "1 and 200"),
    ("one hundred and one and two", {}, "101 and 2"),
]