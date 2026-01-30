ORDINAL_AND_TIME_TALES = [
    # --- basic ordinals ---
    ("first place", {}, "1st place"),
    ("second attempt", {}, "2nd time"),
    ("third try", {}, "3rd time"),
    ("fourth wall", {}, "4th wall"),
    ("twenty-first century", {}, "21st century"),
    ("one hundred seventy-second", {}, "172nd"),
    ("one hundred and first", {}, "101st"),
    ("11th hour", {}, "11th hour"),

    # --- ordinals with formatting ---
    ("first prize", {"fmt_ordinal": "[%n%o]"}, "[1st] prize"),
    ("second prize", {"fmt_ordinal": ""}, " prize"),

    # --- once / twice / thrice ---
    ("once", {}, "1 time"),
    ("twice", {"fmt_rep": "%nx"}, "2x"),
    ("thrice", {"fmt_rep": "%nx"}, "3x"),

    # --- explicit times ---
    ("one time", {}, "1 time"),
    ("two times", {"fmt_rep": "%nx"}, "2x"),
    ("1 time", {"fmt_rep": "%nx"}, "1x"),
    ("3 times", {"fmt_rep": "%nx"}, "3x"),
    ("he tried two times", {"fmt_rep": "%nx"}, "he tried 2x"),

    # --- nth time ---
    ("first time", {}, "1st time"),
    ("second time", {}, "2nd time"),
    ("twenty-first time", {"fmt_nth_time": "%n%ox"}, "21stx"),
    ("one hundredth time", {"fmt_nth_time": "%n%ox"}, "100thx"),
    ("five hundredth time", {"fmt_nth_time": "%n%o occurence"}, "500th occurence"),

    # --- custom repetition formatting ---
    ("twice", {"fmt_rep": "%n×"}, "2×"),
    ("third time", {"fmt_nth_time": "<%n%o x>"}, "<3rd x>"),

    # --- boundaries ---
    ("first and second time", {}, "1st and 2nd time"),
    ("once and twice", {"fmt_rep": "%nx"}, "1x and 2x"),
    ("one time and two", {"fmt_rep": "%nx"}, "1x and 2"),
]