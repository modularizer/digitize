FORMATTING_TALES = [
    ("one hundred and thirty five", {"use_commas": True}, "135"),
    ("one hundred and thirty five thousand", {"use_commas": True}, "135,000"),
    ("one hundred and thirty five pigs", {"fmt": "[%n]"}, "[135] pigs"),
    ("he saw one hundred and thirty five birds", {"fmt": ""}, "he saw  birds"),

    ("one million and two hundred and thirty four", {"use_commas": True}, "1,000,234"),
    ("balance: 0,000 and one", {}, "balance: 0,000 and 1"),
    ("he saw one hundred and one birds", {"fmt": "<%n>"}, "he saw <101> birds"),
    ("he saw one hundred and one birds", {"fmt": ""}, "he saw  birds"),
]