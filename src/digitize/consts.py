PI_DECIMAL = "3.14159265358979323846264338327950288419716939937510"

N019 = [
    "zero","one","two","three","four","five","six","seven",
    "eight","nine","ten","eleven","twelve","thirteen","fourteen",
    "fifteen","sixteen","seventeen","eighteen","nineteen"
]
DIGIT_WORD_TO_DIGIT = {
    "o": "0", "oh": "0",
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
}
TENS_WORDS = ["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
MAGNITUDE_VALUE = {
    "thousand": 10**3,
    "million": 10**6,
    "billion": 10**9,
    "trillion": 10**12,
    "quadrillion": 10**15,
}
FRACTION_DEN_WORD = {
    "half": 2, "halves": 2,
    "third": 3, "thirds": 3,

    # "hundredth(s)" etc
    "hundredth": 100, "hundredths": 100,
    "thousandth": 10**3, "thousandths": 10**3,
    "millionth": 10**6, "millionths": 10**6,
    "billionth": 10**9, "billionths": 10**9,
    "trillionth": 10**12, "trillionths": 10**12,
    "quadrillionth": 10**15, "quadrillionths": 10**15,
}
REPEAT_WORDS = {"once": 1, "twice": 2, "thrice": 3}
POWERS = {"squared": 2, "cubed": 3}