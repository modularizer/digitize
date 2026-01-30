from digitize.config import COMPLEX_REP_PATTERN

EXAMPLES: list[tuple[str, dict, str]] = [
    ("i have one hundred and one dalmatians", {}, 'i have 101 dalmatians'),
    ("one million and two hundred and thirty four", {"use_commas": True}, '1,000,234'),
    ("Traffic hit 10K requests per second.", {}, 'Traffic hit 10000 requests per second.'),
    ("one hundred and first place", {}, '101st place'),
    ("we shipped on the 11th hour", {}, 'we shipped on the 11th hour'),
    ("she pinged me twice", {}, 'she pinged me 2 times'),
    ("he tried two times and failed", {"fmt_rep": "%nx", "rep_signifiers": r"times?"}, 'he tried 2x and failed'),
    ("third time was the charm", {"fmt_nth_time": "%n-thx", "rep_signifiers": r"times?"}, '3-thx was the charm'),
    ("the second attempt at which it worked", {"fmt_rep": "%nx", "rep_signifiers": COMPLEX_REP_PATTERN}, 'the 2nd time it worked'),
    ("first place, two times, and 3M users", {"config": "token"}, '[NUM=1,ORD=st,OG=first] place, [NUM=2,REP=times,OG=two times], and [NUM=3000000,MULT=M,OG=3M] users'),
]