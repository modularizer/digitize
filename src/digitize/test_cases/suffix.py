SUFFIX_SAGAS = [
    ("We raised 3M dollars.", {}, "We raised 3000000 dollars."),
    ("Traffic hit 10K requests per second.", {}, "Traffic hit 10000 requests per second."),
    ("Storage is 2G and climbing.", {}, "Storage is 2000000000 and climbing."),
    ("Budget: 2.5M for phase one.", {}, "Budget: 2500000 for phase 1."),
    ("Big money: 1T reasons to care.", {}, "Big money: 1000000000000 reasons to care."),
    ("Rare air: 1P possibilities.", {}, "Rare air: 1000000000000000 possibilities."),

    ("We raised 3m dollars.", {}, "We raised 3m dollars."),
    ("Traffic hit 10k requests per second.", {}, "Traffic hit 10000 requests per second."),
    ("He wrote 2g on the napkin.", {}, "He wrote 2g on the napkin."),

    ("I have 10K and one dreams.", {}, "I have 10000 and 1 dreams."),
    ("Deploy to 3M users and keep two backups.", {}, "Deploy to 3000000 users and keep 2 backups."),
    ("Worth 10K, maybe 2.5M, never 3m.", {}, "Worth 10000, maybe 2500000, never 3m."),

    ("Edge: 10K.", {"use_commas": True}, "Edge: 10,000."),

    ("Edge: 10K, 2.5M, and 1G.", {}, "Edge: 10000, 2500000, and 1000000000."),
    ("Edge: 10K.", {"use_commas": False}, "Edge: 10000."),
    ("Edge: 10K.", {"use_commas": True}, "Edge: 10,000."),
    ("not a suffix: 10Km and 2.5Ms", {}, "not a suffix: 10Km and 2.5Ms"),
    ("we raised 3M dollars and spent 2K", {"use_commas": True}, "we raised 3,000,000 dollars and spent 2,000"),

    ("I have $10K and one dreams.", {}, "I have $10000 and 1 dreams."),
]