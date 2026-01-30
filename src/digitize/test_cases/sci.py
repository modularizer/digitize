SCI = [
  ("six e five", {}, "600000"),
  ("6 e -5", {}, "0.00006"),
  ("six times ten to the fifth", {"power":"^", "mult": " x "}, "600000"),   # via 6*10^5 -> 6e5
  ("6*10^5", {}, "600000"),
  ("6*10**(5)", {"do_simple_evals": False}, "6*(10**(5))"),
]