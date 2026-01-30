import re


def replace_multipliers(s: str, fmt_multipliers: str = "%n", use_commas: bool = False):
    suffix_multipliers = {"k": 10**3, "K": 10**3, "M": 10**6, "G": 10**9, "T": 10**12, "P": 10**15}

    def _expand_suffix(m):
        n = float(m.group(1))
        suf = m.group(2)
        value = int(n * suffix_multipliers[suf])
        n_fmt = f"{value:,}" if use_commas else str(value)
        m_fmt = suf
        return fmt_multipliers.replace("%n", n_fmt).replace("%m", m_fmt).replace("%i", f"{m.group(1)}{m.group(2)}")

    return re.sub(
        r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?)([kKMGTP])(?=[^A-Za-z0-9]|$)",
        _expand_suffix,
        s,
    )