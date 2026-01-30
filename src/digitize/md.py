from digitize.bulk import digitize


def build_md_table(suite):
    results = []
    def limit_width(s, n = 60):
        if len(s) > n:
            return s[:n - 3] + "..."
        return s
    for prompt, params, _ in suite:
        if "\n" in prompt:
            continue
        out = digitize(prompt, **params)
        results.append({
            "prompt": limit_width(str(prompt)),
            "params": limit_width(str(params)),
            "output": limit_width(str(out)),
        })

    headers = ["prompt", "output", "params"]

    # compute column widths
    widths = {
        h: max(len(h), max(len(row[h]) for row in results))
        for h in headers
    }

    def row(values):
        return "| " + " | ".join(
            values[h].ljust(widths[h]) for h in headers
        ) + " |"

    lines = []
    lines.append(row({h: h for h in headers}))
    lines.append("| " + " | ".join("-" * widths[h] for h in headers) + " |")

    for r in results:
        lines.append(row(r))

    return "\n".join(lines)

if __name__ == "__main__":
    from digitize.test_cases import TESTS
    print(build_md_table(TESTS))