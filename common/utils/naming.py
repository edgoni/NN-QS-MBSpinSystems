def short_variants(names):
    """{name: label that tells these variants apart}, or '' when there is one.

    Drops the longest common tail of `_`-separated tokens, so the pair above
    reduces to `new_standard` / `standard` instead of two 40-character names
    that differ in one word and would not fit a legend. Whole tokens, not
    characters: one name here is a suffix of the other, and cutting on the
    character-wise common tail left `s` and `new_s`. At least one token always
    survives, so no label comes back empty.
    """
    names = sorted(names)
    if len(names) < 2:
        return {n: "" for n in names}
    parts = {n: n.split("_") for n in names}
    limit = min(len(v) for v in parts.values()) - 1
    i = 0
    while i < limit and len({tuple(v[len(v) - 1 - i:]) for v in parts.values()}) == 1:
        i += 1
    return {n: "_".join(v[:len(v) - i]) for n, v in parts.items()}
