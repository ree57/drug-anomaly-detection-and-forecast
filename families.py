"""Product families, shared by alerts.py, the watch list and export_dashboard.py (pandas only, no Spark).

Two drug groups (of the same week / origin) are one family if they share a NAME STEM, or if both moved by the same
big % change (|change| >= PCT_MIN, rounded to 2 decimals). Links chain (union-find).

Name stem = first word of the description (split on space and '-') AFTER removing a store-brand / repackager
prefix. Those prefixes are derived from the data: a first word W is a prefix if, for at least MIN_REST_EXISTS
groups, the rest of the name ("GNP ASPIRIN 81 MG" -> "ASPIRIN 81 MG") is itself another group's name, AND W is
followed by at least MIN_NEXT_WORDS different words (it labels unrelated products, unlike "ARMOUR THYROID").
Without this, a store's OTC line (FT cough syrup, FT allergy, ...) chains into one giant family.
"""
import re

import numpy as np
import pandas as pd

PCT_MIN = 30
MIN_REST_EXISTS = 10
MIN_NEXT_WORDS = 20
_STEM = re.compile(r"^([^ |-]+)")


def derive_store_prefixes(descriptions):
    """Return {prefix: (groups, rest_exists, next_words)} for first words that behave like store brands."""
    desc = pd.Series(pd.unique(pd.Series(descriptions)))
    names = set(desc)
    parts = desc.str.extract(r"^([^ ]+) (.+)$").dropna()
    parts.columns = ["word", "rest"]
    parts["rest_exists"] = parts.rest.isin(names)
    parts["next_word"] = parts.rest.str.split(" ").str[0]
    s = parts.groupby("word").agg(groups=("rest", "size"), rest_exists=("rest_exists", "sum"),
                                  next_words=("next_word", "nunique"))
    s = s[(s.rest_exists >= MIN_REST_EXISTS) & (s.next_words >= MIN_NEXT_WORDS)].sort_values("rest_exists", ascending=False)
    return {w: (int(r.groups), int(r.rest_exists), int(r.next_words)) for w, r in s.iterrows()}


def load_store_prefixes(path="data/tables/group_dim"):
    return list(derive_store_prefixes(pd.read_parquet(path, columns=["description"]).description))


def name_stem(description, prefixes=()):
    words = description.split(" ", 1)
    if len(words) == 2 and words[0] in prefixes:
        description = words[1]
    m = _STEM.match(description)
    return m.group(1) if m else description


def family_ids(descriptions, pct_changes, prefixes=()):
    """Family label (index of one member) for each row; rows are ONE week or ONE origin."""
    n = len(descriptions)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    first = {}
    for i, (d, p) in enumerate(zip(descriptions, pct_changes)):
        keys = [("n", name_stem(d, prefixes))]
        if p is not None and not np.isnan(p) and abs(p) >= PCT_MIN:
            keys.append(("p", round(float(p), 2)))
        for k in keys:
            if k in first:
                parent[find(i)] = find(first[k])
            else:
                first[k] = i
    return [find(i) for i in range(n)]


def assign_families(df, by, desc_col, pct_col, prefixes=(), out="family"):
    """Add a family id column: '<by value>|<member index>' computed separately for each value of `by`."""
    df = df.copy()
    df[out] = ""
    for key, idx in df.groupby(by).groups.items():
        sub = df.loc[idx]
        ids = family_ids(sub[desc_col].tolist(), sub[pct_col].to_numpy(dtype=float), prefixes)
        df.loc[idx, out] = [f"{key}|{i}" for i in ids]
    return df
