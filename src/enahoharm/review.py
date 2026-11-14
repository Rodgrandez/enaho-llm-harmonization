"""Human review queue: the LLM mappings a person must check before they are used."""
import pandas as pd

from .evaluate import KEYS, _norm, join
from .schema import LABEL_SIMILARITY, label_similarity

COLUMNS = KEYS + ["label", "proposed", "confidence", "justification", "truth", "reason", "decision",
                  "corrected_target"]
PRIORITY = ["invalid", "disagrees", "label_changed", "low_confidence"]


def _label_changed(dictionary: pd.DataFrame) -> set[tuple]:
    d = _norm(dictionary)
    flagged = set()
    for (module, name), g in d.groupby(["module", "name"]):
        if g["year"].nunique() < 2:
            continue
        labels = g.sort_values("year")["label"].tolist()
        if label_similarity(labels[0], labels[-1]) < LABEL_SIMILARITY:
            flagged |= {(y, module, name) for y in g["year"]}
    return flagged


def queue(mapped: pd.DataFrame, truth: pd.DataFrame, dictionary: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    """One row per variable needing review, with the highest-priority reason.

    label_changed only applies to variables that the LLM or the ground truth maps to a target: a label change
    on an out-of-scope variable does not affect the harmonized data.
    """
    j = join(mapped, truth).merge(_norm(dictionary)[KEYS + ["label"]], on=KEYS, how="left")
    valid = j["status"] == "ok"
    relevant = (valid & (j["proposed"] != "none")) | (j["truth"] != "none")
    changed = _label_changed(dictionary)
    is_changed = pd.Series([(y, m, n) in changed for y, m, n in zip(j["year"], j["module"], j["name"])], index=j.index)
    conf = pd.to_numeric(j["confidence"], errors="coerce")
    flags = {
        "invalid": ~valid,
        "disagrees": valid & (j["proposed"] != j["truth"]),
        "label_changed": relevant & is_changed,
        "low_confidence": valid & (conf < threshold),
    }
    j["reason"] = None
    for reason in reversed(PRIORITY):  # highest priority written last
        j.loc[flags[reason], "reason"] = reason
    q = j[j["reason"].notna()].copy()
    q["decision"] = ""
    q["corrected_target"] = ""
    q["_p"] = q["reason"].map(PRIORITY.index)
    return q.sort_values(["_p", "module", "name", "year"])[COLUMNS].reset_index(drop=True)


def keep_decisions(new: pd.DataFrame, old: pd.DataFrame) -> pd.DataFrame:
    """Carry human work from a previous queue into a regenerated one (matched on year, module, name).

    - A decision whose LLM proposal changed since it was made is cleared and flagged in `note`.
    - Decided rows that no longer qualify for the queue are kept, so a decision is never lost.
    - Extra reviewer columns (e.g. `note`) are preserved.
    """
    o = _norm(old).fillna("")
    for col in ("decision", "corrected_target", "proposed"):
        o[col] = o[col].astype(str) if col in o else ""
    extra = [c for c in o.columns if c not in COLUMNS]
    if "note" not in extra:
        o["note"] = ""
        extra.append("note")
    n = _norm(new).drop(columns=["decision", "corrected_target"])
    keep = o[KEYS + ["decision", "corrected_target", "proposed"] + extra].rename(columns={"proposed": "_old_prop"})
    merged = n.merge(keep, on=KEYS, how="left")
    merged[["decision", "corrected_target", "note"]] = merged[["decision", "corrected_target", "note"]].fillna("")
    new_prop = merged["proposed"].fillna("").astype(str)
    stale = (merged["decision"] != "") & merged["_old_prop"].notna() & (merged["_old_prop"] != new_prop)
    merged.loc[stale, "note"] = (merged.loc[stale, "note"] + " [proposal changed from "
                                 + merged.loc[stale, "_old_prop"] + "; decision cleared]").str.strip()
    merged.loc[stale, ["decision", "corrected_target"]] = ""
    in_new = set(zip(merged["year"], merged["module"], merged["name"]))
    gone = o[(o["decision"] != "") & [k not in in_new for k in zip(o["year"], o["module"], o["name"])]]
    out = pd.concat([merged.drop(columns="_old_prop"), gone], ignore_index=True)
    return out[COLUMNS + extra].fillna("")
