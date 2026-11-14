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
    """Carry human decisions from a previous queue into a regenerated one (matched on year, module, name)."""
    o = _norm(old)[KEYS + ["decision", "corrected_target"]].fillna("")
    n = _norm(new).drop(columns=["decision", "corrected_target"])
    merged = n.merge(o, on=KEYS, how="left")
    merged[["decision", "corrected_target"]] = merged[["decision", "corrected_target"]].fillna("")
    return merged[COLUMNS]
