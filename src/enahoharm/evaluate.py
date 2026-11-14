"""Evaluate LLM mappings against the ground truth."""
import pandas as pd

KEYS = ["year", "module", "name"]
BUCKETS = [(0.0, 0.5, "[0,0.5)"), (0.5, 0.8, "[0.5,0.8)"), (0.8, 0.95, "[0.8,0.95)"), (0.95, 1.0001, "[0.95,1]")]


def _norm(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"] = df["year"].astype(int)
    df["module"] = df["module"].astype(str).str.zfill(2)
    return df


def join(mapped: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    j = _norm(mapped).merge(_norm(truth)[KEYS + ["truth"]], on=KEYS, how="left", validate="one_to_one")
    j["truth"] = j["truth"].fillna("none")
    valid = j["status"] == "ok"
    j["correct"] = valid & (j["proposed"] == j["truth"])
    return j


def _rate(mask_num: pd.Series, mask_den: pd.Series) -> float | None:
    den = int(mask_den.sum())
    return round(float((mask_num & mask_den).sum()) / den, 4) if den else None


def metrics(mapped: pd.DataFrame, truth: pd.DataFrame, usage: dict | None = None) -> dict:
    j = join(mapped, truth)
    valid = j["status"] == "ok"
    truth_mapped = j["truth"] != "none"
    proposed_mapped = valid & (j["proposed"] != "none")
    out = {
        "n_variables": len(j),
        "n_mapped_truth": int(truth_mapped.sum()),
        "n_invalid": int((~valid).sum()),
        # mappable variables the LLM recognised as mappable (any target)
        "recall_mapped": _rate(proposed_mapped, truth_mapped),
        # mappable variables assigned to the right target (headline)
        "accuracy_mapped": _rate(j["correct"], truth_mapped),
        # of the variables the LLM mapped to some target, share that were right
        "precision_mapped": _rate(j["correct"], proposed_mapped),
        "accuracy_none": _rate(j["correct"], ~truth_mapped),
        "accuracy_all": _rate(j["correct"], pd.Series(True, index=j.index)),
    }
    cal = []
    conf = pd.to_numeric(j["confidence"], errors="coerce")
    for lo, hi, label in BUCKETS:
        m = valid & (conf >= lo) & (conf < hi)
        cal.append({"bucket": label, "n": int(m.sum()), "accuracy": _rate(j["correct"], m)})
    out["calibration"] = cal
    usage = usage or {}
    for k in ("input_tokens", "output_tokens", "cost_usd"):
        out[k] = usage.get(k, 0)
    return out
