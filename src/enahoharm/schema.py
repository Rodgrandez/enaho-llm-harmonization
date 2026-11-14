"""Target schema and the ground truth used to evaluate the LLM mappings."""
import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
import yaml

GT_COLUMNS = ["year", "module", "name", "truth", "source"]
MANUAL_COLUMNS = ["year", "module", "name", "label", "truth", "reason", "note", "reviewed_by"]
LABEL_SIMILARITY = 0.85  # below this, a schema variable's label is treated as changed between years


@dataclass
class Target:
    name: str
    unit: str
    kind: str
    description: str
    module: str = ""
    source_names: list[str] = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)
    values: str = ""


def load(path: Path) -> list[Target]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return [Target(**{k: v for k, v in t.items()}) for t in raw["targets"]]


def source_map(targets: list[Target]) -> dict[str, str]:
    return {s: t.name for t in targets if t.kind == "direct" for s in t.source_names}


def _key(df: pd.DataFrame) -> pd.Series:
    return df["year"].astype(int).astype(str) + "|" + df["module"].astype(str).str.zfill(2) + "|" + df["name"]


def ground_truth(dictionary: pd.DataFrame, targets: list[Target], manual: pd.DataFrame) -> pd.DataFrame:
    """Auto truth from source_names (else "none"); manual rows override."""
    smap = source_map(targets)
    gt = dictionary[["year", "module", "name"]].copy()
    gt["module"] = gt["module"].astype(str).str.zfill(2)
    gt["truth"] = gt["name"].map(smap).fillna("none")
    gt["source"] = "auto"
    if len(manual):
        m = manual.copy()
        m["module"] = m["module"].astype(str).str.zfill(2)
        override = dict(zip(_key(m), m["truth"]))
        k = _key(gt)
        hit = k.isin(override)
        gt.loc[hit, "truth"] = k[hit].map(override)
        gt.loc[hit, "source"] = "manual"
    return gt[GT_COLUMNS].reset_index(drop=True)


def normalize_label(s: str) -> str:
    """Lower-case, strip accents and punctuation; INEI processing notes like "(imputado, deflactado)" are kept."""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def label_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_label(a), normalize_label(b)).ratio()


def draft_manual(dictionary: pd.DataFrame, targets: list[Target], n_audit: int = 30, seed: int = 2026) -> pd.DataFrame:
    """Rows for human review: schema variables whose label changed between years, plus an audit sample."""
    d = dictionary.copy()
    d["module"] = d["module"].astype(str).str.zfill(2)
    smap = source_map(targets)
    d["truth"] = d["name"].map(smap).fillna("none")
    years = sorted(d["year"].unique())
    rows = []
    relevant = d[d["truth"] != "none"]
    for (module, name), g in relevant.groupby(["module", "name"]):
        if g["year"].nunique() < len(years):
            changed, note = True, "present in only one year"
        else:
            labels = g.sort_values("year")["label"].tolist()
            sim = label_similarity(labels[0], labels[-1])
            changed, note = sim < LABEL_SIMILARITY, f"label similarity {sim:.2f}"
        if changed:
            for _, r in g.iterrows():
                rows.append({**r[["year", "module", "name", "label", "truth"]].to_dict(),
                             "reason": "label_changed", "note": note})
    flagged = {(r["year"], r["module"], r["name"]) for r in rows}
    pool = d[[(y, m, n) not in flagged for y, m, n in zip(d["year"], d["module"], d["name"])]]
    mapped, unmapped = pool[pool["truth"] != "none"], pool[pool["truth"] == "none"]
    half = n_audit // 2
    audit = pd.concat(
        [mapped.sample(min(half, len(mapped)), random_state=seed),
         unmapped.sample(min(n_audit - min(half, len(mapped)), len(unmapped)), random_state=seed)]
    )
    for _, r in audit.iterrows():
        rows.append({**r[["year", "module", "name", "label", "truth"]].to_dict(),
                     "reason": "audit", "note": "random audit of the automatic rule"})
    out = pd.DataFrame(rows)
    out["reviewed_by"] = ""
    return out[MANUAL_COLUMNS].sort_values(["reason", "name", "year"]).reset_index(drop=True)


def load_manual(path: Path) -> pd.DataFrame:
    m = pd.read_csv(path, dtype={"module": str}, keep_default_na=False)
    m["year"] = m["year"].astype(int)
    return m
