"""Final mapping (LLM proposal + human decisions) and the harmonized household and person tables."""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .evaluate import KEYS, _norm
from .schema import Target

log = logging.getLogger(__name__)

# Join keys are read by their INEI names, independently of the LLM mapping, so a mapping error can never
# break the merge. They are written out under their harmonized names.
RAW_KEYS = {"conglome": "cluster_id", "vivienda": "dwelling_id", "hogar": "household_id", "codperso": "person_id"}
HH_KEYS = ["cluster_id", "dwelling_id", "household_id"]
PERSON_KEYS = HH_KEYS + ["person_id"]
PERSON_MODULES = ["02", "03", "05"]


# ---------------------------------------------------------------------------------------------- mapping
def final_mapping(mapped: pd.DataFrame, reviewed: pd.DataFrame) -> pd.DataFrame:
    """LLM proposals overridden by human decisions; rejected rows and "none" are dropped.

    provenance: llm (not reviewed), human_accepted, human_corrected.
    """
    m = _norm(mapped)
    r = _norm(reviewed)[KEYS + ["decision", "corrected_target"]].fillna("")
    j = m.merge(r, on=KEYS, how="left").fillna({"decision": "", "corrected_target": ""})
    valid = (j["status"] == "ok") & j["proposed"].notna() & (j["proposed"] != "none")
    j["target"] = np.where(valid, j["proposed"], None)
    j["provenance"] = np.where(valid, "llm", None)
    acc = j["decision"] == "accept"
    j.loc[acc & valid, "provenance"] = "human_accepted"
    cor = (j["decision"] == "correct") & (j["corrected_target"] != "")
    j.loc[cor, "target"] = j.loc[cor, "corrected_target"]
    j.loc[cor, "provenance"] = "human_corrected"
    j.loc[cor & (j["corrected_target"] == "none"), "target"] = None
    j.loc[j["decision"] == "reject", "target"] = None
    out = j[j["target"].notna()]
    return out[KEYS + ["target", "provenance"]].reset_index(drop=True)


# ---------------------------------------------------------------------------------------------- derived
def derive_area(df: pd.DataFrame) -> pd.Series:
    """INEI convention: strata 1-5 urban, 6-8 rural."""
    s = pd.to_numeric(df["strata"], errors="coerce")
    return pd.Series(np.select([s.between(1, 5), s.between(6, 8)], ["urban", "rural"], default=None), index=df.index)


def _per_capita_monthly(df: pd.DataFrame) -> pd.Series:
    size = pd.to_numeric(df["household_size"], errors="coerce").where(lambda x: x > 0)
    return df["household_expenditure"] / size / 12


def derive_poor(df: pd.DataFrame) -> pd.Series:
    """Per capita monthly expenditure below the total poverty line."""
    pc = _per_capita_monthly(df)
    return (pc < df["poverty_line"]).astype("boolean").mask(pc.isna() | df["poverty_line"].isna())


def derive_extreme_poor(df: pd.DataFrame) -> pd.Series:
    """Per capita monthly expenditure below the extreme (food) poverty line."""
    pc = _per_capita_monthly(df)
    return (pc < df["extreme_poverty_line"]).astype("boolean").mask(pc.isna() | df["extreme_poverty_line"].isna())


def derive_employed(df: pd.DataFrame) -> pd.Series:
    s = pd.to_numeric(df.get("labour_force_status"), errors="coerce")
    return (s == 1).astype("boolean").mask(s.isna())


def derive_informal(df: pd.DataFrame) -> pd.Series:
    """informal_status == 1; all missing when the year has no informality variable (ENAHO 2024 module 05)."""
    if "informal_status" not in df:
        return pd.Series(pd.NA, index=df.index, dtype="boolean")
    s = pd.to_numeric(df["informal_status"], errors="coerce")
    return (s == 1).astype("boolean").mask(s.isna())


LABOUR_INCOME_PARTS = ["income_dependent_main", "income_independent_main", "income_dependent_secondary",
                       "income_independent_secondary"]


def derive_labour_income(df: pd.DataFrame) -> pd.Series:
    """Sum of the imputed annual monetary income components present (missing components count as 0)."""
    parts = [c for c in LABOUR_INCOME_PARTS if c in df]
    if not parts:
        return pd.Series(np.nan, index=df.index)
    return df[parts].sum(axis=1, min_count=1)


def derive_years_schooling(df: pd.DataFrame) -> pd.Series:
    """Years of schooling from ENAHO p301a level (codes 1-12) and last year (p301b) or grade (p301c).

    Peru: 6 years of primary, 5 of secondary. Approximation documented in docs/methodology.md.
    """
    lvl = pd.to_numeric(df["education_level"], errors="coerce")
    yr = pd.to_numeric(df.get("last_year_completed"), errors="coerce")
    gr = pd.to_numeric(df.get("last_grade_completed"), errors="coerce")
    in_primary = yr.where(yr > 0, gr).fillna(0).clip(0, 6)
    yr0 = yr.fillna(0)
    conds = [lvl.isin([1, 2]), lvl == 3, lvl == 4, lvl == 5, lvl == 6, lvl.between(7, 10), lvl == 11]
    vals = [0, in_primary, 6, 6 + yr0.clip(0, 5), 11, 11 + yr0.clip(0, 7), 16 + yr0.clip(0, 6)]
    return pd.Series(np.select(conds, vals, default=np.nan), index=df.index)


DERIVE = {
    "area": derive_area,
    "years_schooling": derive_years_schooling,
    "employed": derive_employed,
    "informal": derive_informal,
    "labour_income": derive_labour_income,
    "poor": derive_poor,
    "extreme_poor": derive_extreme_poor,
}


def valid_weight(df: pd.DataFrame, col: str) -> pd.Series:
    w = pd.to_numeric(df.get(col), errors="coerce")
    return w.notna() & (w > 0)


# ---------------------------------------------------------------------------------------------- build
def _read(dta: Path, columns: list[str]) -> pd.DataFrame:
    with pd.io.stata.StataReader(dta) as reader:
        varlist = list(reader.variable_labels())  # public call that also reads the header
    by_lower = {v.lower(): v for v in varlist}
    wanted = list(dict.fromkeys(by_lower[c] for c in columns if c in by_lower))
    df = pd.read_stata(dta, columns=wanted, convert_categoricals=False)
    df.columns = [c.lower() for c in df.columns]
    return df


def apply_mapping(raw: pd.DataFrame, final: pd.DataFrame, year: int, module: str) -> pd.DataFrame:
    """Keep the join keys and the mapped columns of one module, renamed to target names."""
    f = final[(final["year"] == year) & (final["module"] == module)]
    out = pd.DataFrame(index=raw.index)
    for raw_key, key in RAW_KEYS.items():
        if raw_key in raw:
            out[key] = raw[raw_key].astype(str).str.strip()
    for name, target in zip(f["name"], f["target"]):
        if name in RAW_KEYS:  # already written as a join key
            continue
        if name not in raw or target in out:
            if target in out:
                log.warning("%s m%s: %s also maps to %s; keeping the first source", year, module, name, target)
            continue
        out[target] = raw[name]
    return out


def _derive_all(df: pd.DataFrame, targets: list[Target], unit: str) -> pd.DataFrame:
    for t in targets:
        if t.kind == "derived" and t.unit == unit and set(t.inputs) & set(df.columns):
            try:
                df[t.name] = DERIVE[t.name](df)
            except KeyError as e:
                log.warning("cannot derive %s: missing %s", t.name, e)
    return df


def build(final: pd.DataFrame, raw_dir: Path = config.RAW, targets: list[Target] | None = None,
          out_dir: Path = config.PROCESSED) -> dict[str, Path]:
    targets = targets or []
    units = {t.name: t.unit for t in targets}
    final = _norm(final)
    households, persons = [], []
    for year in config.YEARS:
        f = final[final["year"] == year]
        hh_names = f.loc[f["module"] == "34", "name"].tolist()
        hh = apply_mapping(_read(Path(raw_dir) / str(year) / "m34.dta", list(RAW_KEYS) + hh_names), f, year, "34")
        hh = hh[[c for c in hh.columns if c in HH_KEYS or units.get(c, "household") == "household"]]
        hh = _derive_all(hh, targets, "household")
        hh["valid_weight"] = valid_weight(hh, "household_weight")
        hh.insert(0, "year", year)
        households.append(hh)

        person = None
        for module in PERSON_MODULES:
            names = f.loc[f["module"] == module, "name"].tolist()
            part = apply_mapping(_read(Path(raw_dir) / str(year) / f"m{module}.dta", list(RAW_KEYS) + names),
                                 f, year, module)
            if person is None:
                person = part
            else:
                new = [c for c in part.columns if c not in person.columns]
                person = person.merge(part[PERSON_KEYS + new], on=PERSON_KEYS, how="left", validate="one_to_one")
        person = _derive_all(person, targets, "person")
        if "area" not in person and "strata" in person:
            person["area"] = derive_area(person)
        person.insert(0, "year", year)
        persons.append(person)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"household": out_dir / "household.parquet", "person": out_dir / "person.parquet"}
    pd.concat(households, ignore_index=True).to_parquet(paths["household"], index=False)
    pd.concat(persons, ignore_index=True).to_parquet(paths["person"], index=False)
    return paths
