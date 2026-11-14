"""Weighted poverty and labour indicators (DuckDB over the harmonized tables) and the INEI benchmark."""
import duckdb
import pandas as pd

WORKING_AGE = 14  # INEI working-age population (PET)


def _group(by: str | None) -> str:
    return "year" + (f", {by}" if by else "")


def poverty(hh: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    """Population-weighted poverty rates in %, weight = household_weight * household_size.

    Households with a missing or non-positive weight, a non-positive size, or no poverty status are excluded
    and counted in n_excluded (per group).
    """
    df = hh.copy()
    w = pd.to_numeric(df["household_weight"], errors="coerce")
    size = pd.to_numeric(df["household_size"], errors="coerce")
    df["ok"] = w.notna() & (w > 0) & size.notna() & (size > 0) & df["poor"].notna() & df["extreme_poor"].notna()
    df["w"] = (w * size).where(df["ok"])
    df["poor_d"] = df["poor"].astype("Float64").astype(float)
    df["ext_d"] = df["extreme_poor"].astype("Float64").astype(float)
    df["flag_d"] = pd.to_numeric(df["poverty_status"], errors="coerce").isin([1, 2]).astype(float)
    g = _group(by)
    q = f"""
        SELECT {g},
               100 * SUM(w * poor_d) FILTER (WHERE ok) / SUM(w) FILTER (WHERE ok) AS poverty,
               100 * SUM(w * ext_d) FILTER (WHERE ok) / SUM(w) FILTER (WHERE ok) AS extreme_poverty,
               100 * SUM(w * flag_d) FILTER (WHERE ok) / SUM(w) FILTER (WHERE ok) AS poverty_flag,
               COUNT(*) FILTER (WHERE NOT ok) AS n_excluded
        FROM df GROUP BY {g} ORDER BY {g}
    """
    return duckdb.query_df(df, "df", q).df()


def labour(person: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    """employment_rate: employed / population 14+; informality_rate: informal / employed (both % with fac500a).

    informality_rate is NaN for a year without an informality variable.
    """
    df = person.copy()
    w = pd.to_numeric(df["employment_weight"], errors="coerce")
    age = pd.to_numeric(df["age"], errors="coerce")
    df["pet"] = (age >= WORKING_AGE) & w.notna() & (w > 0) & df["employed"].notna()
    df["w"] = w.where(df["pet"])
    df["emp_d"] = df["employed"].astype("Float64").astype(float)
    df["inf_d"] = df["informal"].astype("Float64").astype(float) if "informal" in df else float("nan")
    g = _group(by)
    q = f"""
        SELECT {g},
               100 * SUM(w * emp_d) FILTER (WHERE pet) / SUM(w) FILTER (WHERE pet) AS employment_rate,
               100 * SUM(w * inf_d) FILTER (WHERE pet AND emp_d = 1 AND inf_d IS NOT NULL AND NOT isnan(inf_d))
                   / SUM(w) FILTER (WHERE pet AND emp_d = 1 AND inf_d IS NOT NULL AND NOT isnan(inf_d))
                   AS informality_rate,
               COUNT(*) FILTER (WHERE pet) AS n_working_age
        FROM df GROUP BY {g} ORDER BY {g}
    """
    out = duckdb.query_df(df, "df", q).df()
    out["informality_rate"] = out["informality_rate"].astype(float)
    return out


def benchmark(pov: pd.DataFrame, official: dict, tol: float = 0.5) -> pd.DataFrame:
    rows = []
    for _, r in pov.iterrows():
        year = int(r["year"])
        for measure in ("poverty", "extreme_poverty"):
            if year in official and measure in official[year]:
                ours, inei = round(float(r[measure]), 2), official[year][measure]
                rows.append({"year": year, "measure": measure, "ours": ours, "inei": inei,
                             "diff": round(ours - inei, 2), "within_tol": abs(ours - inei) <= tol})
    return pd.DataFrame(rows)
