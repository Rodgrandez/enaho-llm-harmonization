import numpy as np
import pandas as pd

from enahoharm import harmonize


def _mapped(names, proposed, status="ok"):
    return pd.DataFrame({"year": 2024, "module": "02", "name": names, "proposed": proposed,
                         "confidence": 0.9, "justification": "", "status": status})


def _reviewed(names, decision, corrected):
    return pd.DataFrame({"year": 2024, "module": "02", "name": names, "decision": decision,
                         "corrected_target": corrected})


def test_human_decision_wins():
    f = harmonize.final_mapping(_mapped(["p207"], ["age"]), _reviewed(["p207"], ["correct"], ["sex"]))
    row = f.iloc[0]
    assert row["target"] == "sex" and row["provenance"] == "human_corrected"


def test_rejected_dropped_and_llm_default():
    f = harmonize.final_mapping(_mapped(["p207", "p208a", "x"], ["sex", "age", "none"]),
                                _reviewed(["p207"], ["reject"], [""]))
    assert list(f["name"]) == ["p208a"]
    assert f.iloc[0]["provenance"] == "llm"


def test_accept_and_rescue_of_missed_variable():
    f = harmonize.final_mapping(_mapped(["p207", "p208a"], ["sex", None], status=["ok", "invalid"]),
                                _reviewed(["p207", "p208a"], ["accept", "correct"], ["", "age"]))
    prov = dict(zip(f["name"], f["provenance"]))
    assert prov == {"p207": "human_accepted", "p208a": "human_corrected"}


def test_derive_area():
    df = pd.DataFrame({"strata": [1, 5, 6, 8]})
    assert harmonize.derive_area(df).tolist() == ["urban", "urban", "rural", "rural"]


def test_derive_poor():
    # 1800 per year / 1 member / 12 = 150 per month: not poor against 120, poor against 160
    df = pd.DataFrame({"household_expenditure": [1800.0, 1800.0], "household_size": [1, 1],
                       "poverty_line": [120.0, 160.0], "extreme_poverty_line": [60.0, 60.0]})
    assert harmonize.derive_poor(df).tolist() == [False, True]
    assert harmonize.derive_extreme_poor(df).tolist() == [False, False]


def test_derive_poor_missing_size_is_na():
    df = pd.DataFrame({"household_expenditure": [1200.0], "household_size": [0],
                       "poverty_line": [120.0], "extreme_poverty_line": [60.0]})
    assert pd.isna(harmonize.derive_poor(df).iloc[0])


def test_derive_labour_income_and_informal():
    df = pd.DataFrame({"income_dependent_main": [100.0, np.nan], "income_independent_main": [np.nan, 50.0]})
    assert harmonize.derive_labour_income(df).tolist() == [100.0, 50.0]
    assert harmonize.derive_informal(df).isna().all()  # no informality variable that year
    assert harmonize.derive_informal(pd.DataFrame({"informal_status": [1, 2]})).tolist() == [True, False]


def test_derive_years_schooling():
    df = pd.DataFrame({"education_level": [1, 4, 5, 6, 10],
                       "last_year_completed": [np.nan, np.nan, 3, 5, 5],
                       "last_grade_completed": [np.nan, 6, np.nan, np.nan, np.nan]})
    assert harmonize.derive_years_schooling(df).tolist() == [0, 6, 9, 11, 16]


def test_missing_weight_rows_flagged():
    hh = pd.DataFrame({"household_weight": [1.5, np.nan, 0.0]})
    assert harmonize.valid_weight(hh, "household_weight").tolist() == [True, False, False]


def test_apply_mapping_renames_and_keeps_keys():
    raw = pd.DataFrame({"conglome": ["1"], "vivienda": ["2"], "hogar": ["11"], "p207": [2], "junk": [0]})
    final = pd.DataFrame({"year": 2024, "module": "02", "name": ["p207"], "target": ["sex"], "provenance": "llm"})
    out = harmonize.apply_mapping(raw, final, 2024, "02")
    assert list(out.columns) == ["cluster_id", "dwelling_id", "household_id", "sex"]


def test_every_derived_target_has_a_function():
    from enahoharm import config, schema

    derived = {t.name for t in schema.load(config.SCHEMA) if t.kind == "derived"}
    assert derived == set(harmonize.DERIVE)
