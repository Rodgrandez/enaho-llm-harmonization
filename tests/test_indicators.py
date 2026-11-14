import numpy as np
import pandas as pd
import pytest

from enahoharm import config, indicators


def _hh(**over):
    df = pd.DataFrame({
        "year": [2024, 2024], "area": ["urban", "rural"], "household_weight": [1.0, 3.0], "household_size": [1, 1],
        "poor": [False, True], "extreme_poor": [False, False], "poverty_status": [3, 2],
    })
    for k, v in over.items():
        df[k] = v
    return df


def test_poverty_weighted():
    p = indicators.poverty(_hh())
    row = p.iloc[0]
    assert row["poverty"] == pytest.approx(75.0)
    assert row["extreme_poverty"] == pytest.approx(0.0)
    assert row["poverty_flag"] == pytest.approx(75.0)
    assert row["n_excluded"] == 0


def test_poverty_population_weight_uses_household_size():
    p = indicators.poverty(_hh(household_size=[3, 1]))
    assert p.iloc[0]["poverty"] == pytest.approx(50.0)  # weights 1*3 and 3*1


def test_poverty_by_area():
    p = indicators.poverty(_hh(), by="area").set_index("area")
    assert p.loc["rural", "poverty"] == pytest.approx(100.0)
    assert p.loc["urban", "poverty"] == pytest.approx(0.0)


def test_excludes_bad_rows():
    df = pd.concat([_hh(), _hh(household_size=[0, 1], household_weight=[np.nan, 2.0])], ignore_index=True)
    df.loc[3, "poor"] = pd.NA
    p = indicators.poverty(df)
    assert p.iloc[0]["n_excluded"] == 2
    assert np.isfinite(p.iloc[0]["poverty"])


def test_labour_rates():
    person = pd.DataFrame({"year": 2019, "age": [10, 20, 30, 40], "employment_weight": [1.0, 1.0, 1.0, 2.0],
                           "employed": [True, True, False, True], "informal": [pd.NA, True, pd.NA, False],
                           "area": "urban"})
    lab = indicators.labour(person).iloc[0]
    assert lab["employment_rate"] == pytest.approx(75.0)  # (1 + 2) / 4 among 14+
    assert lab["informality_rate"] == pytest.approx(100 / 3)  # 1 of weight 3 employed


def test_labour_informality_missing_year_is_nan():
    person = pd.DataFrame({"year": 2024, "age": [20], "employment_weight": [1.0], "employed": [True],
                           "informal": pd.array([pd.NA], dtype="boolean"), "area": "urban"})
    assert np.isnan(indicators.labour(person).iloc[0]["informality_rate"])


def test_benchmark_flags():
    pov = pd.DataFrame({"year": [2019, 2024], "poverty": [20.19, 21.0], "extreme_poverty": [2.85, 5.5]})
    b = indicators.benchmark(pov, config.INEI_OFFICIAL, tol=0.5).set_index(["year", "measure"])
    assert bool(b.loc[(2019, "poverty"), "within_tol"])
    assert not bool(b.loc[(2024, "poverty"), "within_tol"])
    assert b.loc[(2019, "poverty"), "diff"] == pytest.approx(-0.01)


def test_labour_reports_excluded():
    person = pd.DataFrame({"year": 2019, "age": [20, 30, 40], "employment_weight": [1.0, np.nan, 1.0],
                           "employed": [True, True, pd.NA], "informal": pd.NA, "area": "urban"})
    assert indicators.labour(person).iloc[0]["n_excluded"] == 2
