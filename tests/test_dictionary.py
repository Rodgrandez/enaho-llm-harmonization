import pandas as pd

from enahoharm import dictionary


def _write_dta(path, labels, value_labels=None):
    df = pd.DataFrame({"P207": [1, 2, 1], "P208A": [30, 41, 7]})
    df.to_stata(
        path,
        write_index=False,
        variable_labels=labels,
        value_labels=value_labels,
        version=118,
    )


def test_reads_names_and_labels(tmp_path):
    p = tmp_path / "m02.dta"
    _write_dta(p, {"P207": "sexo", "P208A": "edad en años"}, {"P207": {1: "hombre", 2: "mujer"}})
    d = dictionary.read_dictionary(p, 2024, "02")
    row = d.set_index("name").loc["p207"]
    assert row["label"] == "sexo"
    assert "1=hombre" in row["value_labels"]
    assert row["year"] == 2024 and row["module"] == "02"
    assert row["n_nonmissing"] == 3
    assert list(d.columns) == ["year", "module", "name", "label", "value_labels", "dtype", "n_nonmissing"]


def test_accents_survive(tmp_path):
    p = tmp_path / "m03.dta"
    _write_dta(p, {"P207": "sexo", "P208A": "año de educación"})
    d = dictionary.read_dictionary(p, 2019, "03")
    label = d.set_index("name").loc["p208a", "label"]
    assert label == "año de educación"
    assert "�" not in "".join(d["label"])


def test_value_labels_truncated(tmp_path):
    p = tmp_path / "m05.dta"
    vl = {i: f"categoria larga numero {i}" for i in range(1, 60)}
    df = pd.DataFrame({"X": [1, 2]})
    df.to_stata(p, write_index=False, variable_labels={"X": "x"}, value_labels={"X": vl}, version=118)
    d = dictionary.read_dictionary(p, 2019, "05")
    assert len(d.loc[0, "value_labels"]) <= 300
