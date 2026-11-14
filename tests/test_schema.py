import pandas as pd

from enahoharm import config, schema


def _dictionary():
    return pd.DataFrame(
        {
            "year": [2019, 2024, 2024, 2019],
            "module": ["02", "02", "02", "05"],
            "name": ["p207", "p207", "p999", "i524a1"],
            "label": ["sexo", "sexo", "algo nuevo", "(imputado, deflactado, anualizado) ingreso"],
            "value_labels": ["1=hombre; 2=mujer", "1=hombre; 2=mujer", "", ""],
        }
    )


def _targets():
    return [
        schema.Target("sex", "person", "direct", "Sex", "02", ["p207"], []),
        schema.Target("income_dependent_main", "person", "direct", "Income", "05", ["i524a1"], []),
    ]


def test_schema_is_consistent():
    targets = schema.load(config.SCHEMA)
    names = [t.name for t in targets]
    assert len(names) == len(set(names))
    assert 40 <= len(names) <= 60
    direct = {t.name for t in targets if t.kind == "direct"}
    for t in targets:
        assert t.unit in ("household", "person")
        if t.kind == "derived":
            assert t.inputs and set(t.inputs) <= direct, t.name
        else:
            assert t.source_names, t.name
    sources = [s for t in targets for s in t.source_names]
    assert len(sources) == len(set(sources)), "a source name maps to two targets"


def test_ground_truth_auto_and_none():
    gt = schema.ground_truth(_dictionary(), _targets(), manual=pd.DataFrame(columns=["year", "module", "name", "truth"]))
    t = gt.set_index(["year", "name"])["truth"]
    assert t[(2019, "p207")] == "sex"
    assert t[(2024, "p999")] == "none"
    assert set(gt["source"]) == {"auto"}
    assert list(gt.columns) == ["year", "module", "name", "truth", "source"]


def test_manual_overrides_auto():
    manual = pd.DataFrame({"year": [2024], "module": ["02"], "name": ["p999"], "truth": ["sex"]})
    gt = schema.ground_truth(_dictionary(), _targets(), manual=manual)
    row = gt[(gt.year == 2024) & (gt.name == "p999")].iloc[0]
    assert row["truth"] == "sex" and row["source"] == "manual"
    assert len(gt) == 4


def test_draft_manual_flags_label_changes_and_audit_sample():
    d = pd.concat(
        [
            _dictionary(),
            pd.DataFrame(
                {
                    "year": [2024] + [2019] * 40,
                    "module": ["05"] * 41,
                    "name": ["i524a1"] + [f"x{i}" for i in range(40)],
                    "label": ["(imp., anual) ingreso"] + ["otra"] * 40,
                    "value_labels": [""] * 41,
                }
            ),
        ],
        ignore_index=True,
    )
    m = schema.draft_manual(d, _targets(), n_audit=10, seed=2026)
    changed = m[m.reason == "label_changed"]
    assert set(changed.name) == {"i524a1"}
    assert (m.reason == "audit").sum() == 10
    assert m["reviewed_by"].isna().all() or (m["reviewed_by"] == "").all()
    assert {"year", "module", "name", "label", "truth", "reason", "reviewed_by"} <= set(m.columns)
