import pandas as pd
import pytest

from enahoharm import evaluate, review


def _frames():
    names = ["a", "b", "c", "d", "e", "f"]
    truth = pd.DataFrame({"year": 2024, "module": "02", "name": names,
                          "truth": ["sex", "age", "strata", "cluster_id", "none", "none"], "source": "auto"})
    mapped = pd.DataFrame({"year": 2024, "module": "02", "name": names,
                           "proposed": ["sex", "age", "strata", "dwelling_id", "none", "sex"],
                           "confidence": [0.99, 0.9, 0.7, 0.6, 0.95, 0.85],
                           "justification": "", "status": "ok"})
    return mapped, truth


def test_metrics_hand_example():
    m = evaluate.metrics(*_frames())
    assert m["accuracy_mapped"] == pytest.approx(0.75)
    assert m["recall_mapped"] == pytest.approx(1.0)
    assert m["precision_mapped"] == pytest.approx(3 / 5)
    assert m["accuracy_none"] == pytest.approx(0.5)
    assert m["n_mapped_truth"] == 4 and m["n_invalid"] == 0
    buckets = {b["bucket"]: b for b in m["calibration"]}
    assert buckets["[0.95,1]"]["n"] == 2 and buckets["[0.95,1]"]["accuracy"] == 1.0
    assert buckets["[0.5,0.8)"]["n"] == 2 and buckets["[0.5,0.8)"]["accuracy"] == 0.5


def test_invalid_counts_as_wrong():
    mapped, truth = _frames()
    mapped.loc[0, ["proposed", "confidence", "status"]] = [None, None, "invalid"]
    m = evaluate.metrics(mapped, truth)
    assert m["n_invalid"] == 1
    assert m["accuracy_mapped"] == pytest.approx(0.5)


def test_metrics_include_usage():
    m = evaluate.metrics(*_frames(), usage={"input_tokens": 10, "output_tokens": 5, "cost_usd": 0.01})
    assert m["input_tokens"] == 10 and m["output_tokens"] == 5 and m["cost_usd"] == 0.01


def test_queue_reasons():
    mapped, truth = _frames()
    mapped.loc[5, ["proposed", "confidence", "status"]] = [None, None, "invalid"]
    extra_m = pd.DataFrame({"year": [2019, 2024], "module": "02", "name": ["g", "g"], "proposed": "age",
                            "confidence": 0.99, "justification": "", "status": "ok"})
    extra_t = pd.DataFrame({"year": [2019, 2024], "module": "02", "name": ["g", "g"], "truth": "age", "source": "auto"})
    mapped, truth = pd.concat([mapped, extra_m]), pd.concat([truth, extra_t])
    dictionary = mapped[["year", "module", "name"]].copy()
    dictionary["label"] = ["l"] * 6 + ["edad en años cumplidos", "monto del ingreso anual imputado"]
    q = review.queue(mapped, truth, dictionary, threshold=0.8)
    reasons = dict(zip(q["year"].astype(str) + q["name"], q["reason"]))
    assert reasons["2024f"] == "invalid"
    assert reasons["2024d"] == "disagrees"  # also low confidence: disagreement wins
    assert reasons["2024c"] == "low_confidence"
    assert reasons["2019g"] == "label_changed" and reasons["2024g"] == "label_changed"
    assert "2024a" not in reasons and "2024e" not in reasons
    assert not q.duplicated(["year", "module", "name"]).any()
    assert list(q.columns) == ["year", "module", "name", "label", "proposed", "confidence", "justification",
                               "truth", "reason", "decision", "corrected_target"]
    assert (q["decision"] == "").all() and (q["corrected_target"] == "").all()


def test_regenerated_queue_keeps_human_decisions():
    new = review.queue(*_frames(), dictionary=_frames()[0][["year", "module", "name"]].assign(label="l"))
    old = new.copy()
    old.loc[0, ["decision", "corrected_target"]] = ["correct", "cluster_id"]
    merged = review.keep_decisions(new, old)
    row = merged[merged["name"] == old.loc[0, "name"]].iloc[0]
    assert row["decision"] == "correct" and row["corrected_target"] == "cluster_id"
    assert len(merged) == len(new)
