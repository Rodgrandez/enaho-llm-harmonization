import pandas as pd

from enahoharm import report

METRICS = {
    "n_variables": 4151, "n_mapped_truth": 172, "n_invalid": 3, "recall_mapped": 0.9651,
    "accuracy_mapped": 0.9302, "precision_mapped": 0.8889, "accuracy_none": 0.995, "accuracy_all": 0.99,
    "calibration": [{"bucket": "[0,0.5)", "n": 10, "accuracy": 0.5}, {"bucket": "[0.5,0.8)", "n": 20, "accuracy": 0.7},
                    {"bucket": "[0.8,0.95)", "n": 100, "accuracy": 0.9}, {"bucket": "[0.95,1]", "n": 4000, "accuracy": 0.99}],
    "input_tokens": 500000, "output_tokens": 90000, "cost_usd": 0.095,
}
BENCH = pd.DataFrame({"year": [2019, 2019], "measure": ["poverty", "extreme_poverty"], "ours": [20.19, 2.85],
                      "inei": [20.2, 2.9], "diff": [-0.01, -0.05], "within_tol": [True, True]})
POVERTY = pd.DataFrame({"year": [2019, 2019, 2024, 2024], "area": ["urban", "rural", "urban", "rural"],
                        "poverty": [14.6, 40.8, 24.8, 39.3]})


def _review(decisions):
    return pd.DataFrame({"reason": ["disagrees", "low_confidence", "invalid"], "decision": decisions,
                         "corrected_target": ["sex", "", ""]})


def test_update_readme_block(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("# Title\nintro\n<!-- RESULTS:START -->\nold\n<!-- RESULTS:END -->\nfooter\n", encoding="utf-8")
    report.update_readme(readme, METRICS, BENCH, _review(["correct", "accept", "reject"]))
    text = readme.read_text(encoding="utf-8")
    assert text.startswith("# Title\nintro\n<!-- RESULTS:START -->") and text.endswith("<!-- RESULTS:END -->\nfooter\n")
    assert "old" not in text
    assert "93.0%" in text  # accuracy on mappable variables
    assert "| 2019 | poverty | 20.19 | 20.2 |" in text
    assert "USD 0.10" in text
    assert "human review pending" not in text


def test_readme_marks_pending_review(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("<!-- RESULTS:START -->\n<!-- RESULTS:END -->\n", encoding="utf-8")
    report.update_readme(readme, METRICS, BENCH, _review(["", "", ""]))
    assert "human review pending" in readme.read_text(encoding="utf-8")


def test_figures_written(tmp_path):
    paths = report.figures(METRICS, POVERTY, _review(["correct", "accept", ""]), tmp_path)
    assert {p.name for p in paths} == {"llm_calibration.png", "poverty_area.png", "review_outcomes.png"}
    assert all(p.stat().st_size > 0 for p in paths)


def test_ground_truth_signoff_reported(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("<!-- RESULTS:START -->\n<!-- RESULTS:END -->\n", encoding="utf-8")
    gt = pd.DataFrame({"reviewed_by": ["Rodrigo Grandez", "", "Rodrigo Grandez"]})
    report.update_readme(readme, METRICS, BENCH, _review(["", "", ""]), ground_truth=gt)
    text = readme.read_text(encoding="utf-8")
    assert "2 of 3 manual ground-truth rows signed off" in text
