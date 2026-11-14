import json
import logging
from types import SimpleNamespace

import pandas as pd
import pytest

from enahoharm import llm, prompts
from enahoharm.schema import Target

TARGETS = [
    Target("sex", "person", "direct", "Sex of the person", "02", ["p207"], []),
    Target("age", "person", "direct", "Age in years", "02", ["p208a"], []),
    Target("area", "household", "derived", "Urban/rural", "", [], ["sex"]),
]


def _dictionary(names=("p207", "p208a")):
    return pd.DataFrame(
        {
            "year": [2024] * len(names),
            "module": ["02"] * len(names),
            "name": list(names),
            "label": ["sexo", "edad"][: len(names)],
            "value_labels": ["1=hombre; 2=mujer", ""][: len(names)],
        }
    )


def _response(payload, input_tokens=100, output_tokens=50):
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens,
                              cache_creation_input_tokens=0, cache_read_input_tokens=0),
        stop_reason="end_turn",
    )


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.messages = self

    def create(self, **kwargs):
        self.calls += 1
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


GOOD = {"mappings": [
    {"name": "p207", "target": "sex", "confidence": 0.97, "justification": "label says sex"},
    {"name": "p208a", "target": "age", "confidence": 0.9, "justification": "age in years"},
]}


def test_prompt_lists_only_direct_targets():
    system = prompts.system_prompt(TARGETS)
    assert "sex" in system and "age" in system
    assert "area" not in system.split("TARGETS")[-1]
    assert "p207" in prompts.render(_dictionary())


def test_uses_cache_without_client(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    first = llm.map_variables(_dictionary(), TARGETS, tmp_path, client=FakeClient([_response(GOOD)]))
    again = llm.map_variables(_dictionary(), TARGETS, tmp_path, client=None)
    pd.testing.assert_frame_equal(first, again)
    assert list(again.columns) == ["year", "module", "name", "proposed", "confidence", "justification", "status"]
    assert again.set_index("name").loc["p207", "proposed"] == "sex"
    assert (again["status"] == "ok").all()


def test_cache_miss_without_key_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(llm.CacheMiss) as exc:
        llm.map_variables(_dictionary(), TARGETS, tmp_path, client=None)
    assert "p207" in str(exc.value)


def test_invalid_answers_marked(tmp_path):
    bad = {"mappings": [
        {"name": "p207", "target": "gender_xyz", "confidence": 0.9, "justification": "?"},
        {"name": "p208a", "target": "age", "confidence": 1.7, "justification": "?"},
    ]}
    out = llm.map_variables(_dictionary(), TARGETS, tmp_path, client=FakeClient([_response(bad)]))
    assert (out["status"] == "invalid").all()
    out2 = llm.map_variables(_dictionary(), TARGETS, tmp_path / "b", client=FakeClient([_response("not json")]))
    assert (out2["status"] == "invalid").all() and len(out2) == 2


def test_missing_variable_in_answer_is_invalid(tmp_path):
    partial = {"mappings": [GOOD["mappings"][0]]}
    out = llm.map_variables(_dictionary(), TARGETS, tmp_path, client=FakeClient([_response(partial)]))
    s = out.set_index("name")["status"]
    assert s["p207"] == "ok" and s["p208a"] == "invalid"


def test_budget_abort(tmp_path):
    names = [f"v{i}" for i in range(6)]
    d = pd.DataFrame({"year": 2024, "module": "02", "name": names, "label": "x", "value_labels": ""})
    # 60M input tokens at $0.10/MTok = $6 per call
    responses = [_response({"mappings": [{"name": n, "target": "none", "confidence": 0.9, "justification": ""}
                                         for n in names[i * 2:(i + 1) * 2]]}, input_tokens=60_000_000)
                 for i in range(3)]
    client = FakeClient(responses)
    with pytest.raises(llm.BudgetExceeded):
        llm.map_variables(d, TARGETS, tmp_path, client=client, batch_size=2, budget_usd=10.0)
    assert client.calls == 2
    assert len(list(tmp_path.glob("*.json"))) == 2  # spend already made is kept in the cache


def test_key_not_leaked(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-SECRET")
    caplog.set_level(logging.DEBUG)
    err = RuntimeError("upstream failure")
    with pytest.raises(llm.LLMError) as exc:
        llm.map_variables(_dictionary(), TARGETS, tmp_path, client=FakeClient([err]))
    assert "SECRET" not in str(exc.value)
    assert "SECRET" not in caplog.text
    for f in tmp_path.rglob("*"):
        if f.is_file():
            assert "SECRET" not in f.read_text(encoding="utf-8")
