"""Map source variables to the target schema with Claude, caching every answer on disk.

The API key is read only by the Anthropic SDK from ANTHROPIC_API_KEY. This module never reads, prints or
stores it, and wraps API errors so their messages cannot leak request details.
"""
import hashlib
import json
import logging
import os
from pathlib import Path

import pandas as pd

from . import prompts
from .schema import Target

log = logging.getLogger(__name__)

MODEL = "claude-haiku-5-5"
# USD per million tokens, Claude Haiku 5.5 list price for prompts up to 100K tokens (claude-api skill, 2026-11).
PRICES = {"input": 0.10, "output": 0.50, "cache_write": 0.125, "cache_read": 0.01}
OUT_COLUMNS = ["year", "module", "name", "proposed", "confidence", "justification", "status"]


class CacheMiss(Exception):
    """Answers are missing from the cache and no API key is available."""


class BudgetExceeded(Exception):
    """Cumulative estimated spend went over the budget."""


class LLMError(Exception):
    """An API call failed (message deliberately carries no request details)."""


def cache_key(model: str, prompt: str) -> str:
    return hashlib.sha256(f"{model}\n{prompt}".encode()).hexdigest()


def cost_usd(usage: dict) -> float:
    return (
        usage.get("input_tokens", 0) * PRICES["input"]
        + usage.get("output_tokens", 0) * PRICES["output"]
        + usage.get("cache_creation_input_tokens", 0) * PRICES["cache_write"]
        + usage.get("cache_read_input_tokens", 0) * PRICES["cache_read"]
    ) / 1e6


def _batches(dictionary: pd.DataFrame, batch_size: int):
    d = dictionary.copy()
    d["module"] = d["module"].astype(str).str.zfill(2)
    for _, g in d.groupby(["year", "module"], sort=True):
        for i in range(0, len(g), batch_size):
            yield g.iloc[i : i + batch_size]


def _call(client, model: str, system: str, user: str, schema: dict) -> dict:
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=16000,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
            output_config={"effort": "medium", "format": {"type": "json_schema", "schema": schema}},
        )
    except Exception as e:  # noqa: BLE001 - re-raised without the original message
        status = getattr(e, "status_code", None)
        raise LLMError(f"LLM call failed ({type(e).__name__}, status {status})") from None
    text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    u = resp.usage
    usage = {k: int(getattr(u, k, 0) or 0) for k in
             ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")}
    return {"model": model, "text": text, "usage": usage, "stop_reason": resp.stop_reason}


def _parse(text: str, batch: pd.DataFrame, allowed: set[str]) -> pd.DataFrame:
    rows = {n: {"proposed": None, "confidence": None, "justification": "", "status": "invalid"}
            for n in batch["name"]}
    try:
        items = json.loads(text)["mappings"]
    except (json.JSONDecodeError, KeyError, TypeError):
        items = []
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict) or it.get("name") not in rows:
            continue
        target, conf = it.get("target"), it.get("confidence")
        ok = target in allowed and isinstance(conf, (int, float)) and 0 <= conf <= 1
        rows[it["name"]] = {"proposed": target, "confidence": conf,
                            "justification": str(it.get("justification", "")), "status": "ok" if ok else "invalid"}
    out = batch[["year", "module", "name"]].copy()
    for col in ("proposed", "confidence", "justification", "status"):
        out[col] = [rows[n][col] for n in batch["name"]]
    return out


def map_variables(dictionary: pd.DataFrame, targets: list[Target], cache_dir: Path, client=None,
                  model: str = MODEL, batch_size: int = 40, budget_usd: float = 10.0) -> pd.DataFrame:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    system = prompts.system_prompt(targets)
    schema = prompts.answer_schema(targets)
    allowed = {t.name for t in targets if t.kind == "direct"} | {"none"}

    plan = []
    for batch in _batches(dictionary, batch_size):
        user = prompts.render(batch)
        plan.append((batch, user, cache_dir / f"{cache_key(model, system + chr(10) + user)}.json"))

    missing = [b for b, _, path in plan if not path.exists()]
    if missing and client is None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            names = sorted({f"{y}/{m}/{n}" for b in missing for y, m, n in zip(b["year"], b["module"], b["name"])})
            shown = ", ".join(names[:20]) + (" ..." if len(names) > 20 else "")
            raise CacheMiss(f"{len(names)} variables have no cached LLM answer and ANTHROPIC_API_KEY is not set: {shown}")
        import anthropic

        client = anthropic.Anthropic()

    spent, frames = 0.0, []
    for batch, user, path in plan:
        if path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
        else:
            record = _call(client, model, system, user, schema)
            path.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
            spent += cost_usd(record["usage"])
            log.info("LLM batch %s/%s (%d vars): spent so far USD %.4f", batch["year"].iloc[0],
                     batch["module"].iloc[0], len(batch), spent)
            if spent > budget_usd:
                raise BudgetExceeded(f"estimated spend USD {spent:.2f} exceeds budget USD {budget_usd:.2f}")
        frames.append(_parse(record["text"], batch, allowed))
    return pd.concat(frames, ignore_index=True)[OUT_COLUMNS]


def cache_usage(cache_dir: Path) -> dict:
    """Sum tokens and cost over every cached answer."""
    tot = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cache_creation_input_tokens": 0,
           "cache_read_input_tokens": 0}
    for f in Path(cache_dir).glob("*.json"):
        u = json.loads(f.read_text(encoding="utf-8"))["usage"]
        tot["calls"] += 1
        for k in u:
            tot[k] = tot.get(k, 0) + u[k]
    tot["cost_usd"] = round(cost_usd(tot), 4)
    return tot
