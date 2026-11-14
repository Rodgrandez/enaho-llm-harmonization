"""Command-line entry point: python -m enahoharm.pipeline <stage>."""
import argparse
import logging


def stage_fetch():
    from .fetch import fetch_all

    fetch_all()


def stage_dictionary():
    from .dictionary import build

    d = build()
    print(d.groupby(["year", "module"]).size().to_string())


def stage_draft_ground_truth():
    """Write the draft manual ground truth; never overwrites a file that may hold human review."""
    import pandas as pd

    from . import config, schema

    out = config.VALIDATION / "ground_truth.csv"
    if out.exists():
        raise SystemExit(f"{out} exists (may contain human review); delete it explicitly to redraft")
    d = pd.read_csv(config.INTERIM / "dictionary.csv", dtype={"module": str}, keep_default_na=False)
    m = schema.draft_manual(d, schema.load(config.SCHEMA))
    out.parent.mkdir(parents=True, exist_ok=True)
    m.to_csv(out, index=False, encoding="utf-8")
    print(m["reason"].value_counts().to_string())


def _load_dictionary():
    import pandas as pd

    from . import config

    d = pd.read_csv(config.INTERIM / "dictionary.csv", dtype={"module": str}, keep_default_na=False)
    d["module"] = d["module"].str.zfill(2)
    return d


def stage_llm():
    """Map every variable with the LLM; with no API key it runs from the committed cache only."""
    from . import config, llm, schema

    out = llm.map_variables(_load_dictionary(), schema.load(config.SCHEMA), config.CACHE)
    out.to_csv(config.INTERIM / "llm_mappings.csv", index=False, encoding="utf-8")
    usage = llm.cache_usage(config.CACHE)
    print(f"{len(out)} variables mapped, {(out.status == 'invalid').sum()} invalid; "
          f"{usage['calls']} calls, USD {usage['cost_usd']:.4f}")


def _load_truth(dictionary):
    from . import config, schema

    manual = schema.load_manual(config.VALIDATION / "ground_truth.csv")
    return schema.ground_truth(dictionary, schema.load(config.SCHEMA), manual)


def stage_evaluate():
    import json

    import pandas as pd

    from . import config, evaluate, llm, review

    d = _load_dictionary()
    mapped = pd.read_csv(config.INTERIM / "llm_mappings.csv", dtype={"module": str}, keep_default_na=False,
                         na_values={"confidence": [""]})
    truth = _load_truth(d)
    m = evaluate.metrics(mapped, truth, usage=llm.cache_usage(config.CACHE))
    config.REPORTS.mkdir(exist_ok=True)
    (config.REPORTS / "llm_metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    q = review.queue(mapped, truth, d)
    path = config.VALIDATION / "review_queue.csv"
    if path.exists():
        q = review.keep_decisions(q, pd.read_csv(path, dtype={"module": str}, keep_default_na=False))
    q.to_csv(path, index=False, encoding="utf-8")
    print(json.dumps({k: v for k, v in m.items() if k != "calibration"}, indent=1))
    print(q["reason"].value_counts().to_string())


STAGES = {
    "fetch": stage_fetch,
    "evaluate": stage_evaluate,
    "dictionary": stage_dictionary,
    "draft_ground_truth": stage_draft_ground_truth,
    "llm": stage_llm,
}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="enahoharm.pipeline")
    parser.add_argument("stage", choices=sorted(STAGES))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    STAGES[args.stage]()


if __name__ == "__main__":
    main()
