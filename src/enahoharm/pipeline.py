"""Command-line entry point: python -m enahoharm.pipeline <stage>."""
import argparse


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


STAGES = {"fetch": stage_fetch, "dictionary": stage_dictionary, "draft_ground_truth": stage_draft_ground_truth}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="enahoharm.pipeline")
    parser.add_argument("stage", choices=sorted(STAGES))
    args = parser.parse_args(argv)
    STAGES[args.stage]()


if __name__ == "__main__":
    main()
