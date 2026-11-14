"""Command-line entry point: python -m enahoharm.pipeline <stage>."""
import argparse


def stage_fetch():
    from .fetch import fetch_all

    fetch_all()


def stage_dictionary():
    from .dictionary import build

    d = build()
    print(d.groupby(["year", "module"]).size().to_string())


STAGES = {"fetch": stage_fetch, "dictionary": stage_dictionary}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="enahoharm.pipeline")
    parser.add_argument("stage", choices=sorted(STAGES))
    args = parser.parse_args(argv)
    STAGES[args.stage]()


if __name__ == "__main__":
    main()
