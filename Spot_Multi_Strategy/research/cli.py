"""Command-line management for the unified research ledger."""

import argparse
import json

from .data_store import utc_us
from .registry import ResearchRegistry


def parser():
    root = argparse.ArgumentParser(description="Quant research registry")
    commands = root.add_subparsers(dest="command", required=True)
    initialize = commands.add_parser("init-db")
    initialize.add_argument("--db", required=True)
    preregister = commands.add_parser("preregister")
    preregister.add_argument("--db", required=True)
    preregister.add_argument("--experiment", required=True)
    preregister.add_argument("--registered", required=True)
    preregister.add_argument("--oos-start", required=True)
    preregister.add_argument("--spec", required=True)
    show = commands.add_parser("show")
    show.add_argument("--db", required=True)
    show.add_argument("--experiment", required=True)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    registry = ResearchRegistry(args.db)
    try:
        if args.command == "init-db":
            print(json.dumps({"database": args.db, "initialized": True}))
        elif args.command == "preregister":
            with open(args.spec, encoding="utf-8") as source:
                specification = json.load(source)
            registry.preregister(
                args.experiment, utc_us(args.registered),
                utc_us(args.oos_start), specification,
            )
            print(json.dumps({"experiment": args.experiment, "registered": True}))
        else:
            experiment = registry.experiment(args.experiment)
            print(json.dumps(experiment, indent=2, sort_keys=True))
    finally:
        registry.close()


if __name__ == "__main__":
    main()
