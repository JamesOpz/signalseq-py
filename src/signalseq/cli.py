"""Command-line entry point.

    signalseq adt validate   --config c.yaml            check config + inputs, run nothing
    signalseq adt run        --config c.yaml --sublibrary NAME
                                                        one sublibrary
    signalseq adt run-multi  --config c.yaml            every sublibrary, then aggregate
    signalseq adt aggregate  --config c.yaml            aggregate existing outputs only
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import __version__
from .readstructure import ConfigError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="signalseq",
        description="YAML-driven ADT quantification for SIGNAL-seq / Parse Evercode libraries.",
    )
    parser.add_argument("--version", action="version", version=f"signalseq {__version__}")
    modalities = parser.add_subparsers(dest="modality", metavar="MODALITY")

    adt = modalities.add_parser("adt", help="antibody-derived tag (ADT) quantification")
    verbs = adt.add_subparsers(dest="verb", metavar="VERB")

    def add_config(p):
        p.add_argument("--config", required=True, type=Path, metavar="PATH",
                       help="YAML run configuration")
        return p

    v = add_config(verbs.add_parser(
        "validate", help="check the config and all inputs, then exit without running"))
    v.add_argument("--peek-reads", type=int, default=1000, metavar="N",
                   help="reads to inspect per FASTQ when checking read lengths (default: 1000)")
    v.add_argument("--skip-read-check", action="store_true",
                   help="skip reading the FASTQs (config and reference files only)")

    r = add_config(verbs.add_parser("run", help="process a single sublibrary"))
    r.add_argument("--sublibrary", metavar="NAME",
                   help="sublibrary to process; required when the config lists more than one")
    r.add_argument("--debug-reads", type=int, default=None, metavar="N",
                   help="process only the first N read pairs (smoke test)")

    m = add_config(verbs.add_parser(
        "run-multi", help="process every sublibrary in the config, then aggregate"))
    m.add_argument("--debug-reads", type=int, default=None, metavar="N",
                   help="process only the first N read pairs per sublibrary (smoke test)")
    m.add_argument("--skip-aggregate", action="store_true",
                   help="process the sublibraries but do not aggregate")
    m.add_argument("--continue-on-error", action="store_true",
                   help="carry on with the remaining sublibraries if one fails")

    a = add_config(verbs.add_parser(
        "aggregate", help="aggregate sublibrary outputs that already exist"))
    a.add_argument("--sublibrary-dir", action="append", dest="sublibrary_dirs", type=Path,
                   metavar="DIR",
                   help="output directory to aggregate; repeatable. "
                        "Defaults to every sublibrary in the config.")

    return parser


# --------------------------------------------------------------------------
# Verbs
# --------------------------------------------------------------------------

def _print_report(report, header: str) -> None:
    print(header)
    for note in report.notes:
        print(f"  {note}")
    for warning in report.warnings:
        print(f"  WARNING: {warning}")
    if report.errors:
        print(f"\n{len(report.errors)} error(s):")
        for err in report.errors:
            print(f"  ERROR: {err}")


def cmd_validate(args) -> int:
    from .config import load_config
    from .validate import validate

    cfg = load_config(args.config)
    report = validate(cfg, peek_reads=args.peek_reads, check_reads=not args.skip_read_check)

    _print_report(report, f"Validating {cfg.path}\n  run: {cfg.name}\n  output_dir: {cfg.output_dir}")
    print()
    if report.ok:
        n = len(cfg.sublibraries)
        print(f"OK -- config and inputs are valid ({n} sublibrar{'y' if n == 1 else 'ies'}).")
        return 0
    print("FAILED -- fix the errors above and re-run validate.")
    return 1


def _validate_or_die(cfg, *, check_reads: bool = True) -> None:
    """Every run validates first: a long job should not die on a typo at minute 90."""
    from .validate import validate
    report = validate(cfg, check_reads=check_reads)
    for note in report.notes:
        print(f"  {note}")
    for warning in report.warnings:
        print(f"  WARNING: {warning}")
    report.raise_if_failed()


def cmd_run(args) -> int:
    from .config import load_config
    from .pipeline import process_sublibrary

    cfg = load_config(args.config)
    name = args.sublibrary
    if name is None:
        if len(cfg.sublibraries) != 1:
            raise ConfigError(
                f"--sublibrary is required when the config lists more than one: "
                f"{[s.name for s in cfg.sublibraries]}. Or use `run-multi` to process all of them."
            )
        name = cfg.sublibraries[0].name
    cfg.sublibrary(name)   # fail early on an unknown name

    print(f"signalseq {__version__} -- adt run: {name}")
    _validate_or_die(cfg)
    out = process_sublibrary(cfg, name, debug_reads=args.debug_reads)
    print(f"\nDone. Outputs in {out}")
    return 0


def cmd_run_multi(args) -> int:
    from .aggregate import aggregate
    from .config import load_config
    from .pipeline import process_sublibrary

    cfg = load_config(args.config)
    names = [s.name for s in cfg.sublibraries]
    print(f"signalseq {__version__} -- adt run-multi: {len(names)} sublibraries: {', '.join(names)}")
    _validate_or_die(cfg)

    # Sequential by design: peak memory is dominated by the UMI-dedup sort, so
    # running sublibraries concurrently risks swapping on a modest machine.
    done, failed = [], []
    t_start = time.time()
    for i, name in enumerate(names, 1):
        print(f"\n=== [{i}/{len(names)}] {name} ===")
        try:
            process_sublibrary(cfg, name, debug_reads=args.debug_reads)
            done.append(name)
        except Exception as exc:
            if not args.continue_on_error:
                raise
            print(f"  ERROR: {name} failed: {exc}", file=sys.stderr)
            failed.append((name, str(exc)))

    if failed:
        print(f"\n{len(failed)} sublibrar{'y' if len(failed) == 1 else 'ies'} failed:")
        for name, err in failed:
            print(f"  {name}: {err}")

    if args.skip_aggregate:
        print("\nSkipping aggregation (--skip-aggregate).")
    elif not done:
        print("\nNothing to aggregate: no sublibrary completed.", file=sys.stderr)
        return 1
    else:
        print(f"\n=== Aggregating {len(done)} sublibraries ===")
        aggregate(cfg, [cfg.sublibrary_output_dir(n) for n in done])

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed/60:.1f} min. Outputs in {cfg.output_dir}")
    return 1 if failed else 0


def cmd_aggregate(args) -> int:
    from .aggregate import aggregate
    from .config import load_config

    cfg = load_config(args.config)
    dirs = args.sublibrary_dirs
    if dirs:
        dirs = [Path(d).expanduser().resolve() for d in dirs]
    print(f"signalseq {__version__} -- adt aggregate")
    out = aggregate(cfg, dirs)
    print(f"\nDone. Outputs in {out}")
    return 0


VERBS = {
    "validate": cmd_validate,
    "run": cmd_run,
    "run-multi": cmd_run_multi,
    "aggregate": cmd_aggregate,
}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.modality != "adt" or getattr(args, "verb", None) is None:
        parser.print_help()
        return 2

    try:
        return VERBS[args.verb](args)
    except ConfigError as exc:
        print(f"\nConfiguration error:\n{exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
