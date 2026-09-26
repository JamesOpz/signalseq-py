"""Input validation: the checks that need to read from disk.

`load_config` already covers everything derivable from the YAML alone. This
module does the rest -- files exist, barcode lengths agree with the declared
element lengths, elements fit inside the actual reads -- so a long run fails in
seconds rather than hours, with a message naming the config key at fault.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .config import Config
from .fastq_io import peek_read_lengths
from .readstructure import ConfigError
from .whitelists import load_all

READ_PEEK_DEFAULT = 1000


@dataclass
class Report:
    """Accumulated findings. Errors block the run; warnings do not."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def note(self, msg: str) -> None:
        self.notes.append(msg)

    def raise_if_failed(self) -> None:
        if self.errors:
            joined = "\n".join(f"  - {e}" for e in self.errors)
            raise ConfigError(f"Config validation failed with {len(self.errors)} error(s):\n{joined}")


def validate(cfg: Config, peek_reads: int = READ_PEEK_DEFAULT,
             check_reads: bool = True) -> Report:
    """Run every input check. Returns a Report; does not raise on findings."""
    report = Report()
    _check_files_exist(cfg, report)
    _check_output_dir(cfg, report)
    whitelists = _check_barcode_lengths(cfg, report)
    if check_reads:
        _check_read_lengths(cfg, report, peek_reads)
    _check_encoding(cfg, report, whitelists)
    _describe(cfg, report)
    return report


# --------------------------------------------------------------------------
# Individual checks
# --------------------------------------------------------------------------

def _check_files_exist(cfg: Config, report: Report) -> None:
    for sub in cfg.sublibraries:
        for read, path in (("r1", sub.r1), ("r2", sub.r2)):
            if not path.is_file():
                report.error(f"inputs.sublibraries[{sub.name}].{read}: file not found: {path}")
            elif path.stat().st_size == 0:
                report.error(f"inputs.sublibraries[{sub.name}].{read}: file is empty: {path}")

    for name, spec in cfg.whitelists.items():
        if not spec.csv.is_file():
            report.error(f"inputs.whitelists.{name}.csv: file not found: {spec.csv}")
        if spec.edit_distance_json is not None and not spec.edit_distance_json.is_file():
            report.error(
                f"inputs.whitelists.{name}.edit_distance_json: file not found: "
                f"{spec.edit_distance_json}"
            )
    if not cfg.panel.csv.is_file():
        report.error(f"inputs.adt_panel.csv: file not found: {cfg.panel.csv}")
    if not cfg.layout.csv.is_file():
        report.error(f"inputs.rt_layout.csv: file not found: {cfg.layout.csv}")


def _check_output_dir(cfg: Config, report: Report) -> None:
    out = cfg.output_dir
    parent = out.parent
    if not parent.exists():
        report.note(f"run.output_dir parent {parent} does not exist yet; it will be created")
    if out.exists() and any(out.iterdir()) and not cfg.overwrite:
        report.error(
            f"run.output_dir is not empty: {out}. Set run.overwrite: true to write into it "
            f"anyway, or choose a fresh directory."
        )


def _check_barcode_lengths(cfg: Config, report: Report):
    """Every whitelist's barcodes must be exactly as long as its element declares."""
    try:
        panel, whitelists = load_all(cfg)
    except ConfigError as exc:
        report.error(str(exc))
        return None

    adt_el = cfg.read_structure.adt
    if panel.seq_length != adt_el.length:
        report.error(
            f"inputs.adt_panel.csv holds {panel.seq_length}bp barcodes but "
            f"read_structure.elements.{adt_el.name}.length is {adt_el.length}. "
            f"Set length: {panel.seq_length}, or point at the right panel file."
        )

    for element in cfg.read_structure.rounds:
        wl = whitelists[element.name]
        if wl.seq_length != element.length:
            spec = cfg.whitelists[element.whitelist]
            report.error(
                f"inputs.whitelists.{element.whitelist} ({spec.csv.name}) holds "
                f"{wl.seq_length}bp barcodes but read_structure.elements.{element.name}.length "
                f"is {element.length}. Set length: {wl.seq_length}, or point at the right "
                f"whitelist file."
            )
    return whitelists


def _check_read_lengths(cfg: Config, report: Report, peek_reads: int) -> None:
    """Confirm the declared elements actually fit inside the reads."""
    for sub in cfg.sublibraries:
        for read in ("R1", "R2"):
            needed = cfg.read_structure.min_length(read)
            if needed == 0:
                continue
            path = sub.read_path(read)
            if not path.is_file():
                continue          # already reported by _check_files_exist
            try:
                lengths = peek_read_lengths(
                    path, peek_reads, cfg.performance["gzip_binaries"]
                )
            except (OSError, IOError) as exc:
                report.error(f"inputs.sublibraries[{sub.name}]: could not read {path}: {exc}")
                continue
            if not lengths:
                report.error(f"inputs.sublibraries[{sub.name}]: {path} yielded no reads")
                continue

            shortest, longest = min(lengths), max(lengths)
            if shortest < needed:
                culprits = [
                    e.describe() for e in cfg.read_structure.elements_on(read)
                    if e.end0 > shortest
                ]
                n_short = sum(1 for l in lengths if l < needed)
                report.error(
                    f"inputs.sublibraries[{sub.name}].{read.lower()}: the read structure needs "
                    f"{needed}bp but {n_short}/{len(lengths)} of the first reads in {path.name} "
                    f"are shorter (min {shortest}bp). Element(s) running past the end: "
                    f"{'; '.join(culprits)}"
                )
            elif longest > needed:
                report.note(
                    f"{sub.name} {read}: reads are {shortest}-{longest}bp; the structure uses "
                    f"the first {needed}bp"
                )


def _check_encoding(cfg: Config, report: Report, whitelists) -> None:
    """Confirm the chemistry fits in 64 bits, and report the packing."""
    if whitelists is None:
        return
    from .encoding import Encoder, EncodingError
    from .whitelists import load_adt_panel

    try:
        panel = load_adt_panel(
            cfg.panel, cfg.matching["adt"]["max_mismatches"], cfg.matching["adt"]["drop_ambiguous"]
        )
        encoder = Encoder.build(
            [whitelists[el.name].n for el in cfg.read_structure.rounds],
            panel.n,
            cfg.read_structure.umi.length,
        )
    except (EncodingError, ConfigError) as exc:
        report.error(str(exc))
        return
    report.note(f"encoding: {encoder.describe()}")


def _describe(cfg: Config, report: Report) -> None:
    report.note("read structure:\n" + _indent(cfg.read_structure.diagram()))
    census = cfg.matching["cell_barcode"]["census"]
    report.note(
        f"census: {census['n_reads']:,} reads from offset {census['offset_reads']:,}; "
        f"threshold = max({census['min_count_threshold']}, "
        f"count at {census['perfect_cell_threshold']:.0%} cumulative reads)"
    )
    filt = cfg.filtering
    threshold = ("automatic knee" if filt["min_umi_per_cell"] is None
                 else f"fixed floor {filt['min_umi_per_cell']:g} UMIs")
    report.note(
        f"cell calling: {threshold}, min {filt['min_adts_detected']} ADT(s) detected"
        + (f", max {filt['max_umi_per_cell']:g} UMIs" if filt["max_umi_per_cell"] is not None else "")
    )


def _indent(text: str, prefix: str = "      ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())
