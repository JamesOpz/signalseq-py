"""Known defects carried over from the original pipeline, and their fixes.

Each defect found while porting has a flag under `compatibility:`. The fixes are
**on by default**: once equivalence with the original has been established, and
it has, shipping known-wrong numbers as the default is worse than a documented
difference from the original.

To reproduce the original exactly -- for an equivalence run, or to match results
produced before the fixes landed -- set `compatibility.reproduce_original: true`,
which turns every fix off at once. Individual flags can also be set directly.

Each entry below records what the original did, what the fix does, and which
outputs move, so a changed number can always be traced to a decision.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Fix:
    flag: str
    summary: str
    original_behaviour: str
    fixed_behaviour: str
    affects: str


FIXES: tuple[Fix, ...] = (
    Fix(
        flag="fix_valid_cell_barcode_counter",
        summary="`valid_cell_barcode` double-counts `valid_all`",
        original_behaviour=(
            "hot_loop increments valid_cell_barcode and valid_all at the same point, "
            "after an unmatched ADT tag has already skipped the read. The two counters "
            "are therefore always equal, and valid_cell_barcode does not report how many "
            "reads had a resolvable cell barcode."
        ),
        fixed_behaviour=(
            "barcode correction runs before the ADT-tag gate, so valid_cell_barcode counts "
            "every read whose barcode rounds resolve, independent of the ADT tag. Correction "
            "is the expensive step, but the ADT gate rejects only a few percent of reads, so "
            "running it on all reads rather than on the survivors costs a few percent of "
            "scan time and needs no extra pass over the FASTQ."
        ),
        affects="processing_stats.valid_cell_barcode in *_qc_stats.json and adata.uns",
    ),
    Fix(
        flag="fix_missing_timepoint_label",
        summary="an unparseable timepoint renders as the literal '<NA>' in `condition`",
        original_behaviour=(
            "sample_layout builds condition by concatenating a pandas Int64 timepoint cast "
            "to str, so a missing value becomes the four characters '<NA>', producing "
            "condition strings like 'MonoCulture_<NA>hr_SN-38'. Latent with the Parse v3 "
            "layout, where every well parses cleanly."
        ),
        fixed_behaviour="a missing timepoint uses `sample_annotation.unassigned_label` instead.",
        affects="obs.condition, and every plot or table grouped by condition",
    ),
)

FIXES_BY_FLAG = {f.flag: f for f in FIXES}


def active_fixes(compatibility_cfg: dict) -> tuple[Fix, ...]:
    return tuple(f for f in FIXES if compatibility_cfg.get(f.flag))


def describe(compatibility_cfg: dict) -> str:
    active = active_fixes(compatibility_cfg)
    if not active:
        return ("compatibility: reproduce_original -- all "
                f"{len(FIXES)} known defects preserved, output matches the original pipeline")
    if len(active) == len(FIXES):
        head = (f"compatibility: all {len(FIXES)} known defects fixed "
                f"-- these outputs differ from the original pipeline:")
    else:
        head = (f"compatibility: {len(active)} of {len(FIXES)} known defects fixed "
                f"-- these outputs differ from the original pipeline:")
    lines = [head]
    for f in active:
        lines.append(f"  - {f.flag}: {f.summary}")
        lines.append(f"      affects {f.affects}")
    return "\n".join(lines)
