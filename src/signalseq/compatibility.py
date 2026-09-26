"""Known defects carried over from the original pipeline, and their opt-in fixes.

The port reproduces the original parsev3_pipeline bug-for-bug by default, so its
output can be proven identical. Each defect found during the port gets a flag
under `compatibility:` in the config, defaulting to the original (buggy)
behaviour. Turning a flag on changes the numbers -- deliberately -- which is why
they are explicit rather than silently corrected.

Flip them only after an equivalence run has established a baseline.
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
            "valid_cell_barcode is computed in a separate pass over the barcode read and "
            "counts every read whose barcode rounds resolve, independent of the ADT tag. "
            "This costs one extra pass over the FASTQ."
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
        return ("compatibility: reproducing the original pipeline exactly "
                f"(all {len(FIXES)} known defects preserved)")
    lines = [f"compatibility: {len(active)} of {len(FIXES)} known defects FIXED "
             f"-- output will differ from the original:"]
    for f in active:
        lines.append(f"  - {f.flag}: {f.summary}")
        lines.append(f"      affects {f.affects}")
    return "\n".join(lines)
