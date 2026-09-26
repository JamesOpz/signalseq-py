"""Experimental sample identity from the RT-well plate layout.

The layout CSV maps a well ID to a sample name; this parses that name into
culture type, timepoint and treatment, and joins the result onto cells by their
RT well. Purely annotation -- it never affects counts.

The Parse v3 experiment's Sample_name column uses two token orders depending on
whether the well is treated or a vehicle control:

    treated:  "{culture}_{timepoint}_Treatment"   e.g. "PDO+CAF_6hr_Treatment"
    control:  "{culture}_DMSO_{timepoint}hrs"     e.g. "PDO_DMSO_48hrs"

Which token marks a control, what the treated arm is called, and how culture
tokens map to display names are all config-driven (`sample_annotation`), so a
different experiment's labelling needs no code change.
"""

from __future__ import annotations

import re
from typing import Mapping

import numpy as np
import pandas as pd

from .config import LayoutSpec
from .readstructure import ConfigError

_TIMEPOINT_RE = re.compile(r"^(\d+)(?:hrs?)?$", re.IGNORECASE)

ANNOTATION_COLUMNS = ("sample_name", "culture_type", "timepoint_hrs", "treatment", "condition")


def _parse_sample_name(sample_name: str, ann_cfg: Mapping) -> tuple[str, float, str]:
    """Split one sample name into (culture_type, timepoint_hrs, treatment)."""
    culture_map = ann_cfg.get("culture_map") or {}
    treatment_cfg = ann_cfg.get("treatment") or {}
    control_token = str(treatment_cfg.get("control_token", "DMSO")).upper()
    treated_label = str(treatment_cfg.get("treated_label", "Treated"))

    tokens = str(sample_name).split("_")
    culture_raw = tokens[0]
    culture_type = culture_map.get(culture_raw, culture_raw)

    if len(tokens) == 3 and tokens[1].upper() == control_token:
        treatment = treatment_cfg.get("control_label", control_token)
        timepoint_tok = tokens[2]
    elif len(tokens) == 3:
        treatment = treated_label
        timepoint_tok = tokens[1]
    else:
        # Unexpected format -- keep what we can rather than failing a run over a label.
        treatment = "Unknown"
        timepoint_tok = ""

    m = _TIMEPOINT_RE.match(timepoint_tok)
    timepoint_hrs = int(m.group(1)) if m else np.nan
    return culture_type, timepoint_hrs, treatment


def load_rt_layout(spec: LayoutSpec, ann_cfg: Mapping) -> pd.DataFrame:
    """Read the layout CSV, indexed by well, with the parsed annotation columns."""
    if not spec.csv.is_file():
        raise ConfigError(f"inputs.rt_layout.csv: file not found: {spec.csv}")
    df = pd.read_csv(spec.csv)
    for col, key in ((spec.well_column, "well_column"), (spec.sample_column, "sample_column")):
        if col not in df.columns:
            raise ConfigError(
                f"inputs.rt_layout.{key}: column {col!r} not in {spec.csv.name}. "
                f"Columns present: {list(df.columns)}"
            )

    parsed = df[spec.sample_column].apply(lambda s: _parse_sample_name(s, ann_cfg))
    df["culture_type"] = [p[0] for p in parsed]
    df["timepoint_hrs"] = [p[1] for p in parsed]
    df["treatment"] = [p[2] for p in parsed]

    template = str(ann_cfg.get("condition_template", "{culture_type}_{timepoint_hrs}hr_{treatment}"))
    fix_missing = bool(ann_cfg.get("_fix_missing_timepoint_label", False))
    unassigned = str(ann_cfg.get("unassigned_label", "Unassigned"))

    # pandas renders a missing Int64 as the literal "<NA>", which the original
    # pipeline then embedded in the condition string. Ported as-is; the fix
    # substitutes the unassigned label instead. See compatibility.py.
    timepoint_str = df["timepoint_hrs"].astype("Int64").astype(str)
    if fix_missing:
        timepoint_str = timepoint_str.replace("<NA>", unassigned)

    df["condition"] = [
        template.format(culture_type=c, timepoint_hrs=t, treatment=tr)
        for c, t, tr in zip(df["culture_type"], timepoint_str, df["treatment"])
    ]

    out = df.set_index(spec.well_column)[
        [spec.sample_column, "culture_type", "timepoint_hrs", "treatment", "condition"]
    ]
    return out.rename(columns={spec.sample_column: "_sample_name"})


def annotate_sample_identity(adata, layout_df: pd.DataFrame, join_column: str,
                             ann_cfg: Mapping):
    """Add the annotation columns to adata.obs by joining on `join_column`.

    Wells absent from the layout (RT barcodes not used in this experiment) are
    labelled with `unassigned_label`. They should be near-zero among real called
    cells, and are a useful QC signal in their own right.
    """
    if join_column not in adata.obs.columns:
        raise ConfigError(
            f"sample_annotation: cannot join the layout on {join_column!r} -- "
            f"no such column in obs. Available: {list(adata.obs.columns)}"
        )
    unassigned = str(ann_cfg.get("unassigned_label", "Unassigned"))

    joined = adata.obs[[join_column]].join(layout_df, on=join_column)
    for col in ("_sample_name", "culture_type", "treatment", "condition"):
        joined[col] = joined[col].fillna(unassigned)
    joined["timepoint_hrs"] = joined["timepoint_hrs"].astype(float)

    adata.obs["sample_name"] = joined["_sample_name"].values
    adata.obs["culture_type"] = joined["culture_type"].values
    adata.obs["timepoint_hrs"] = joined["timepoint_hrs"].values
    adata.obs["treatment"] = joined["treatment"].values
    adata.obs["condition"] = joined["condition"].values
    return adata
