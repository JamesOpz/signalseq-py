"""Experimental sample identity from the RT-well plate layout
(reference/Parse_Exp_RT_Layout.csv), joined onto cells by RT_well.

The layout file's Sample_name uses two slightly different token orders
depending on whether the well is chemo-treated or a DMSO control:
  - treated:  "{culture}_{timepoint}_Treatment"     e.g. "PDO+CAF_6hr_Treatment"
  - control:  "{culture}_DMSO_{timepoint}hrs"        e.g. "PDO_DMSO_48hrs"
culture is "PDO" (single organoid, MonoCulture) or "PDO+CAF" (organoid +
cancer-associated fibroblast, CoCulture); treatment is the chemotherapy
(SN-38) or its DMSO vehicle control.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import config

DEFAULT_RT_LAYOUT_CSV = config.REFERENCE_DIR / "Parse_Exp_RT_Layout.csv"

_CULTURE_MAP = {"PDO": "MonoCulture", "PDO+CAF": "CoCulture"}
_TIMEPOINT_RE = re.compile(r"^(\d+)(?:hrs?)?$", re.IGNORECASE)


def _parse_sample_name(sample_name: str):
    tokens = sample_name.split("_")
    culture_raw = tokens[0]
    culture_type = _CULTURE_MAP.get(culture_raw, culture_raw)

    if len(tokens) == 3 and tokens[1].upper() == "DMSO":
        treatment = "DMSO"
        timepoint_tok = tokens[2]
    elif len(tokens) == 3:
        treatment = "SN-38"
        timepoint_tok = tokens[1]
    else:
        # Unexpected format -- keep raw fields, don't fail the run over a label.
        treatment = "Unknown"
        timepoint_tok = ""

    m = _TIMEPOINT_RE.match(timepoint_tok)
    timepoint_hrs = int(m.group(1)) if m else np.nan

    return culture_type, timepoint_hrs, treatment


def load_rt_layout(csv_path=DEFAULT_RT_LAYOUT_CSV) -> pd.DataFrame:
    """Returns a DataFrame indexed by well (e.g. 'A1') with columns:
    sample_name, culture_type, timepoint_hrs, treatment, condition."""
    df = pd.read_csv(csv_path)
    parsed = df["Sample_name"].apply(_parse_sample_name)
    df["culture_type"] = [p[0] for p in parsed]
    df["timepoint_hrs"] = [p[1] for p in parsed]
    df["treatment"] = [p[2] for p in parsed]
    df["condition"] = (
        df["culture_type"] + "_" + df["timepoint_hrs"].astype("Int64").astype(str) + "hr_" + df["treatment"]
    )
    return df.set_index("Well_ID")[["Sample_name", "culture_type", "timepoint_hrs", "treatment", "condition"]]


def annotate_sample_identity(adata, layout_df: pd.DataFrame | None = None):
    """Adds sample_name/culture_type/timepoint_hrs/treatment/condition to
    adata.obs by joining on RT_well. RT wells absent from the layout (i.e. RT
    barcodes not used in this experiment) are labeled 'Unassigned' -- these
    should be rare/near-zero among real called cells and are a useful QC
    signal in their own right."""
    if layout_df is None:
        layout_df = load_rt_layout()

    joined = adata.obs[["RT_well"]].join(layout_df, on="RT_well")
    for col in ("Sample_name", "culture_type", "treatment", "condition"):
        joined[col] = joined[col].fillna("Unassigned")
    joined["timepoint_hrs"] = joined["timepoint_hrs"].astype(float)

    adata.obs["sample_name"] = joined["Sample_name"].values
    adata.obs["culture_type"] = joined["culture_type"].values
    adata.obs["timepoint_hrs"] = joined["timepoint_hrs"].values
    adata.obs["treatment"] = joined["treatment"].values
    adata.obs["condition"] = joined["condition"].values
    return adata
