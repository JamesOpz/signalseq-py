"""Read-structure offsets and default parameters for the SIGNAL-seq Parse v3 ADT pipeline."""

from dataclasses import dataclass
from pathlib import Path

REFERENCE_DIR = Path(__file__).resolve().parent.parent / "reference"

ADT_WHITELIST_CSV = REFERENCE_DIR / "adt_barcodes.csv"
RT_WHITELIST_CSV = REFERENCE_DIR / "bc_data_rt.csv"
LIGATION_WHITELIST_CSV = REFERENCE_DIR / "bc_data_ligation.csv"
RT_BC_DICT_JSON = REFERENCE_DIR / "bc_dict_n99_v5.json"
LIGATION_BC_DICT_JSON = REFERENCE_DIR / "bc_dict_R3_v3.json"


@dataclass(frozen=True)
class ReadLayout:
    """0-indexed byte offsets into R1 (ADT read) and R2 (cell barcode + UMI read)."""

    ADT_START: int = 0
    ADT_END: int = 15

    UMI_START: int = 0
    UMI_END: int = 10
    R3_START: int = 10   # ligation round 3 (bc3)
    R3_END: int = 18
    R2_START: int = 30   # ligation round 2 (bc2)
    R2_END: int = 38
    RT_START: int = 50   # RT round (bc1)
    RT_END: int = 58

    @property
    def min_r2_len(self) -> int:
        return self.RT_END


LAYOUT = ReadLayout()

# Whitelist sizes (fixed by chemistry) -- used for packing cell identity into a
# single integer: cell_id = rt_idx * (N_LIGATION * N_LIGATION) + r2_idx * N_LIGATION + r3_idx
N_RT = 99
N_LIGATION = 96

# ---- Cell-barcode correction (must match the Parse RNA pipeline, see bc_correct.py) ----
BC_EDIT_DIST_DEFAULT = 2       # splitpipe hard-caps this at 2 regardless of config
BC_EDIT_DIST_MAX = 2
CENSUS_SAMPLE_OFFSET = 3_000_000   # splitpipe's default fastq_samp_slice = [3_000_000, 4_000_000]
CENSUS_SAMPLE_SIZE = 1_000_000
BC_CORRECT_PERFECT_CELL_THRESH = 0.92   # splitpipe's BC_CORRECT_PREFECT_CELL_THRESH
BC_CORRECT_MIN_COUNT_THRESH = 5         # splitpipe's max(5, calc) floor

# ---- ADT tag correction (custom to this CITE-seq-style protocol, no RNA-side equivalent) ----
ADT_BC_N_MUTS = 1

# ---- UMI dedup ----
UMI_N_MUTS = 1        # 1bp mismatch tolerance, applied post-hoc per (cell,adt) group

# ---- Streaming / memory ----
CHUNK_ROWS = 25_000_000
LIBDEFLATE_GZIP_BIN = "libdeflate-gzip"

# ---- Cell calling ----
DEFAULT_MIN_ADTS_FLOOR = 2
KNEE_EXCLUDE_TOP_N = 3

# ---- Saturation curve ----
SATURATION_FRACTIONS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
SATURATION_SEED = 0

# ---- Analysis ----
N_PCS = 10
PHATE_KWARGS = dict(n_components=2, knn=15, decay=40, t="auto", n_jobs=-1, random_state=42)
