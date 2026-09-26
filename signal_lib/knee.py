"""Automatic knee/inflection-point detection on the UMI-rank curve (CellRanger /
dropletUtils style), replacing the pilot notebook's hardcoded pilot-scale
MIN_UMI_PER_CELL/MAX_UMI_PER_CELL thresholds -- those numbers don't transfer to
full NovaSeq-depth data, so the threshold is derived from the data itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config


@dataclass
class KneeResult:
    knee_rank: int
    knee_umi: float
    sorted_counts: np.ndarray   # descending, for the knee plot
    ranks: np.ndarray


def find_knee_point(umi_counts: np.ndarray, exclude_top_n: int = config.KNEE_EXCLUDE_TOP_N) -> KneeResult:
    """Max-distance-to-chord method on the log-log rank curve.

    1. Sort UMI counts descending, drop zeros, work in log10(rank)/log10(count) space.
    2. Skip the first `exclude_top_n` ranks (steep/noisy head) when fitting the chord.
    3. Chord = line from the point just after the excluded head to the last point.
    4. knee = the point with the largest perpendicular distance to that chord.
    """
    sorted_counts = np.sort(np.asarray(umi_counts))[::-1]
    sorted_counts = sorted_counts[sorted_counts > 0]
    n = len(sorted_counts)
    ranks = np.arange(1, n + 1)

    if n <= exclude_top_n + 2:
        # Too few points for a meaningful knee; fall back to the median.
        knee_rank = max(1, n // 2)
        knee_umi = float(sorted_counts[knee_rank - 1]) if n else 0.0
        return KneeResult(knee_rank, knee_umi, sorted_counts, ranks)

    log_rank = np.log10(ranks[exclude_top_n:])
    log_count = np.log10(sorted_counts[exclude_top_n:])

    x1, y1 = log_rank[0], log_count[0]
    x2, y2 = log_rank[-1], log_count[-1]
    # Perpendicular distance from each point to the line through (x1,y1)-(x2,y2).
    num = np.abs((y2 - y1) * log_rank - (x2 - x1) * log_count + x2 * y1 - y2 * x1)
    den = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    distances = num / den if den > 0 else np.zeros_like(num)

    local_idx = int(np.argmax(distances))
    knee_rank = int(ranks[exclude_top_n:][local_idx])
    knee_umi = float(sorted_counts[exclude_top_n:][local_idx])

    return KneeResult(knee_rank, knee_umi, sorted_counts, ranks)


def call_cells(adata, min_adts_floor: int = config.DEFAULT_MIN_ADTS_FLOOR,
               max_umi_per_cell: float | None = None,
               min_umi_override: float | None = None):
    """Compose a UMI threshold with a min-ADTs-detected floor.

    By default the UMI threshold is the automatic knee point. Pass
    `min_umi_override` to use a fixed UMI floor instead (e.g. for an
    intentionally permissive first-pass cut where downstream filtering will
    happen jointly with the RNA modality) -- the knee is still computed and
    returned so it remains visible on the QC plot for comparison.

    Returns (mask: np.ndarray[bool], knee_result: KneeResult, umi_threshold_used: float).
    """
    total_counts = adata.obs["total_counts"].to_numpy()
    knee_result = find_knee_point(total_counts)

    umi_threshold = knee_result.knee_umi if min_umi_override is None else min_umi_override

    mask = total_counts >= umi_threshold
    if max_umi_per_cell is not None:
        mask &= total_counts <= max_umi_per_cell
    mask &= adata.obs["n_adts_detected"].to_numpy() >= min_adts_floor

    return mask, knee_result, umi_threshold
