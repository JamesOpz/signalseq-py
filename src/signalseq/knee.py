"""Automatic knee/inflection detection on the UMI-rank curve.

CellRanger / dropletUtils style: the threshold is derived from the data rather
than carried over as a fixed number, because a pilot-scale UMI cutoff does not
transfer to full-depth data. A config may still pin a fixed floor
(`filtering.min_umi_per_cell`), in which case the knee is still computed and
reported so it stays visible on the QC plots for comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class KneeResult:
    knee_rank: int
    knee_umi: float
    sorted_counts: np.ndarray   # descending, for the knee plot
    ranks: np.ndarray


def find_knee_point(umi_counts: np.ndarray, exclude_top_n: int) -> KneeResult:
    """Max-distance-to-chord on the log-log rank curve.

    1. Sort counts descending, drop zeros, work in log10(rank)/log10(count).
    2. Skip the first `exclude_top_n` ranks -- the steep, noisy head -- when
       fitting the chord.
    3. Chord runs from the first retained point to the last.
    4. The knee is the point furthest (perpendicular) from that chord.
    """
    sorted_counts = np.sort(np.asarray(umi_counts))[::-1]
    sorted_counts = sorted_counts[sorted_counts > 0]
    n = len(sorted_counts)
    ranks = np.arange(1, n + 1)

    if n <= exclude_top_n + 2:
        # Too few points for a meaningful knee; fall back to the median rank.
        knee_rank = max(1, n // 2)
        knee_umi = float(sorted_counts[knee_rank - 1]) if n else 0.0
        return KneeResult(knee_rank, knee_umi, sorted_counts, ranks)

    log_rank = np.log10(ranks[exclude_top_n:])
    log_count = np.log10(sorted_counts[exclude_top_n:])

    x1, y1 = log_rank[0], log_count[0]
    x2, y2 = log_rank[-1], log_count[-1]
    num = np.abs((y2 - y1) * log_rank - (x2 - x1) * log_count + x2 * y1 - y2 * x1)
    den = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    distances = num / den if den > 0 else np.zeros_like(num)

    local_idx = int(np.argmax(distances))
    knee_rank = int(ranks[exclude_top_n:][local_idx])
    knee_umi = float(sorted_counts[exclude_top_n:][local_idx])
    return KneeResult(knee_rank, knee_umi, sorted_counts, ranks)


def call_cells(adata, filtering_cfg: dict):
    """Compose a UMI threshold with a minimum-ADTs-detected floor.

    The UMI threshold is `filtering.min_umi_per_cell` when set, otherwise the
    automatic knee. Returns (mask, knee_result, umi_threshold_used).
    """
    total_counts = adata.obs["total_counts"].to_numpy()
    exclude_top_n = int((filtering_cfg.get("knee") or {}).get("exclude_top_n", 0))
    knee_result = find_knee_point(total_counts, exclude_top_n)

    min_umi = filtering_cfg.get("min_umi_per_cell")
    max_umi = filtering_cfg.get("max_umi_per_cell")
    umi_threshold = knee_result.knee_umi if min_umi is None else float(min_umi)

    mask = total_counts >= umi_threshold
    if max_umi is not None:
        mask &= total_counts <= float(max_umi)
    mask &= adata.obs["n_adts_detected"].to_numpy() >= int(filtering_cfg["min_adts_detected"])

    return mask, knee_result, umi_threshold
