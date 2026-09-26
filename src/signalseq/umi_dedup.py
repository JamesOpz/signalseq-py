"""Vectorized UMI deduplication.

Two stages:

1. One `np.unique` pass on the packed (cell, ADT, UMI) key collapses exact
   duplicate reads. This replaces the pilot notebook's per-read online
   dict-of-dict-of-dict tracking, which does not scale to hundreds of millions
   of reads.
2. A mismatch-tolerant ("directional", UMI-tools style) collapse runs only on
   the (cell, ADT) groups that actually hold more than one distinct UMI. Most
   groups are singletons and skip it entirely.

The UMI length, mismatch tolerance and directional count ratio are config-driven
and passed into the numba kernels as arguments, so they stay compiled once
rather than specialising per chemistry.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from .encoding import Encoder

METHOD_DIRECTIONAL = "directional"
METHOD_EXACT = "exact"


@njit(cache=True)
def _hamming_packed(a, b, n_bases):
    """Hamming distance between two 2-bits-per-base packed UMIs."""
    diff = 0
    x = a ^ b
    for i in range(n_bases):
        if (x >> (2 * i)) & 0b11:
            diff += 1
    return diff


@njit(cache=True)
def _collapse_group_directional(umi_codes, read_counts, n_bases, max_mismatches, count_ratio):
    """UMI-tools 'directional' collapse on one (cell, ADT) group's distinct UMIs.

    Walk UMIs in descending read-count order. A lower-count UMI is absorbed into
    an already-formed cluster when it is within `max_mismatches` of that
    cluster's seed AND its own count satisfies
    `count <= count_ratio * seed_count - 1` (the directional-adjacency rule).
    Returns the cluster count, i.e. the corrected molecule count for this pair.
    """
    n = len(umi_codes)
    order = np.argsort(-read_counts)
    assigned = np.zeros(n, dtype=np.bool_)
    n_clusters = 0
    for oi in range(n):
        i = order[oi]
        if assigned[i]:
            continue
        n_clusters += 1
        assigned[i] = True
        ci = read_counts[i]
        for oj in range(oi + 1, n):
            j = order[oj]
            if assigned[j]:
                continue
            if (read_counts[j] <= count_ratio * ci - 1
                    and _hamming_packed(umi_codes[i], umi_codes[j], n_bases) <= max_mismatches):
                assigned[j] = True
    return n_clusters


@njit(cache=True)
def _collapse_all_groups(umi_codes, read_counts, group_start, group_sizes,
                         n_bases, max_mismatches, count_ratio):
    """Run the directional collapse over every multi-UMI group, in nopython mode."""
    n_groups = len(group_start)
    out = np.ones(n_groups, dtype=np.int32)
    for g in range(n_groups):
        size = group_sizes[g]
        if size <= 1:
            continue
        s = group_start[g]
        out[g] = _collapse_group_directional(
            umi_codes[s:s + size], read_counts[s:s + size],
            n_bases, max_mismatches, count_ratio,
        )
    return out


def collapse_umis(cell_id_arr: np.ndarray, adt_idx_arr: np.ndarray, umi_packed_arr: np.ndarray,
                  encoder: Encoder, umi_cfg: dict):
    """Collapse reads to molecules.

    Returns (cell_id, adt_idx, count): one row per surviving (cell, ADT) pair
    with its UMI-collapsed molecule count.
    """
    composite = encoder.pack_composite(cell_id_arr, adt_idx_arr, umi_packed_arr)
    distinct_keys, read_counts = np.unique(composite, return_counts=True)

    ca_key = encoder.cell_adt_key(distinct_keys)
    group_keys, group_start, group_sizes = np.unique(
        ca_key, return_index=True, return_counts=True
    )

    method = str(umi_cfg.get("method", METHOD_DIRECTIONAL))
    max_mismatches = int(umi_cfg.get("max_mismatches", 1))

    if method == METHOD_EXACT or max_mismatches == 0:
        # Every distinct UMI is its own molecule.
        final_counts = group_sizes.astype(np.int32)
    else:
        umi_codes = encoder.umi_of(distinct_keys)
        final_counts = _collapse_all_groups(
            umi_codes,
            read_counts.astype(np.int64),
            group_start.astype(np.int64),
            group_sizes.astype(np.int64),
            encoder.umi_length,
            max_mismatches,
            int(umi_cfg.get("directional_count_ratio", 2)),
        )

    cell_out, adt_out = encoder.unpack_cell_adt(group_keys)
    return cell_out, adt_out, final_counts
