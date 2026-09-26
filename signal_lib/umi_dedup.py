"""Vectorized UMI deduplication.

Two stages:
  1. One vectorized `np.unique` pass on a packed uint64 (cell, adt, umi) key
     collapses exact-duplicate reads -- this replaces the pilot notebook's
     per-read online dict-of-dict-of-dict UMI tracking, which does not scale
     to hundreds of millions of reads.
  2. A much cheaper 1bp-mismatch ("directional", UMI-tools-style) collapse
     runs only on the (cell,adt) groups that actually have more than one
     distinct UMI -- the large majority of groups are singletons and skip
     this entirely.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from . import encoding

UMI_BITS = encoding.UMI_BITS
UMI_LEN = encoding.UMI_LEN


@njit(cache=True)
def _hamming_umi(a, b):
    diff = 0
    x = a ^ b
    for i in range(UMI_LEN):
        if (x >> (2 * i)) & 0b11:
            diff += 1
    return diff


@njit(cache=True)
def _collapse_group_directional(umi_codes: np.ndarray, read_counts: np.ndarray) -> int:
    """UMI-tools 'directional' collapse on one (cell,adt) group's distinct UMIs.
    Process UMIs in descending read-count order; a lower-count UMI is absorbed
    into an already-formed cluster if it is within Hamming distance 1 of the
    cluster's seed AND its own count satisfies count <= 2*seed_count - 1
    (the directional-adjacency rule). Returns the number of clusters, i.e. the
    corrected molecule count for this (cell,adt) pair."""
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
            if read_counts[j] <= 2 * ci - 1 and _hamming_umi(umi_codes[i], umi_codes[j]) <= 1:
                assigned[j] = True
    return n_clusters


def collapse_umis(cell_id_arr: np.ndarray, adt_idx_arr: np.ndarray, umi_packed_arr: np.ndarray):
    """Returns (cell_id, adt_idx, count) -- one row per surviving (cell,adt)
    pair with its UMI-collapsed molecule count."""
    composite = encoding.pack_composite(cell_id_arr, adt_idx_arr, umi_packed_arr)
    distinct_keys, read_counts = np.unique(composite, return_counts=True)

    ca_key = encoding.cell_adt_key(distinct_keys)
    group_keys, group_start, group_sizes = np.unique(ca_key, return_index=True, return_counts=True)

    umi_mask = np.uint64((1 << UMI_BITS) - 1)
    umi_codes_sorted = (distinct_keys & umi_mask).astype(np.int64)
    read_counts_i64 = read_counts.astype(np.int64)

    final_counts = np.ones(len(group_keys), dtype=np.int32)
    multi = np.where(group_sizes > 1)[0]
    for g in multi:
        s, n = group_start[g], group_sizes[g]
        final_counts[g] = _collapse_group_directional(
            umi_codes_sorted[s:s + n], read_counts_i64[s:s + n]
        )

    cell_out, adt_out = encoding.unpack_cell_adt(group_keys)
    return cell_out, adt_out, final_counts
