"""Cell-barcode (bc1/RT, bc2, bc3) correction matching the Parse Biosciences RNA
pipeline's algorithm (splitpipe.bcutils.correct_barcodes / get_min_edit_dists),
so ADT and RNA sublibraries resolve the same physical R2 read to the same cell.

Two-pass design, mirroring splitpipe/preprocess.py:
  1. run_census(): sample a block of reads, keep only those where all three
     barcode rounds are EXACT whitelist matches, count (bc1,bc2,bc3) combo
     frequencies, and derive a count threshold (the 92nd-percentile-by-
     cumulative-read-fraction combo count, floored at 5).
  2. correct_triplet(): per read, fast-path exact triple match; otherwise
     gather each round's candidate whitelist indices from the precomputed
     edit-distance dict (bc_dict_n99_v5.json / bc_dict_R3_v3.json, levels
     0..BC_EDIT_DIST_MAX) and accept only if exactly one candidate
     combination clears the census count threshold.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from . import config
from .fastq_io import _FastqStream

LAYOUT = config.LAYOUT


def extract_raw_barcodes(r2_seq: bytes):
    if len(r2_seq) < LAYOUT.min_r2_len:
        return None
    r3 = r2_seq[LAYOUT.R3_START:LAYOUT.R3_END]
    r2 = r2_seq[LAYOUT.R2_START:LAYOUT.R2_END]
    rt = r2_seq[LAYOUT.RT_START:LAYOUT.RT_END]
    return rt, r2, r3


def run_census(r2_path, rt_whitelist, ligation_whitelist,
                offset: int = config.CENSUS_SAMPLE_OFFSET,
                n_samp: int = config.CENSUS_SAMPLE_SIZE):
    """Sample n_samp reads starting at `offset` from r2_path; count
    (rt_idx, r2_idx, r3_idx) combos among reads where all three rounds are
    EXACT (0-edit) whitelist matches. Returns (combo_counts: Counter, count_thresh: int).
    """
    rt_exact = {seq: i for i, seq in enumerate(rt_whitelist.seqs)}
    lig_exact = {seq: i for i, seq in enumerate(ligation_whitelist.seqs)}

    combo_counts = Counter()
    stream = _FastqStream(r2_path)
    try:
        for _ in range(offset):
            for _ in range(4):
                stream.readline()

        n_seen = 0
        while n_seen < n_samp:
            stream.readline()  # header
            seq = stream.readline().rstrip(b"\n")
            stream.readline()  # +
            stream.readline()  # qual
            if not seq:
                break
            n_seen += 1

            raw = extract_raw_barcodes(seq)
            if raw is None:
                continue
            rt_bc, r2_bc, r3_bc = raw
            rt_idx = rt_exact.get(rt_bc)
            if rt_idx is None:
                continue
            r2_idx = lig_exact.get(r2_bc)
            if r2_idx is None:
                continue
            r3_idx = lig_exact.get(r3_bc)
            if r3_idx is None:
                continue
            combo_counts[(rt_idx, r2_idx, r3_idx)] += 1
    finally:
        stream.close()

    if not combo_counts:
        return combo_counts, config.BC_CORRECT_MIN_COUNT_THRESH

    counts_sorted = np.array(sorted(combo_counts.values(), reverse=True))
    cum_frac = np.cumsum(counts_sorted) / counts_sorted.sum()
    idx = np.argmin(np.abs(cum_frac - config.BC_CORRECT_PERFECT_CELL_THRESH))
    calc = int(counts_sorted[idx])
    count_thresh = max(config.BC_CORRECT_MIN_COUNT_THRESH, calc)
    return combo_counts, count_thresh


def get_min_edit_matches(bc: bytes, edit_dicts: dict, max_d: int):
    """Mirror of splitpipe's get_min_edit_dists: return the candidate index
    list at the smallest edit distance where a match exists (checking 0,1,2,...
    in order), plus that edit distance. Empty list / None distance if nothing
    found within max_d."""
    for d in range(0, max_d + 1):
        matches = edit_dicts.get(d, {}).get(bc)
        if matches:
            return matches, d
    return [], None


def correct_triplet(rt_bc: bytes, r2_bc: bytes, r3_bc: bytes,
                     rt_whitelist, ligation_whitelist,
                     combo_counts: dict, count_thresh: int,
                     max_d: int = config.BC_EDIT_DIST_MAX):
    """Return ((rt_idx, r2_idx, r3_idx) | None, is_exact_triple: bool).

    Fast path: all three exact (edit distance 0) -> direct accept.
    Otherwise: gather per-round candidates from the precomputed edit-distance
    dicts, and accept only if exactly one (rt,r2,r3) combination among the
    candidate cartesian product clears count_thresh in the census combo_counts.
    """
    rt_matches, ed_rt = get_min_edit_matches(rt_bc, rt_whitelist.edit_dicts, max_d)
    if not rt_matches:
        return None, False
    r2_matches, ed_r2 = get_min_edit_matches(r2_bc, ligation_whitelist.edit_dicts, max_d)
    if not r2_matches:
        return None, False
    r3_matches, ed_r3 = get_min_edit_matches(r3_bc, ligation_whitelist.edit_dicts, max_d)
    if not r3_matches:
        return None, False

    if ed_rt == 0 and ed_r2 == 0 and ed_r3 == 0:
        return (rt_matches[0], r2_matches[0], r3_matches[0]), True

    accepted = None
    n_accepted = 0
    for rt_i in rt_matches:
        for r2_i in r2_matches:
            for r3_i in r3_matches:
                if combo_counts.get((rt_i, r2_i, r3_i), 0) > count_thresh:
                    accepted = (rt_i, r2_i, r3_i)
                    n_accepted += 1
                    if n_accepted > 1:
                        return None, False
    return accepted, False
