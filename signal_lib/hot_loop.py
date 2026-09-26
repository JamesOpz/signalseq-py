"""Per-read scan: FASTQ pair -> flat numpy arrays of (cell_id, adt_idx, umi_packed)
for every read that passes ADT-tag matching and cell-barcode correction.

No per-read string concatenation, no nested dict UMI tracking -- cell identity
is a packed integer (encoding.pack_cell_id) and hits accumulate into
preallocated numpy buffers, flushed into a chunk list and concatenated once at
the end. This is the direct, scale-safe replacement for the pilot notebook's
per-read `cell_bc = f"{rt}_{r2}_{r3}"` dict-of-dict-of-dict approach.
"""

from __future__ import annotations

import numpy as np

from . import bc_correct, config, encoding
from .adt_match import match_adt
from .fastq_io import paired_fastq_records

LAYOUT = config.LAYOUT


def _new_chunk(rows: int):
    return (
        np.empty(rows, dtype=np.uint32),  # cell_id
        np.empty(rows, dtype=np.uint8),   # adt_idx
        np.empty(rows, dtype=np.uint32),  # umi_packed
    )


def scan_sublibrary(r1_path, r2_path, adt_whitelist, rt_whitelist, ligation_whitelist,
                     combo_counts, count_thresh,
                     chunk_rows: int = config.CHUNK_ROWS,
                     debug_reads: int | None = None):
    """Stream a sublibrary's FASTQ pair and return:

      cell_id_arr, adt_idx_arr, umi_packed_arr : flat numpy arrays, one row/valid read
      stats        : dict of funnel counters
      reads_per_rt : int64[n_rt] reads landing on each RT barcode (post-correction)
    """
    stats = {
        "total_reads": 0,
        "r2_too_short": 0,
        "valid_adt": 0,
        "valid_cell_barcode": 0,   # all 3 rounds resolved (exact or corrected)
        "valid_all": 0,            # ADT tag AND cell barcode both resolved
        "bc_exact_triple": 0,      # of valid_all, the subset with 0-edit-distance triple
    }
    reads_per_rt = np.zeros(rt_whitelist.n, dtype=np.int64)

    chunks = []
    cur_cell, cur_adt, cur_umi = _new_chunk(chunk_rows)
    ptr = 0

    for r1_seq, r2_seq in paired_fastq_records(r1_path, r2_path, max_reads=debug_reads):
        stats["total_reads"] += 1

        if len(r2_seq) < LAYOUT.min_r2_len:
            stats["r2_too_short"] += 1
            continue

        adt_idx = match_adt(r1_seq[LAYOUT.ADT_START:LAYOUT.ADT_END], adt_whitelist)
        if adt_idx is None:
            continue
        stats["valid_adt"] += 1

        rt_bc, r2_bc, r3_bc = bc_correct.extract_raw_barcodes(r2_seq)
        triplet, is_exact = bc_correct.correct_triplet(
            rt_bc, r2_bc, r3_bc, rt_whitelist, ligation_whitelist, combo_counts, count_thresh
        )
        if triplet is None:
            continue
        rt_idx, r2_idx, r3_idx = triplet
        if is_exact:
            stats["bc_exact_triple"] += 1

        stats["valid_cell_barcode"] += 1
        stats["valid_all"] += 1
        reads_per_rt[rt_idx] += 1

        if ptr == chunk_rows:
            chunks.append((cur_cell, cur_adt, cur_umi))
            cur_cell, cur_adt, cur_umi = _new_chunk(chunk_rows)
            ptr = 0

        cur_cell[ptr] = encoding.pack_cell_id(rt_idx, r2_idx, r3_idx)
        cur_adt[ptr] = adt_idx
        cur_umi[ptr] = encoding.pack_umi10(r2_seq[LAYOUT.UMI_START:LAYOUT.UMI_END])
        ptr += 1

    chunks.append((cur_cell[:ptr], cur_adt[:ptr], cur_umi[:ptr]))

    cell_id_arr = np.concatenate([c[0] for c in chunks]) if len(chunks) > 1 else chunks[0][0]
    adt_idx_arr = np.concatenate([c[1] for c in chunks]) if len(chunks) > 1 else chunks[0][1]
    umi_packed_arr = np.concatenate([c[2] for c in chunks]) if len(chunks) > 1 else chunks[0][2]

    return cell_id_arr, adt_idx_arr, umi_packed_arr, stats, reads_per_rt
