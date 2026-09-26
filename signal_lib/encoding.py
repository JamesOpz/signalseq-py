"""Packed-integer encodings used to avoid per-read string building in the hot loop.

Cell identity (rt_idx, r2_idx, r3_idx) and the 10bp UMI are packed into small
integers up front, so the hot loop and UMI-dedup steps operate on plain numpy
arrays instead of Python strings / nested dicts.
"""

import numpy as np

from . import config

N_RT = config.N_RT
N_LIGATION = config.N_LIGATION

UMI_LEN = config.LAYOUT.UMI_END - config.LAYOUT.UMI_START  # 10
UMI_BITS = 2 * UMI_LEN                                      # 20 bits
ADT_BITS = 4                                                # up to 16 ADTs
CELL_BITS = 20                                               # 99*96*96 = 912,384 < 2**20

_BASE_CODE = {65: 0, 67: 1, 71: 2, 84: 3, 78: 0}  # A,C,G,T -> 0..3 ; N -> 0 (dedup hash only)


def pack_umi10(umi_bytes: bytes) -> int:
    """Pack a 10bp UMI into a 20-bit integer (2 bits/base)."""
    val = 0
    for b in umi_bytes:
        val = (val << 2) | _BASE_CODE.get(b, 0)
    return val


def pack_cell_id(rt_idx: int, r2_idx: int, r3_idx: int) -> int:
    return rt_idx * (N_LIGATION * N_LIGATION) + r2_idx * N_LIGATION + r3_idx


def unpack_cell_id(cell_id: np.ndarray):
    """Vectorized inverse of pack_cell_id. Returns (rt_idx, r2_idx, r3_idx) arrays."""
    cell_id = np.asarray(cell_id)
    rt_idx = cell_id // (N_LIGATION * N_LIGATION)
    rem = cell_id % (N_LIGATION * N_LIGATION)
    r2_idx = rem // N_LIGATION
    r3_idx = rem % N_LIGATION
    return rt_idx, r2_idx, r3_idx


def pack_composite(cell_id: np.ndarray, adt_idx: np.ndarray, umi_packed: np.ndarray) -> np.ndarray:
    """Pack (cell_id, adt_idx, umi) into one uint64 key: cell_id in the top bits,
    then adt_idx, then umi in the bottom bits. Sorting this key groups by
    (cell_id, adt_idx) with umi ascending within each group."""
    cell_id = cell_id.astype(np.uint64)
    adt_idx = adt_idx.astype(np.uint64)
    umi_packed = umi_packed.astype(np.uint64)
    return (cell_id << np.uint64(ADT_BITS + UMI_BITS)) | (adt_idx << np.uint64(UMI_BITS)) | umi_packed


def cell_adt_key(composite_or_cell_adt: np.ndarray, has_umi: bool = True) -> np.ndarray:
    """Drop the bottom UMI_BITS to get a (cell_id, adt_idx) grouping key."""
    if has_umi:
        return composite_or_cell_adt >> np.uint64(UMI_BITS)
    return composite_or_cell_adt


def unpack_cell_adt(cell_adt_key_arr: np.ndarray):
    """Inverse of cell_adt_key: returns (cell_id, adt_idx) arrays (both uint64/int)."""
    adt_mask = np.uint64((1 << ADT_BITS) - 1)
    adt_idx = (cell_adt_key_arr & adt_mask).astype(np.int64)
    cell_id = (cell_adt_key_arr >> np.uint64(ADT_BITS)).astype(np.int64)
    return cell_id, adt_idx


def unpack_composite(key: np.ndarray):
    """Inverse of pack_composite: returns (cell_id, adt_idx, umi) arrays."""
    umi_mask = np.uint64((1 << UMI_BITS) - 1)
    adt_mask = np.uint64((1 << ADT_BITS) - 1)
    umi = (key & umi_mask).astype(np.int64)
    rest = key >> np.uint64(UMI_BITS)
    adt_idx = (rest & adt_mask).astype(np.int64)
    cell_id = (rest >> np.uint64(ADT_BITS)).astype(np.int64)
    return cell_id, adt_idx, umi
