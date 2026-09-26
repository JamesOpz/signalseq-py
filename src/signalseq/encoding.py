"""Packed-integer encodings, so the read scan and UMI dedup never build strings.

Cell identity (one whitelist index per barcode round) and the UMI are packed
into small integers up front; the hot loop and dedup steps then operate on plain
numpy arrays rather than Python strings or nested dicts.

Bit widths are DERIVED from the loaded whitelists and panel, not hardcoded. The
original pipeline fixed them at 99 RT x 96 x 96 wells and 4 ADT bits (a silent
16-antibody ceiling); computing them means a larger panel or a differently sized
plate works without an edit, and the arithmetic is bit-identical whenever the
sizes match the original.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log2
from typing import Sequence

import numpy as np

# A,C,G,T -> 0..3. Anything else (N, and any stray byte) folds to 0: the packed
# UMI is only ever used as a dedup hash, never decoded back to sequence.
_BASE_CODE = {65: 0, 67: 1, 71: 2, 84: 3}
_BASE_TABLE = np.zeros(256, dtype=np.uint64)
for _b, _v in _BASE_CODE.items():
    _BASE_TABLE[_b] = _v


def _bits_for(n_values: int) -> int:
    """Bits needed to hold indices 0..n_values-1."""
    if n_values <= 1:
        return 1
    return int(ceil(log2(n_values)))


class EncodingError(ValueError):
    """Raised when a chemistry cannot be packed into the available 64 bits."""


@dataclass(frozen=True)
class Encoder:
    """Packing rules for one chemistry.

    `round_sizes` is the whitelist size of each cell-barcode round, in the
    config's `cell_barcode.order` (most significant first), so the combined
    cell id is a plain mixed-radix number.
    """

    round_sizes: tuple[int, ...]
    n_adts: int
    umi_length: int

    # -- construction ------------------------------------------------------
    @classmethod
    def build(cls, round_sizes: Sequence[int], n_adts: int, umi_length: int) -> "Encoder":
        round_sizes = tuple(int(s) for s in round_sizes)
        if not round_sizes:
            raise EncodingError("At least one cell-barcode round is required")
        if any(s < 1 for s in round_sizes):
            raise EncodingError(f"Every barcode round needs a non-empty whitelist, got sizes {round_sizes}")
        if n_adts < 1:
            raise EncodingError("The ADT panel must contain at least one antibody")
        if umi_length < 1:
            raise EncodingError("The UMI element must be at least 1bp")

        enc = cls(round_sizes=round_sizes, n_adts=int(n_adts), umi_length=int(umi_length))
        if enc.total_bits > 64:
            raise EncodingError(
                f"This chemistry needs {enc.total_bits} bits to pack (cell, ADT, UMI) into one "
                f"integer, over the 64-bit limit: {enc.n_cells:,} cell barcodes "
                f"({enc.cell_bits} bits) x {enc.n_adts} ADTs ({enc.adt_bits} bits) x "
                f"{enc.umi_length}bp UMI ({enc.umi_bits} bits). Shorten the UMI or reduce the "
                f"whitelist sizes."
            )
        return enc

    # -- derived widths ----------------------------------------------------
    @property
    def n_cells(self) -> int:
        """Size of the cell-barcode space (product of the round whitelist sizes)."""
        total = 1
        for s in self.round_sizes:
            total *= s
        return total

    @property
    def cell_bits(self) -> int:
        return _bits_for(self.n_cells)

    @property
    def adt_bits(self) -> int:
        return _bits_for(self.n_adts)

    @property
    def umi_bits(self) -> int:
        return 2 * self.umi_length

    @property
    def total_bits(self) -> int:
        return self.cell_bits + self.adt_bits + self.umi_bits

    @property
    def radix_multipliers(self) -> tuple[int, ...]:
        """Place value of each round, most significant first.

        For Parse v3's [99, 96, 96] this is (9216, 96, 1), i.e.
        rt*96*96 + bc2*96 + bc3 -- identical to the original packing.
        """
        mults = []
        acc = 1
        for size in reversed(self.round_sizes):
            mults.append(acc)
            acc *= size
        return tuple(reversed(mults))

    # -- dtypes ------------------------------------------------------------
    @property
    def cell_dtype(self):
        return np.uint32 if self.n_cells <= np.iinfo(np.uint32).max else np.uint64

    @property
    def adt_dtype(self):
        return np.uint8 if self.n_adts <= np.iinfo(np.uint8).max else np.uint32

    @property
    def umi_dtype(self):
        return np.uint32 if self.umi_bits <= 32 else np.uint64

    # -- UMI ---------------------------------------------------------------
    def pack_umi(self, umi_bytes: bytes) -> int:
        """Pack a UMI into a 2-bits-per-base integer."""
        val = 0
        for b in umi_bytes:
            val = (val << 2) | _BASE_CODE.get(b, 0)
        return val

    # -- cell identity -----------------------------------------------------
    def pack_cell_id(self, round_idxs: Sequence[int]) -> int:
        total = 0
        for idx, mult in zip(round_idxs, self.radix_multipliers):
            total += idx * mult
        return total

    def unpack_cell_id(self, cell_id: np.ndarray) -> tuple[np.ndarray, ...]:
        """Vectorized inverse of pack_cell_id, returning one index array per round."""
        cell_id = np.asarray(cell_id, dtype=np.int64)
        out = []
        rem = cell_id
        for size, mult in zip(self.round_sizes, self.radix_multipliers):
            out.append(rem // mult)
            rem = rem % mult
        return tuple(out)

    # -- composite (cell, adt, umi) ---------------------------------------
    def pack_composite(self, cell_id: np.ndarray, adt_idx: np.ndarray,
                       umi_packed: np.ndarray) -> np.ndarray:
        """One uint64 per read: cell in the high bits, then ADT, then UMI.

        Sorting this key groups by (cell, ADT) with the UMI ascending inside
        each group, which is what makes the dedup pass a single np.unique.
        """
        cell_id = cell_id.astype(np.uint64)
        adt_idx = adt_idx.astype(np.uint64)
        umi_packed = umi_packed.astype(np.uint64)
        shift_cell = np.uint64(self.adt_bits + self.umi_bits)
        shift_adt = np.uint64(self.umi_bits)
        return (cell_id << shift_cell) | (adt_idx << shift_adt) | umi_packed

    def cell_adt_key(self, composite: np.ndarray) -> np.ndarray:
        """Drop the UMI bits, leaving a (cell, ADT) grouping key."""
        return composite >> np.uint64(self.umi_bits)

    def umi_of(self, composite: np.ndarray) -> np.ndarray:
        return (composite & np.uint64((1 << self.umi_bits) - 1)).astype(np.int64)

    def unpack_cell_adt(self, key: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Inverse of cell_adt_key: (cell_id, adt_idx)."""
        adt_mask = np.uint64((1 << self.adt_bits) - 1)
        adt_idx = (key & adt_mask).astype(np.int64)
        cell_id = (key >> np.uint64(self.adt_bits)).astype(np.int64)
        return cell_id, adt_idx

    def unpack_composite(self, key: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        umi = self.umi_of(key)
        rest = key >> np.uint64(self.umi_bits)
        cell_id, adt_idx = self.unpack_cell_adt(rest)
        return cell_id, adt_idx, umi

    def describe(self) -> str:
        rounds = " x ".join(str(s) for s in self.round_sizes)
        return (f"cell space {rounds} = {self.n_cells:,} ({self.cell_bits} bits), "
                f"{self.n_adts} ADTs ({self.adt_bits} bits), "
                f"{self.umi_length}bp UMI ({self.umi_bits} bits) "
                f"= {self.total_bits}/64 bits")
