"""Packed-integer encoding: derived widths and exact agreement with the original."""

from __future__ import annotations

import numpy as np
import pytest

from signalseq.encoding import Encoder, EncodingError

# The original pipeline's fixed constants, for the Parse v3 chemistry.
V3_ROUNDS = (99, 96, 96)
V3_ADTS = 14
V3_UMI = 10


def test_v3_widths_match_the_original_hardcoded_values():
    enc = Encoder.build(V3_ROUNDS, V3_ADTS, V3_UMI)
    assert enc.cell_bits == 20     # original CELL_BITS
    assert enc.adt_bits == 4       # original ADT_BITS
    assert enc.umi_bits == 20      # original 2 * UMI_LEN
    assert enc.n_cells == 99 * 96 * 96 == 912_384


def test_v3_cell_packing_matches_the_original_formula():
    enc = Encoder.build(V3_ROUNDS, V3_ADTS, V3_UMI)
    assert enc.radix_multipliers == (96 * 96, 96, 1)
    rng = np.random.default_rng(0)
    for _ in range(500):
        rt, bc2, bc3 = (int(rng.integers(0, n)) for n in V3_ROUNDS)
        assert enc.pack_cell_id([rt, bc2, bc3]) == rt * 96 * 96 + bc2 * 96 + bc3


def test_cell_id_round_trips():
    enc = Encoder.build(V3_ROUNDS, V3_ADTS, V3_UMI)
    rng = np.random.default_rng(1)
    idxs = [rng.integers(0, n, 400) for n in V3_ROUNDS]
    packed = np.array([enc.pack_cell_id([int(a[i]) for a in idxs]) for i in range(400)])
    recovered = enc.unpack_cell_id(packed)
    for original, got in zip(idxs, recovered):
        assert np.array_equal(original, got)


def test_umi_packing_matches_the_original_two_bits_per_base():
    enc = Encoder.build(V3_ROUNDS, V3_ADTS, V3_UMI)
    base_code = {65: 0, 67: 1, 71: 2, 84: 3, 78: 0}   # original _BASE_CODE, N -> 0

    def original(umi: bytes) -> int:
        val = 0
        for b in umi:
            val = (val << 2) | base_code.get(b, 0)
        return val

    for umi in (b"ACGTACGTAC", b"AAAAAAAAAA", b"TTTTTTTTTT", b"NNNNNNNNNN", b"ACGTNACGTN"):
        assert enc.pack_umi(umi) == original(umi), umi


def test_ambiguous_bases_fold_to_a_not_garbage():
    enc = Encoder.build(V3_ROUNDS, V3_ADTS, V3_UMI)
    assert enc.pack_umi(b"NNNNNNNNNN") == enc.pack_umi(b"AAAAAAAAAA")


def test_composite_round_trips():
    enc = Encoder.build(V3_ROUNDS, V3_ADTS, V3_UMI)
    rng = np.random.default_rng(2)
    n = 1000
    cell = rng.integers(0, enc.n_cells, n).astype(np.uint32)
    adt = rng.integers(0, V3_ADTS, n).astype(np.uint8)
    umi = rng.integers(0, 1 << 20, n).astype(np.uint32)

    composite = enc.pack_composite(cell, adt, umi)
    got_cell, got_adt, got_umi = enc.unpack_composite(composite)
    assert np.array_equal(got_cell, cell)
    assert np.array_equal(got_adt, adt)
    assert np.array_equal(got_umi, umi)


def test_composite_sort_order_groups_by_cell_then_adt():
    """Dedup relies on this: sorting the key must group (cell, ADT) with UMI ascending."""
    enc = Encoder.build(V3_ROUNDS, V3_ADTS, V3_UMI)
    cell = np.array([5, 5, 5, 4], dtype=np.uint32)
    adt = np.array([2, 2, 1, 9], dtype=np.uint8)
    umi = np.array([99, 3, 50, 7], dtype=np.uint32)
    keys = enc.pack_composite(cell, adt, umi)
    order = np.argsort(keys)
    assert list(order) == [3, 2, 1, 0]


def test_widths_grow_with_a_bigger_panel():
    """The original's 4 ADT bits silently capped the panel at 16."""
    assert Encoder.build(V3_ROUNDS, 16, V3_UMI).adt_bits == 4
    assert Encoder.build(V3_ROUNDS, 17, V3_UMI).adt_bits == 5
    assert Encoder.build(V3_ROUNDS, 300, V3_UMI).adt_dtype == np.uint32


def test_dtypes_adapt_to_the_cell_space():
    small = Encoder.build((8, 6), 6, 8)
    assert small.cell_dtype == np.uint32
    assert small.umi_dtype == np.uint32
    long_umi = Encoder.build((8, 6), 6, 20)
    assert long_umi.umi_bits == 40
    assert long_umi.umi_dtype == np.uint64


def test_chemistry_too_large_for_64_bits_is_refused():
    with pytest.raises(EncodingError, match="64-bit"):
        Encoder.build((10_000, 10_000, 10_000), 1000, 25)


def test_empty_whitelist_is_refused():
    with pytest.raises(EncodingError, match="non-empty"):
        Encoder.build((99, 0), 14, 10)


def test_alternative_chemistry_packs_consistently():
    """Two rounds of unequal size still form a correct mixed-radix number."""
    enc = Encoder.build((5, 7), 6, 8)
    assert enc.radix_multipliers == (7, 1)
    seen = set()
    for a in range(5):
        for b in range(7):
            packed = enc.pack_cell_id([a, b])
            assert packed not in seen
            seen.add(packed)
            got = enc.unpack_cell_id(np.array([packed]))
            assert (int(got[0][0]), int(got[1][0])) == (a, b)
    assert len(seen) == 35 == enc.n_cells
