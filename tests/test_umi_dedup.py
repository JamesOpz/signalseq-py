"""UMI collapse behaviour."""

from __future__ import annotations

import numpy as np

from signalseq.encoding import Encoder
from signalseq.umi_dedup import collapse_umis

ENC = Encoder.build((99, 96, 96), 14, 10)
DIRECTIONAL = {"method": "directional", "max_mismatches": 1, "directional_count_ratio": 2}
EXACT = {"method": "exact", "max_mismatches": 0}


def _arrays(triples):
    cell = np.array([t[0] for t in triples], dtype=np.uint32)
    adt = np.array([t[1] for t in triples], dtype=np.uint8)
    umi = np.array([t[2] for t in triples], dtype=np.uint32)
    return cell, adt, umi


def test_exact_duplicate_reads_collapse_to_one_molecule():
    cell, adt, umi = _arrays([(1, 0, 42)] * 7)
    c, a, counts = collapse_umis(cell, adt, umi, ENC, DIRECTIONAL)
    assert list(c) == [1] and list(a) == [0]
    assert list(counts) == [1]


def test_distinct_distant_umis_stay_separate():
    # 0b00 repeated vs 0b11 repeated: differs at every base.
    far = int("11" * 10, 2)
    cell, adt, umi = _arrays([(1, 0, 0), (1, 0, far)])
    _, _, counts = collapse_umis(cell, adt, umi, ENC, DIRECTIONAL)
    assert list(counts) == [2]


def test_one_mismatch_umi_is_absorbed_when_counts_allow():
    """A 1-off UMI with a much lower count folds into its neighbour."""
    seed = 0
    neighbour = 1                      # differs in the lowest base only
    reads = [(1, 0, seed)] * 10 + [(1, 0, neighbour)]
    cell, adt, umi = _arrays(reads)
    _, _, counts = collapse_umis(cell, adt, umi, ENC, DIRECTIONAL)
    assert list(counts) == [1]


def test_equal_count_neighbours_are_not_absorbed():
    """count_j <= 2*count_i - 1 fails when both have count 1."""
    reads = [(1, 0, 0), (1, 0, 1)]
    cell, adt, umi = _arrays(reads)
    _, _, counts = collapse_umis(cell, adt, umi, ENC, DIRECTIONAL)
    assert list(counts) == [1]         # 1 <= 2*1-1 = 1, so it IS absorbed
    reads = [(1, 0, 0), (1, 0, 0), (1, 0, 1), (1, 0, 1), (1, 0, 1)]
    cell, adt, umi = _arrays(reads)
    _, _, counts = collapse_umis(cell, adt, umi, ENC, DIRECTIONAL)
    assert list(counts) == [1]         # seed=3 reads absorbs the 2-read neighbour


def test_exact_method_skips_the_mismatch_collapse():
    reads = [(1, 0, 0)] * 10 + [(1, 0, 1)]
    cell, adt, umi = _arrays(reads)
    _, _, counts = collapse_umis(cell, adt, umi, ENC, EXACT)
    assert list(counts) == [2]         # both distinct UMIs kept


def test_groups_are_independent():
    reads = [(1, 0, 0), (1, 0, 0), (1, 1, 0), (2, 0, 0)]
    cell, adt, umi = _arrays(reads)
    c, a, counts = collapse_umis(cell, adt, umi, ENC, DIRECTIONAL)
    assert list(zip(c.tolist(), a.tolist(), counts.tolist())) == [(1, 0, 1), (1, 1, 1), (2, 0, 1)]


def test_output_is_sorted_by_cell_then_adt():
    reads = [(9, 3, 1), (2, 1, 1), (9, 0, 1), (2, 5, 1)]
    cell, adt, umi = _arrays(reads)
    c, a, _ = collapse_umis(cell, adt, umi, ENC, DIRECTIONAL)
    assert list(zip(c.tolist(), a.tolist())) == [(2, 1), (2, 5), (9, 0), (9, 3)]


def test_molecule_count_never_exceeds_read_count():
    rng = np.random.default_rng(3)
    n = 5000
    cell = (rng.integers(0, 50, n)).astype(np.uint32)
    adt = rng.integers(0, 14, n).astype(np.uint8)
    umi = rng.integers(0, 500, n).astype(np.uint32)
    _, _, counts = collapse_umis(cell, adt, umi, ENC, DIRECTIONAL)
    assert counts.sum() <= n
    assert (counts >= 1).all()
