"""End-to-end barcode and UMI extraction from synthetic FASTQs.

Runs against two deliberately different read structures (see conftest): one
shaped like Parse Evercode WT v3, and one with the ADT on R2, the UMI last, two
rounds of unequal length on R1, and nothing starting at position 1. Both are
driven purely by config.
"""

from __future__ import annotations

import numpy as np
import pytest

from signalseq.bc_correct import BarcodeCorrector, run_census
from signalseq.config import load_config
from signalseq.encoding import Encoder
from signalseq.hot_loop import scan_sublibrary
from signalseq.whitelists import load_all


def _prepare(chem, out_dir):
    cfg = load_config(chem.write_config(out_dir))
    panel, whitelists = load_all(cfg)
    rs = cfg.read_structure
    encoder = Encoder.build(
        [whitelists[el.name].n for el in rs.rounds], panel.n, rs.umi.length
    )
    sub = cfg.sublibraries[0]
    barcode_path = sub.read_path(rs.rounds[0].read)
    census = run_census(barcode_path, rs, whitelists,
                        cfg.matching["cell_barcode"]["census"],
                        cfg.performance["gzip_binaries"])
    corrector = BarcodeCorrector.build(rs, whitelists, census, cfg.matching["cell_barcode"])
    return cfg, panel, whitelists, encoder, corrector, sub


def test_extraction_recovers_planted_barcodes_and_umis(chemistry, out_dir):
    """Every read whose ADT tag was planted intact must come back with exactly
    the barcode rounds and UMI it was built from."""
    chem = chemistry
    cfg, panel, whitelists, encoder, corrector, sub = _prepare(chem, out_dir)
    rs = cfg.read_structure

    cell_id, adt_idx, umi, stats, reads_per_round = scan_sublibrary(
        sub.r1, sub.r2, rs, panel, whitelists, corrector, encoder,
        chunk_rows=int(cfg.performance["chunk_rows"]),
        gzip_binaries=cfg.performance["gzip_binaries"],
    )

    expected = [p for p in chem.planted if p.adt_index >= 0]
    assert stats["total_reads"] == len(chem.planted)
    assert stats["valid_adt"] == len(expected)
    assert stats["valid_all"] == len(expected)
    assert len(cell_id) == len(expected)

    # Round indices, in the configured combining order.
    order = [chem.round_names.index(el.name) for el in rs.rounds]
    decoded = encoder.unpack_cell_id(cell_id)
    for row, planted in enumerate(expected):
        got = tuple(int(d[row]) for d in decoded)
        want = tuple(planted.round_indices[i] for i in order)
        assert got == want, f"read {row}: rounds {got} != planted {want}"
        assert int(adt_idx[row]) == planted.adt_index
        assert int(umi[row]) == encoder.pack_umi(planted.umi.encode())


def test_unmatched_adt_tags_are_dropped_not_miscounted(chemistry, out_dir):
    chem = chemistry
    cfg, panel, whitelists, encoder, corrector, sub = _prepare(chem, out_dir)
    n_junk = sum(1 for p in chem.planted if p.adt_index < 0)
    assert n_junk > 0, "fixture should plant some unmatchable reads"

    _, _, _, stats, _ = scan_sublibrary(
        sub.r1, sub.r2, cfg.read_structure, panel, whitelists, corrector, encoder,
        chunk_rows=int(cfg.performance["chunk_rows"]),
        gzip_binaries=cfg.performance["gzip_binaries"],
    )
    assert stats["valid_adt"] == stats["total_reads"] - n_junk


def test_reads_per_round_totals_match_valid_reads(chemistry, out_dir):
    chem = chemistry
    cfg, panel, whitelists, encoder, corrector, sub = _prepare(chem, out_dir)
    _, _, _, stats, reads_per_round = scan_sublibrary(
        sub.r1, sub.r2, cfg.read_structure, panel, whitelists, corrector, encoder,
        chunk_rows=int(cfg.performance["chunk_rows"]),
        gzip_binaries=cfg.performance["gzip_binaries"],
    )
    for name, counts in reads_per_round.items():
        assert counts.sum() == stats["valid_all"], name


def test_shifting_an_element_by_one_breaks_extraction(v3like, out_dir):
    """Guards the 1-based convention: a start one too high must stop matching."""
    chem = v3like
    good = load_config(chem.write_config(out_dir))
    panel, whitelists = load_all(good)
    encoder = Encoder.build(
        [whitelists[el.name].n for el in good.read_structure.rounds], panel.n,
        good.read_structure.umi.length,
    )
    sub = good.sublibraries[0]
    census = run_census(sub.r2, good.read_structure, whitelists,
                        good.matching["cell_barcode"]["census"],
                        good.performance["gzip_binaries"])
    corrector = BarcodeCorrector.build(good.read_structure, whitelists, census,
                                       good.matching["cell_barcode"])
    _, _, _, good_stats, _ = scan_sublibrary(
        sub.r1, sub.r2, good.read_structure, panel, whitelists, corrector, encoder,
        chunk_rows=1000, gzip_binaries=good.performance["gzip_binaries"],
    )
    assert good_stats["valid_all"] > 0

    # Now slide the RT round one base to the right.
    shifted_elements = {k: dict(v) for k, v in chem.elements.items()}
    shifted_elements["rt"]["start"] += 1
    cfg_path = chem.root / "shifted.yaml"
    import yaml
    data = chem.config_dict(out_dir)
    data["read_structure"]["elements"]["rt"]["start"] += 1
    with open(cfg_path, "w") as fh:
        yaml.safe_dump(data, fh)

    shifted = load_config(cfg_path)
    panel2, wl2 = load_all(shifted)
    enc2 = Encoder.build(
        [wl2[el.name].n for el in shifted.read_structure.rounds], panel2.n,
        shifted.read_structure.umi.length,
    )
    census2 = run_census(sub.r2, shifted.read_structure, wl2,
                         shifted.matching["cell_barcode"]["census"],
                         shifted.performance["gzip_binaries"])
    corrector2 = BarcodeCorrector.build(shifted.read_structure, wl2, census2,
                                        shifted.matching["cell_barcode"])
    _, _, _, shifted_stats, _ = scan_sublibrary(
        sub.r1, sub.r2, shifted.read_structure, panel2, wl2, corrector2, enc2,
        chunk_rows=1000, gzip_binaries=shifted.performance["gzip_binaries"],
    )
    assert shifted_stats["valid_all"] == 0, "an off-by-one offset must not still match"


def test_umi_packing_is_reversible_for_planted_umis(chemistry, out_dir):
    chem = chemistry
    cfg, panel, whitelists, encoder, corrector, sub = _prepare(chem, out_dir)
    code_to_base = "ACGT"
    for planted in chem.planted[:20]:
        packed = encoder.pack_umi(planted.umi.encode())
        # Unpack manually, 2 bits per base, most significant base first.
        bases = []
        for i in reversed(range(encoder.umi_length)):
            bases.append(code_to_base[(packed >> (2 * i)) & 0b11])
        assert "".join(bases) == planted.umi


def test_two_structures_produce_different_cell_spaces(v3like, altchem, tmp_path):
    """Sanity: the two fixtures really are different chemistries."""
    a = load_config(v3like.write_config(tmp_path / "outa"))
    b = load_config(altchem.write_config(tmp_path / "outb"))
    assert a.read_structure.min_length("R2") != b.read_structure.min_length("R2")
    assert a.read_structure.adt.read != b.read_structure.adt.read
    assert a.read_structure.umi.length != b.read_structure.umi.length
    assert len(a.read_structure.rounds) != len(b.read_structure.rounds)
