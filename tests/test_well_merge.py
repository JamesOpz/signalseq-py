"""Merging RT primers by well, split-pipe cell names, and per-cell read totals."""

from __future__ import annotations

import re
from collections import Counter

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from signalseq.cli import main
from signalseq.encoding import Encoder
from signalseq.umi_dedup import collapse_umis
from signalseq.well_merge import WellMerge, sum_by_cell_adt
from signalseq.whitelists import well_indexes

PARSE_NAME = re.compile(r"^\d{2}_\d{2}_\d{2}__s3$")


def _pair_rt_primers(chem) -> None:
    """Rewrite the rt whitelist so its 8 barcodes are 4 wells x (polyA, randhex),
    the way Parse's bc_data_rt.csv pairs T and R primers."""
    path = chem.root / "wl_rt.csv"
    df = pd.read_csv(path)
    df["well"] = ["A1", "A2", "A3", "A4"] * 2
    df["stype"] = ["T"] * 4 + ["R"] * 4
    df.to_csv(path, index=False)


def _merge_config(chem, out_dir, **overrides):
    _pair_rt_primers(chem)
    base = dict(
        inputs={
            "sublibraries": [{
                "name": "SubA", "parse_index": 3,
                "r1": str(chem.root / f"{chem.name}_R1.fastq.gz"),
                "r2": str(chem.root / f"{chem.name}_R2.fastq.gz"),
            }],
            "whitelists": {"rt_wl": {"merge_by_well": True,
                                     "stype_labels": {"T": "polyA", "R": "randhex"}}},
        },
        read_structure={"cell_barcode": {
            "name_template": "{rt_wind}_{bc2_wind}_{bc3_wind}__s{parse_index}",
        }},
        matching={"umi": {"method": "exact"}},
    )
    base.update(overrides)
    return chem.write_config(out_dir, **base)


def _run_merged(chem, out_dir, **overrides) -> ad.AnnData:
    cfg_path = _merge_config(chem, out_dir, **overrides)
    assert main(["adt", "run", "--config", str(cfg_path), "--sublibrary", "SubA"]) == 0
    return ad.read_h5ad(out_dir / "SubA" / "SubA_raw.h5ad")


# --------------------------------------------------------------------------
# Units
# --------------------------------------------------------------------------

def test_well_indexes_follow_split_pipe_row_major_numbering():
    assert list(well_indexes(["A1", "A12", "B1", "D12", "H12"])) == ["01", "12", "13", "48", "96"]


def test_well_indexes_fall_back_to_first_appearance_for_non_plate_wells():
    assert list(well_indexes(["x", "y", "x", "z"])) == ["01", "02", "01", "03"]


def test_collapse_can_return_reads_per_pair():
    enc = Encoder.build([4, 4], n_adts=3, umi_length=4)
    cell = np.array([1, 1, 1, 2], dtype=np.uint32)
    adt = np.array([0, 0, 0, 1], dtype=np.uint8)
    umi = np.array([5, 5, 9, 5], dtype=np.uint32)
    c, a, counts, reads = collapse_umis(cell, adt, umi, enc, {"method": "exact"},
                                        return_reads=True)
    assert list(counts) == [2, 1]
    assert list(reads) == [3, 1]


def test_merge_maps_primers_onto_their_well_and_back():
    class _WL:
        wells = np.array(["A1", "A2", "A1", "A2"])
        stype = np.array(["T", "T", "R", "R"])
        n = 4

    class _Spec:
        merge_by_well = True
        stype_labels = {"T": "polyA"}

    class _El:
        name = "rt"
        whitelist = "rt_wl"

    class _RS:
        rounds = (_El(),)

    enc = Encoder.build([4], n_adts=2, umi_length=4)
    merge = WellMerge.build(_RS(), {"rt": _WL()}, {"rt_wl": _Spec()}, enc)
    assert list(merge.merge(np.array([0, 1, 2, 3]))) == [0, 1, 0, 1]
    assert list(merge.type_of(np.array([0, 1, 2, 3]))) == [0, 0, 1, 1]
    assert merge.type_labels == ("polyA", "R")
    assert sorted(merge.members(np.array([1]))) == [1, 3]


def test_sum_by_cell_adt_combines_repeated_pairs():
    cell, adt, w = sum_by_cell_adt(np.array([5, 5, 6]), np.array([1, 1, 1]), 3,
                                   np.array([2, 3, 4]))
    assert list(cell) == [5, 6] and list(adt) == [1, 1] and list(w) == [5, 4]


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------

def test_obs_names_match_split_pipe_format(v3like, out_dir):
    raw = _run_merged(v3like, out_dir)
    assert all(PARSE_NAME.match(n) for n in raw.obs_names), list(raw.obs_names[:5])
    assert not raw.obs_names.duplicated().any()
    for name, row in zip(raw.obs_names, raw.obs.itertuples()):
        assert name == f"{row.rt_wind}_{row.bc2_wind}_{row.bc3_wind}__s3"


def test_polyA_and_randhex_primers_collapse_to_one_cell_per_well(v3like, out_dir):
    raw = _run_merged(v3like, out_dir)
    assert set(raw.obs["rt_well"]) <= {"A1", "A2", "A3", "A4"}
    # The canonical (polyA) sequence is reported.
    polyA = set(v3like.round_seqs[0][:4])
    assert set(raw.obs["rt_sequence"]) <= polyA

    good = [p for p in v3like.planted if p.adt_index >= 0]
    expected_cells = {(p.round_indices[0] % 4, p.round_indices[1], p.round_indices[2])
                      for p in good}
    assert raw.n_obs == len(expected_cells)


def test_per_primer_counts_sum_to_the_merged_total(v3like, out_dir):
    raw = _run_merged(v3like, out_dir)
    obs = raw.obs
    np.testing.assert_allclose(
        obs["total_counts"], obs["total_counts_polyA"] + obs["total_counts_randhex"]
    )
    np.testing.assert_array_equal(
        obs["total_reads"], obs["total_reads_polyA"] + obs["total_reads_randhex"]
    )

    # Exact UMI dedup within each primer type, summed per well.
    good = [p for p in v3like.planted if p.adt_index >= 0]
    molecules = {(p.round_indices, p.adt_index, p.umi) for p in good}
    assert raw.X.sum() == len(molecules)
    n_randhex = sum(1 for m in molecules if m[0][0] >= 4)
    assert obs["total_counts_randhex"].sum() == n_randhex


def test_total_reads_counts_every_assigned_read(v3like, out_dir):
    raw = _run_merged(v3like, out_dir)
    good = [p for p in v3like.planted if p.adt_index >= 0]
    assert raw.obs["total_reads"].sum() == len(good)

    by_cell = Counter(
        f"{p.round_indices[0] % 4 + 1:02d}_{p.round_indices[1] + 1:02d}_"
        f"{p.round_indices[2] + 1:02d}__s3"
        for p in good
    )
    assert raw.obs["total_reads"].to_dict() == dict(by_cell)


def test_saturation_at_full_depth_agrees_with_the_merged_matrix(v3like, out_dir):
    _run_merged(v3like, out_dir, qc={"saturation": {"enabled": True}})
    filtered = ad.read_h5ad(out_dir / "SubA" / "SubA_filtered.h5ad")
    curve = pd.read_csv(out_dir / "SubA" / "SubA_saturation.csv")
    full = curve[curve["fraction"] == 1.0].iloc[0]
    assert full["median_umis_per_cell"] == np.median(filtered.obs["total_counts"])
    assert full["median_adts_detected_per_cell"] == np.median(filtered.obs["n_adts_detected"])


def test_no_merge_keeps_primers_separate_and_adds_no_primer_columns(v3like, out_dir):
    _pair_rt_primers(v3like)
    cfg_path = v3like.write_config(out_dir)
    main(["adt", "run", "--config", str(cfg_path), "--sublibrary", "SubA"])
    raw = ad.read_h5ad(out_dir / "SubA" / "SubA_raw.h5ad")
    assert not any(c.startswith("total_counts_") for c in raw.obs.columns)
    assert "total_reads" in raw.obs.columns
    assert "rt_wind" in raw.obs.columns


def test_aggregate_writes_unfiltered_and_filtered_combined_objects(v3like, out_dir):
    cfg_path = _merge_config(v3like, out_dir)
    import yaml
    data = yaml.safe_load(open(cfg_path))
    sub = data["inputs"]["sublibraries"][0]
    data["inputs"]["sublibraries"].append({**sub, "name": "SubB", "parse_index": 4})
    yaml.safe_dump(data, open(cfg_path, "w"))

    assert main(["adt", "run-multi", "--config", str(cfg_path)]) == 0
    combined_raw = ad.read_h5ad(out_dir / "combined" / "combined_raw.h5ad")
    a = ad.read_h5ad(out_dir / "SubA" / "SubA_raw.h5ad")
    b = ad.read_h5ad(out_dir / "SubB" / "SubB_raw.h5ad")
    assert combined_raw.n_obs == a.n_obs + b.n_obs
    assert not combined_raw.obs_names.duplicated().any()
    assert {n[-4:] for n in combined_raw.obs_names} == {"__s3", "__s4"}
    assert "X_pca" not in combined_raw.obsm
    assert (out_dir / "combined" / "combined_filtered.h5ad").is_file()
