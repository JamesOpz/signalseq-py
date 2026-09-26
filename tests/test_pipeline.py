"""End-to-end runs through the CLI on synthetic data, for both chemistries."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pytest
import yaml

from signalseq.cli import main


def _run(argv) -> int:
    return main(argv)


def test_validate_verb_succeeds_on_a_good_config(v3like, out_dir, capsys):
    cfg_path = v3like.write_config(out_dir)
    assert _run(["adt", "validate", "--config", str(cfg_path)]) == 0
    assert "OK" in capsys.readouterr().out


def test_validate_verb_fails_and_explains(v3like, out_dir, capsys):
    cfg_path = v3like.write_config(out_dir)
    data = yaml.safe_load(open(cfg_path))
    data["read_structure"]["elements"]["rt"]["length"] = 3
    yaml.safe_dump(data, open(cfg_path, "w"))

    assert _run(["adt", "validate", "--config", str(cfg_path)]) == 1
    out = capsys.readouterr().out
    assert "FAILED" in out and "ERROR" in out


def test_run_produces_the_expected_outputs(chemistry, out_dir):
    cfg_path = chemistry.write_config(out_dir)
    assert _run(["adt", "run", "--config", str(cfg_path), "--sublibrary", "SubA"]) == 0

    sub_out = out_dir / "SubA"
    assert (sub_out / "SubA_raw.h5ad").is_file()
    assert (sub_out / "SubA_filtered.h5ad").is_file()
    assert (sub_out / "SubA_qc_stats.json").is_file()
    # Provenance lands in every output directory.
    assert (sub_out / "resolved_config.yaml").is_file()
    assert (sub_out / "provenance.json").is_file()


def test_counts_match_the_planted_reads(chemistry, out_dir):
    """Total molecules must never exceed the reads that were actually planted."""
    cfg_path = chemistry.write_config(out_dir)
    _run(["adt", "run", "--config", str(cfg_path), "--sublibrary", "SubA"])

    raw = ad.read_h5ad(out_dir / "SubA" / "SubA_raw.h5ad")
    stats = json.loads((out_dir / "SubA" / "SubA_qc_stats.json").read_text())["processing_stats"]

    n_good = sum(1 for p in chemistry.planted if p.adt_index >= 0)
    assert stats["total_reads"] == len(chemistry.planted)
    assert stats["valid_all"] == n_good
    assert raw.X.sum() <= n_good
    assert raw.n_vars == chemistry.n_adts


def test_obs_columns_follow_obs_labels(v3like, out_dir):
    """Column naming is config-driven via obs_label."""
    cfg_path = v3like.write_config(out_dir, read_structure={"elements": {
        "adt": {"read": "R1", "start": 1, "length": 15, "whitelist": "adt_panel", "obs_label": "ADT"},
        "umi": {"read": "R2", "start": 1, "length": 10, "obs_label": "UMI"},
        "bc3": {"read": "R2", "start": 11, "length": 8, "whitelist": "bc3_wl", "obs_label": "R3"},
        "bc2": {"read": "R2", "start": 31, "length": 8, "whitelist": "bc2_wl", "obs_label": "R2"},
        "rt": {"read": "R2", "start": 51, "length": 8, "whitelist": "rt_wl", "obs_label": "RT"},
    }})
    _run(["adt", "run", "--config", str(cfg_path), "--sublibrary", "SubA"])

    raw = ad.read_h5ad(out_dir / "SubA" / "SubA_raw.h5ad")
    for col in ("RT_well", "R2_well", "R3_well", "RT_sequence", "R2_sequence", "R3_sequence"):
        assert col in raw.obs.columns, raw.obs.columns.tolist()
    assert "cell_id_packed" in raw.obs.columns


def test_obs_names_use_the_name_template(v3like, out_dir):
    cfg_path = v3like.write_config(out_dir)
    _run(["adt", "run", "--config", str(cfg_path), "--sublibrary", "SubA"])
    raw = ad.read_h5ad(out_dir / "SubA" / "SubA_raw.h5ad")
    for name in raw.obs_names[:5]:
        parts = name.split("_")
        assert parts[0] == "SubA"
        assert len(parts) == 4          # sublibrary + three rounds


def test_run_multi_processes_and_aggregates(v3like, out_dir):
    """Two sublibraries pointing at the same FASTQs, then aggregation."""
    data = v3like.config_dict(out_dir)
    data["inputs"]["sublibraries"].append({
        "name": "SubB",
        "r1": data["inputs"]["sublibraries"][0]["r1"],
        "r2": data["inputs"]["sublibraries"][0]["r2"],
    })
    cfg_path = v3like.root / "multi.yaml"
    yaml.safe_dump(data, open(cfg_path, "w"))

    assert _run(["adt", "run-multi", "--config", str(cfg_path)]) == 0
    assert (out_dir / "SubA" / "SubA_filtered.h5ad").is_file()
    assert (out_dir / "SubB" / "SubB_filtered.h5ad").is_file()

    combined = ad.read_h5ad(out_dir / "combined" / "combined_filtered.h5ad")
    a = ad.read_h5ad(out_dir / "SubA" / "SubA_filtered.h5ad")
    b = ad.read_h5ad(out_dir / "SubB" / "SubB_filtered.h5ad")
    assert combined.n_obs == a.n_obs + b.n_obs
    assert not combined.obs_names.duplicated().any()
    assert (out_dir / "combined" / "combined_qc_stats.json").is_file()


def test_aggregate_verb_runs_on_existing_outputs(v3like, out_dir):
    data = v3like.config_dict(out_dir)
    data["inputs"]["sublibraries"].append({
        "name": "SubB",
        "r1": data["inputs"]["sublibraries"][0]["r1"],
        "r2": data["inputs"]["sublibraries"][0]["r2"],
    })
    cfg_path = v3like.root / "multi.yaml"
    yaml.safe_dump(data, open(cfg_path, "w"))

    _run(["adt", "run-multi", "--config", str(cfg_path), "--skip-aggregate"])
    assert not (out_dir / "combined").exists()

    assert _run(["adt", "aggregate", "--config", str(cfg_path)]) == 0
    assert (out_dir / "combined" / "combined_filtered.h5ad").is_file()


def test_run_requires_sublibrary_when_several_are_configured(v3like, out_dir, capsys):
    data = v3like.config_dict(out_dir)
    data["inputs"]["sublibraries"].append({
        "name": "SubB",
        "r1": data["inputs"]["sublibraries"][0]["r1"],
        "r2": data["inputs"]["sublibraries"][0]["r2"],
    })
    cfg_path = v3like.root / "multi.yaml"
    yaml.safe_dump(data, open(cfg_path, "w"))

    assert _run(["adt", "run", "--config", str(cfg_path)]) == 1
    assert "--sublibrary is required" in capsys.readouterr().err


def test_unknown_sublibrary_is_refused(v3like, out_dir, capsys):
    cfg_path = v3like.write_config(out_dir)
    assert _run(["adt", "run", "--config", str(cfg_path), "--sublibrary", "Nope"]) == 1
    assert "No sublibrary named" in capsys.readouterr().err


def test_debug_reads_caps_the_scan(v3like, out_dir):
    cfg_path = v3like.write_config(out_dir)
    _run(["adt", "run", "--config", str(cfg_path), "--sublibrary", "SubA", "--debug-reads", "25"])
    stats = json.loads((out_dir / "SubA" / "SubA_qc_stats.json").read_text())
    assert stats["processing_stats"]["total_reads"] == 25


def test_provenance_records_version_and_config(v3like, out_dir):
    cfg_path = v3like.write_config(out_dir)
    _run(["adt", "run", "--config", str(cfg_path), "--sublibrary", "SubA"])
    prov = json.loads((out_dir / "SubA" / "provenance.json").read_text())
    assert prov["signalseq_version"]
    assert prov["run_name"] == "test_v3like"
    assert "git" in prov
    assert "read_structure" in prov
    resolved = yaml.safe_load(open(out_dir / "SubA" / "resolved_config.yaml"))
    assert resolved["read_structure"]["elements"]["rt"]["start"] == 51


def test_compatibility_flags_default_to_original_behaviour(v3like, out_dir):
    """valid_cell_barcode == valid_all is the ported (buggy) behaviour."""
    cfg_path = v3like.write_config(out_dir)
    _run(["adt", "run", "--config", str(cfg_path), "--sublibrary", "SubA"])
    stats = json.loads((out_dir / "SubA" / "SubA_qc_stats.json").read_text())["processing_stats"]
    assert stats["valid_cell_barcode"] == stats["valid_all"]


def test_fixing_the_counter_changes_only_that_number(v3like, out_dir, tmp_path):
    """With the fix on, valid_cell_barcode must exceed valid_all (junk ADT reads
    still have resolvable barcodes)."""
    plain = v3like.write_config(out_dir)
    _run(["adt", "run", "--config", str(plain), "--sublibrary", "SubA"])
    before = json.loads((out_dir / "SubA" / "SubA_qc_stats.json").read_text())

    fixed_out = tmp_path / "runs" / "fixed"
    fixed = v3like.write_config(
        fixed_out, compatibility={"fix_valid_cell_barcode_counter": True}
    )
    _run(["adt", "run", "--config", str(fixed), "--sublibrary", "SubA"])
    after = json.loads((fixed_out / "SubA" / "SubA_qc_stats.json").read_text())

    assert after["processing_stats"]["valid_cell_barcode"] > after["processing_stats"]["valid_all"]
    # Everything else is untouched.
    assert before["processing_stats"]["valid_all"] == after["processing_stats"]["valid_all"]
    assert before["matrix_stats"] == after["matrix_stats"]
