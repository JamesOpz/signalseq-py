"""Config loading and the fail-early validation checks."""

from __future__ import annotations

import gzip
from pathlib import Path

import pytest
import yaml

from signalseq.config import load_config, repo_root
from signalseq.readstructure import ConfigError
from signalseq.validate import validate


def _write(chem, out_dir, **overrides) -> Path:
    return chem.write_config(out_dir, **overrides)


def test_valid_config_passes_every_check(v3like, out_dir):
    cfg = load_config(_write(v3like, out_dir))
    report = validate(cfg)
    assert report.ok, report.errors


def test_missing_input_file_is_reported_with_its_key(v3like, out_dir):
    path = _write(v3like, out_dir)
    data = yaml.safe_load(open(path))
    data["inputs"]["sublibraries"][0]["r1"] = str(v3like.root / "nope_R1.fastq.gz")
    yaml.safe_dump(data, open(path, "w"))

    report = validate(load_config(path))
    assert not report.ok
    assert any("sublibraries[SubA].r1" in e and "not found" in e for e in report.errors), report.errors


def test_missing_whitelist_file_is_reported(v3like, out_dir):
    path = _write(v3like, out_dir)
    data = yaml.safe_load(open(path))
    data["inputs"]["whitelists"]["rt_wl"]["csv"] = str(v3like.root / "gone.csv")
    yaml.safe_dump(data, open(path, "w"))

    report = validate(load_config(path))
    assert not report.ok
    assert any("whitelists.rt_wl" in e for e in report.errors), report.errors


def test_output_dir_inside_the_repo_is_refused():
    """The hard rule: outputs must never land inside the git repository."""
    root = repo_root()
    assert root is not None, "this test expects to run from the git checkout"
    with pytest.raises(ConfigError, match="inside the signalseq repository"):
        load_config(_repo_output_config(root))


def _repo_output_config(root: Path) -> Path:
    """A config whose output_dir points inside the repo, written to a temp file."""
    import tempfile
    cfg = {
        "schema_version": 1,
        "run": {"name": "bad", "output_dir": str(root / "outs" / "should_be_refused")},
        "inputs": {
            "sublibraries": [{"name": "S", "r1": "a.fastq.gz", "r2": "b.fastq.gz"}],
            "whitelists": {"rt_wl": {"csv": "wl.csv"}},
            "adt_panel": {"csv": "panel.csv"},
            "rt_layout": {"csv": "layout.csv"},
        },
        "read_structure": {
            "elements": {
                "adt": {"read": "R1", "start": 1, "length": 15, "whitelist": "adt_panel"},
                "umi": {"read": "R2", "start": 1, "length": 10},
                "rt": {"read": "R2", "start": 51, "length": 8, "whitelist": "rt_wl"},
            },
            "cell_barcode": {"order": ["rt"]},
        },
    }
    fh = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump(cfg, fh)
    fh.close()
    return Path(fh.name)


def test_barcode_length_mismatch_is_reported(v3like, out_dir):
    """A declared length that disagrees with the whitelist file must fail."""
    path = _write(v3like, out_dir)
    data = yaml.safe_load(open(path))
    data["read_structure"]["elements"]["rt"]["length"] = 7   # file holds 8bp
    yaml.safe_dump(data, open(path, "w"))

    report = validate(load_config(path))
    assert not report.ok
    assert any("8bp barcodes" in e and "length" in e for e in report.errors), report.errors


def test_adt_panel_length_mismatch_is_reported(v3like, out_dir):
    path = _write(v3like, out_dir)
    data = yaml.safe_load(open(path))
    data["read_structure"]["elements"]["adt"]["length"] = 14  # panel holds 15bp
    yaml.safe_dump(data, open(path, "w"))

    report = validate(load_config(path))
    assert not report.ok
    assert any("adt_panel" in e and "15bp" in e for e in report.errors), report.errors


def test_element_running_past_the_read_end_is_reported(v3like, out_dir):
    """Checked against the actual first reads, not just declared numbers."""
    path = _write(v3like, out_dir)
    data = yaml.safe_load(open(path))
    data["read_structure"]["elements"]["rt"]["start"] = 400   # far beyond a 62bp read
    yaml.safe_dump(data, open(path, "w"))

    report = validate(load_config(path))
    assert not report.ok
    assert any("shorter" in e and "running past the end" in e for e in report.errors), report.errors


def test_overlapping_elements_are_refused_at_load(v3like, out_dir):
    path = _write(v3like, out_dir)
    data = yaml.safe_load(open(path))
    data["read_structure"]["elements"]["bc2"]["start"] = 5   # collides with the UMI
    yaml.safe_dump(data, open(path, "w"))

    with pytest.raises(ConfigError, match="overlap"):
        load_config(path)


def test_non_empty_output_dir_needs_overwrite(v3like, out_dir):
    out_dir.mkdir(parents=True)
    (out_dir / "stale.txt").write_text("previous run")
    path = _write(v3like, out_dir, run={"overwrite": False})

    report = validate(load_config(path))
    assert not report.ok
    assert any("not empty" in e for e in report.errors), report.errors


def test_overwrite_allows_a_non_empty_output_dir(v3like, out_dir):
    out_dir.mkdir(parents=True)
    (out_dir / "stale.txt").write_text("previous run")
    path = _write(v3like, out_dir, run={"overwrite": True})
    assert validate(load_config(path)).ok


def test_unknown_whitelist_reference_is_refused(v3like, out_dir):
    path = _write(v3like, out_dir)
    data = yaml.safe_load(open(path))
    data["read_structure"]["elements"]["rt"]["whitelist"] = "does_not_exist"
    yaml.safe_dump(data, open(path, "w"))

    with pytest.raises(ConfigError, match="not defined"):
        load_config(path)


def test_schema_version_mismatch_is_refused(v3like, out_dir):
    path = _write(v3like, out_dir)
    data = yaml.safe_load(open(path))
    data["schema_version"] = 99
    yaml.safe_dump(data, open(path, "w"))

    with pytest.raises(ConfigError, match="schema_version"):
        load_config(path)


def test_relative_paths_resolve_against_the_config_file(v3like, out_dir):
    """A config plus its reference files should be movable as a unit."""
    path = _write(v3like, out_dir)
    data = yaml.safe_load(open(path))
    data["inputs"]["adt_panel"]["csv"] = "panel.csv"      # sits beside the config
    yaml.safe_dump(data, open(path, "w"))

    cfg = load_config(path)
    assert cfg.panel.csv == (v3like.root / "panel.csv").resolve()


def test_bad_threads_value_is_refused(v3like, out_dir):
    path = _write(v3like, out_dir, run={"threads": 0})
    with pytest.raises(ConfigError, match="threads"):
        load_config(path)


def test_min_above_max_umi_is_refused(v3like, out_dir):
    path = _write(v3like, out_dir,
                  filtering={"min_umi_per_cell": 500, "max_umi_per_cell": 100})
    with pytest.raises(ConfigError, match="greater than"):
        load_config(path)


def test_defaults_reproduce_the_original_parameters(v3like, out_dir):
    """A config that omits the tuning sections must still carry the ported values."""
    path = _write(v3like, out_dir)
    data = yaml.safe_load(open(path))
    for section in ("matching", "filtering"):
        data.pop(section, None)
    yaml.safe_dump(data, open(path, "w"))

    cfg = load_config(path)
    cb = cfg.matching["cell_barcode"]
    assert cfg.matching["adt"]["max_mismatches"] == 1
    assert cfg.matching["adt"]["drop_ambiguous"] is True
    assert cb["max_edit_distance"] == 2
    assert cb["census"]["offset_reads"] == 3_000_000
    assert cb["census"]["n_reads"] == 1_000_000
    assert cb["census"]["perfect_cell_threshold"] == 0.92
    assert cb["census"]["min_count_threshold"] == 5
    assert cb["census"]["accept"] == "strictly_greater"
    assert cfg.matching["umi"] == {
        "method": "directional", "max_mismatches": 1, "directional_count_ratio": 2
    }
    assert cfg.filtering["knee"]["exclude_top_n"] == 3
    assert cfg.performance["chunk_rows"] == 25_000_000
    assert cfg.compatibility == {
        "fix_valid_cell_barcode_counter": False,
        "fix_missing_timepoint_label": False,
    }


def test_shipped_example_config_is_structurally_valid():
    """configs/parse_v3.example.yaml must parse and carry the original's numbers."""
    root = repo_root()
    example = root / "configs" / "parse_v3.example.yaml"
    data = yaml.safe_load(open(example))

    els = data["read_structure"]["elements"]
    assert (els["adt"]["start"], els["adt"]["length"]) == (1, 15)
    assert (els["umi"]["start"], els["umi"]["length"]) == (1, 10)
    assert (els["bc3"]["start"], els["bc3"]["length"]) == (11, 8)
    assert (els["bc2"]["start"], els["bc2"]["length"]) == (31, 8)
    assert (els["rt"]["start"], els["rt"]["length"]) == (51, 8)
    assert data["read_structure"]["cell_barcode"]["order"] == ["rt", "bc2", "bc3"]
    assert data["filtering"]["min_umi_per_cell"] == 250
    assert data["filtering"]["min_adts_detected"] == 1
    # Linkers must ship disabled: the original never inspects them.
    assert all(l["enabled"] is False for l in data["read_structure"]["linkers"])
    # And no defect fixes on by default.
    assert all(v is False for v in data["compatibility"].values())


def test_example_config_contains_no_real_paths():
    """The shipped example must not leak a developer's filesystem."""
    root = repo_root()
    text = (root / "configs" / "parse_v3.example.yaml").read_text()
    assert "/Users/" not in text
    assert "/home/" not in text
    assert "PATH/TO" in text
