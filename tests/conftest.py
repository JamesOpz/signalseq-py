"""Synthetic fixtures. No real data, no reference files from disk.

`Chemistry` builds a complete, self-consistent input set -- whitelists, panel,
layout, and a FASTQ pair whose reads are assembled from known barcodes -- so a
test can assert that extraction recovers exactly what was planted.
"""

from __future__ import annotations

import gzip
import random
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import yaml

BASES = "ACGT"


def _seqs(n: int, length: int, seed: int) -> list[str]:
    """`n` distinct random sequences, all `length` long."""
    rng = random.Random(seed)
    out: set[str] = set()
    while len(out) < n:
        out.add("".join(rng.choice(BASES) for _ in range(length)))
    return sorted(out)


def _wells(n: int) -> list[str]:
    rows = "ABCDEFGH"
    return [f"{rows[i // 12]}{i % 12 + 1}" for i in range(n)]


@dataclass
class PlantedRead:
    """One read pair and the truth it was built from."""

    adt_index: int
    round_indices: tuple[int, ...]
    umi: str
    r1: str
    r2: str


@dataclass
class Chemistry:
    """A synthetic chemistry: layout, reference files and a FASTQ pair."""

    root: Path
    name: str
    adt_length: int
    umi_length: int
    round_lengths: tuple[int, ...]
    round_names: tuple[str, ...]
    elements: dict                     # element name -> {read, start, length, ...}
    n_adts: int = 6
    n_rounds_entries: tuple[int, ...] = (8, 6, 6)
    r1_length: int = 0
    r2_length: int = 0
    planted: list[PlantedRead] = field(default_factory=list)

    # -- reference files ---------------------------------------------------
    def write_references(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.adt_seqs = _seqs(self.n_adts, self.adt_length, seed=1)
        lines = ["Barcode,ADT_identifier,Antigen,Clone"]
        for i, s in enumerate(self.adt_seqs):
            lines.append(f"{s},SYN{i:03d},Antigen_{i},C{i}")
        (self.root / "panel.csv").write_text("\n".join(lines) + "\n")

        self.round_seqs: list[list[str]] = []
        for r, (rname, rlen, n) in enumerate(
            zip(self.round_names, self.round_lengths, self.n_rounds_entries)
        ):
            seqs = _seqs(n, rlen, seed=100 + r)
            self.round_seqs.append(seqs)
            wells = _wells(n)
            rows = ["bci,sequence,uid,well,stype"]
            for i, (s, w) in enumerate(zip(seqs, wells), start=1):
                rows.append(f"{i},{s},syn_{i},{w},T")
            (self.root / f"wl_{rname}.csv").write_text("\n".join(rows) + "\n")

        # Layout covers only the first few wells of the joining round, so the
        # rest exercise the "Unassigned" path.
        join_wells = _wells(self.n_rounds_entries[0])
        rows = ["Well_ID,Sample_name"]
        for i, w in enumerate(join_wells[:4]):
            culture = "PDO" if i % 2 == 0 else "PDO+CAF"
            rows.append(f"{w},{culture}_{6 * (i + 1)}hr_Treatment")
        (self.root / "layout.csv").write_text("\n".join(rows) + "\n")

    # -- reads -------------------------------------------------------------
    def build_reads(self, n_reads: int = 120, seed: int = 7,
                    junk_fraction: float = 0.1) -> None:
        """Assemble read pairs from known barcodes, with some unmatchable junk."""
        rng = random.Random(seed)
        self.r1_length = max(
            e["start"] - 1 + e["length"] for e in self.elements.values() if e["read"] == "R1"
        )
        self.r2_length = max(
            e["start"] - 1 + e["length"] for e in self.elements.values() if e["read"] == "R2"
        )
        # Pad a little so the reads are longer than the structure strictly needs.
        self.r1_length += 4
        self.r2_length += 4

        self.planted = []
        for _ in range(n_reads):
            adt_i = rng.randrange(self.n_adts)
            round_i = tuple(rng.randrange(n) for n in self.n_rounds_entries)
            umi = "".join(rng.choice(BASES) for _ in range(self.umi_length))

            r1 = ["A"] * self.r1_length
            r2 = ["A"] * self.r2_length
            junk = rng.random() < junk_fraction

            def place(read_chars, element, text):
                s = element["start"] - 1
                for k, ch in enumerate(text):
                    read_chars[s + k] = ch

            for ename, element in self.elements.items():
                target = r1 if element["read"] == "R1" else r2
                if ename == self.adt_element:
                    text = ("T" * self.adt_length) if junk else self.adt_seqs[adt_i]
                elif ename == self.umi_element:
                    text = umi
                else:
                    ri = self.round_names.index(ename)
                    text = self.round_seqs[ri][round_i[ri]]
                place(target, element, text)

            self.planted.append(PlantedRead(
                adt_index=-1 if junk else adt_i,
                round_indices=round_i,
                umi=umi,
                r1="".join(r1),
                r2="".join(r2),
            ))

    def write_fastqs(self) -> tuple[Path, Path]:
        r1_path = self.root / f"{self.name}_R1.fastq.gz"
        r2_path = self.root / f"{self.name}_R2.fastq.gz"
        for path, attr in ((r1_path, "r1"), (r2_path, "r2")):
            with gzip.open(path, "wt") as fh:
                for i, read in enumerate(self.planted):
                    seq = getattr(read, attr)
                    fh.write(f"@syn:{i}\n{seq}\n+\n{'F' * len(seq)}\n")
        return r1_path, r2_path

    # -- config ------------------------------------------------------------
    adt_element: str = "adt"
    umi_element: str = "umi"

    def config_dict(self, output_dir: Path, **overrides) -> dict:
        whitelists = {
            f"{rname}_wl": {
                "csv": str(self.root / f"wl_{rname}.csv"),
                "sequence_column": "sequence",
                "well_column": "well",
                "stype_column": "stype",
                "edit_distance_json": None,
            }
            for rname in self.round_names
        }
        elements = {}
        for ename, element in self.elements.items():
            entry = dict(element)
            if ename == self.adt_element:
                entry["whitelist"] = "adt_panel"
            elif ename in self.round_names:
                entry["whitelist"] = f"{ename}_wl"
            elements[ename] = entry

        cfg = {
            "schema_version": 1,
            "run": {
                "name": f"test_{self.name}",
                "output_dir": str(output_dir),
                "threads": 1,
                "overwrite": True,
            },
            "inputs": {
                "sublibraries": [{
                    "name": "SubA",
                    "r1": str(self.root / f"{self.name}_R1.fastq.gz"),
                    "r2": str(self.root / f"{self.name}_R2.fastq.gz"),
                }],
                "whitelists": whitelists,
                "adt_panel": {
                    "csv": str(self.root / "panel.csv"),
                    "barcode_column": "Barcode",
                    "name_column": "Antigen",
                },
                "rt_layout": {
                    "csv": str(self.root / "layout.csv"),
                    "well_column": "Well_ID",
                    "sample_column": "Sample_name",
                },
            },
            "read_structure": {
                "coordinates": "one_based",
                "adt_element": self.adt_element,
                "umi_element": self.umi_element,
                "elements": elements,
                "cell_barcode": {
                    "order": list(self.round_names),
                    "name_template": "{sublibrary}_" + "_".join(
                        "{" + r + "}" for r in self.round_names
                    ),
                },
            },
            "matching": {
                "cell_barcode": {
                    "census": {"offset_reads": 0, "n_reads": 1000, "min_count_threshold": 0},
                },
            },
            "filtering": {"min_umi_per_cell": 0, "min_adts_detected": 0},
            "qc": {
                "saturation": {"enabled": False},
                "analysis": {"enabled": False},
                "plots": {"enabled": False},
            },
        }
        _deep_update(cfg, overrides)
        return cfg

    def write_config(self, output_dir: Path, **overrides) -> Path:
        path = self.root / f"{self.name}_config.yaml"
        with open(path, "w") as fh:
            yaml.safe_dump(self.config_dict(output_dir, **overrides), fh, sort_keys=False)
        return path


def _deep_update(base: dict, extra: dict) -> None:
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v


# --------------------------------------------------------------------------
# The two chemistries
# --------------------------------------------------------------------------

def _parse_v3_like(root: Path) -> Chemistry:
    """Same shape as Parse Evercode WT v3: ADT on R1, UMI then three rounds on R2."""
    chem = Chemistry(
        root=root / "v3like",
        name="v3like",
        adt_length=15,
        umi_length=10,
        round_lengths=(8, 8, 8),
        round_names=("rt", "bc2", "bc3"),
        elements={
            "adt": {"read": "R1", "start": 1, "length": 15},
            "umi": {"read": "R2", "start": 1, "length": 10},
            "bc3": {"read": "R2", "start": 11, "length": 8},
            "bc2": {"read": "R2", "start": 31, "length": 8},
            "rt": {"read": "R2", "start": 51, "length": 8},
        },
    )
    chem.write_references()
    chem.build_reads()
    chem.write_fastqs()
    return chem


def _alt_chemistry(root: Path) -> Chemistry:
    """A deliberately different chemistry: ADT on R2, UMI last, two rounds of
    unequal length on R1, nothing starting at position 1."""
    chem = Chemistry(
        root=root / "altchem",
        name="altchem",
        adt_length=12,
        umi_length=8,
        round_lengths=(6, 10),
        round_names=("cb1", "cb2"),
        n_rounds_entries=(5, 7),
        elements={
            "adt": {"read": "R2", "start": 5, "length": 12},
            "umi": {"read": "R2", "start": 21, "length": 8},
            "cb1": {"read": "R1", "start": 3, "length": 6},
            "cb2": {"read": "R1", "start": 15, "length": 10},
        },
    )
    chem.write_references()
    chem.build_reads()
    chem.write_fastqs()
    return chem


@pytest.fixture
def v3like(tmp_path) -> Chemistry:
    return _parse_v3_like(tmp_path)


@pytest.fixture
def altchem(tmp_path) -> Chemistry:
    return _alt_chemistry(tmp_path)


@pytest.fixture(params=["v3like", "altchem"])
def chemistry(request, tmp_path) -> Chemistry:
    """Both chemistries, so extraction tests run against each."""
    return (_parse_v3_like if request.param == "v3like" else _alt_chemistry)(tmp_path)


@pytest.fixture
def out_dir(tmp_path) -> Path:
    """An output directory outside the repository (tmp_path always is)."""
    return tmp_path / "runs" / "out"
