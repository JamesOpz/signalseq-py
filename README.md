# signalseq

Quantifies antibody-derived tags (ADTs) from SIGNAL-seq libraries built on Parse
Biosciences Evercode chemistry. You point it at a pair of FASTQs and a YAML
config; it matches the antibody tag on one read, resolves the combinatorial
cell barcode on the other against the Parse barcode whitelists (using the same
correction algorithm as the Parse RNA pipeline, so an ADT read and an RNA read
from the same cell resolve to the same cell), collapses UMIs, and writes
barcode- and cell-aligned AnnData matrices plus a full set of QC and composition
plots. Every read position, length, barcode whitelist, mismatch tolerance and
filter threshold comes from the config, so the same code runs Evercode WT v3 and
future custom chemistries where the tag, UMI and barcode rounds sit at different
offsets and lengths.

---

## Requirements

- **Python 3.10 or newer**
- No external tools are required. Two are used opportunistically:
  - **`libdeflate-gzip`** or **`gzip`** for FASTQ decompression, tried in the
    order given by `performance.gzip_binaries`. If neither is on `PATH`, Python's
    built-in `gzip` module is used. On macOS the system `gzip` is already fast;
    `libdeflate-gzip` is preferred when present but is not usually a
    meaningful speedup, because decompression runs in child processes in
    parallel with the read scan and is not the bottleneck.
- The CLR / PCA / PHATE stage needs the optional `analysis` extra (`muon`,
  `phate`). Without it, set `qc.analysis.enabled: false` to write counts and
  core QC only.

### A note on numerical reproducibility

Counts are **bit-exact** regardless of environment: barcode matching and UMI
collapse are integer operations in this package.

The downstream CLR / PCA / PHATE numbers are only bit-reproducible within a
**fixed dependency set**. This is not hypothetical — running the *same*
pipeline code three months apart, with identical inputs and bit-identical
counts and CLR values, produced `X_pca` differing by 3e-5 and a PHATE embedding
differing by ~0.01, purely from library upgrades in between. The PCA seed is now
pinned (`qc.analysis.pca_random_state`), which removes one source of this, but
`scipy`'s ARPACK implementation and `muon`'s CLR can still change between
versions. If you need to compare embeddings across machines or over time, pin
your environment. `provenance.json` records the versions actually used for every
run.

---

## Install

```bash
git clone https://github.com/JamesOpz/signalseq-py.git
cd signalseq-py

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -e '.[analysis]'       # or: pip install -e .   (counts + core QC only)
pip install -e '.[dev]'            # adds pytest
```

Check it:

```bash
signalseq --version
pytest                             # the test suite builds its own synthetic FASTQs
```

---

## Quickstart

```bash
# 1. Copy the example config somewhere OUTSIDE the repo and edit the paths.
cp configs/parse_v3.example.yaml ~/my_run.yaml

# 2. Edit run.output_dir and everything under inputs: to point at your files.

# 3. Check the config and all inputs. Reads nothing but the first ~1000 reads
#    per FASTQ; takes seconds.
signalseq adt validate --config ~/my_run.yaml

# 4. Run.
signalseq adt run-multi --config ~/my_run.yaml
```

Always run `validate` first. It catches a missing file, an off-by-one offset or a
barcode-length mismatch in seconds, rather than at minute 90 of a long run.

### The four commands

| Command | What it does |
|---|---|
| `signalseq adt validate --config C` | Checks the config and every input, then exits. Runs no analysis. Prints the read-structure diagram, the derived bit packing and the active settings. |
| `signalseq adt run --config C --sublibrary NAME` | Processes one sublibrary. `--sublibrary` may be omitted when the config lists exactly one. |
| `signalseq adt run-multi --config C` | Processes every sublibrary in the config, in listed order, then aggregates. Sequential by design: peak memory is dominated by the UMI-dedup sort, so running sublibraries concurrently risks swapping. |
| `signalseq adt aggregate --config C` | Aggregates sublibrary outputs that already exist. Use after a `run-multi --skip-aggregate`, or to re-aggregate a subset with `--sublibrary-dir`. |

Useful flags: `--debug-reads N` caps the scan at N read pairs for a smoke test;
`run-multi --continue-on-error` carries on if one sublibrary fails;
`validate --skip-read-check` skips opening the FASTQs.

---

## Outputs stay outside the repository

`run.output_dir` **must resolve outside the git checkout**, and config validation
refuses it otherwise:

```
run.output_dir resolves to /path/to/signalseq-py/outs/run1, which is inside the
signalseq repository (/path/to/signalseq-py). Outputs, intermediates, logs and
plots must live outside the repo so they can never be committed.
```

The reason is that outputs are both large and partly un-shareable. A single
sublibrary produces hundreds of megabytes of `.h5ad`, and those matrices, the
plots and the barcode-derived tables are all derived from sequencing data and
from licensed Parse barcode whitelists. Keeping them out of the tree means they
can never be committed by accident, so the repository stays shareable and small.
The same applies to the reference files themselves: the Parse barcode CSVs and
edit-distance dictionaries, the ADT panel table and the RT plate layout are named
by the config and are **not** stored in this repository.

A layout that works well:

```
~/analysis/
  signalseq-py/              <- this repo, code only
  reference/                 <- barcode whitelists, ADT panel, RT layout
  runs/
    2026_parse_v3_run_1/     <- run.output_dir
  configs/
    my_run.yaml
```

Relative paths inside a config resolve against **the config file's own
directory**, not the shell's working directory, so a config and its reference
files can be moved around together.

---

## Read structure

All coordinates are **1-based and inclusive of `start`**. The conversion to
Python's 0-based half-open slices happens in exactly one place
(`Element.from_mapping` in `readstructure.py`), so there is no second convention
to keep track of.

### Parse Evercode WT v3

R1 is 64 bp and R2 58 bp as sequenced; the structure uses R1[1..15] and all of R2.

```
R1  (64bp sequenced, 15bp used)
    ┌──────────────┐
    │   ADT tag    │ ................................................
    └──────────────┘
     1           15
     15bp, matched against the ADT panel (exact + 1 mismatch)

R2  (58bp sequenced, all 58bp used)
    ┌──────────┬────────┬────────────┬────────┬────────────┬────────┐
    │   UMI    │  bc3   │  linker    │  bc2   │  linker    │  bc1   │
    │          │  lig3  │            │  lig2  │            │  RT    │
    └──────────┴────────┴────────────┴────────┴────────────┴────────┘
     1      10  11   18  19       30  31   38  39       50  51   58
       10bp      8bp      12bp       8bp      12bp        8bp
     2 bits/base  96-well  ATGAGGGG   96-well  TCCAACCA    99 RT
     packed       ligation TCAG       ligation CCTC        barcodes
                  plate    not        plate    not
                           checked             checked

Cell barcode = rt * 96 * 96  +  bc2 * 96  +  bc3          (912,384 combinations)
obs_names    = {sublibrary}_{rt}_{bc2}_{bc3}
```

**Why the linkers are not checked.** The original pipeline skips them without
inspecting them, and that is the right default here: `linker_bc2_rt` matches
exactly in only ~76% of reads, because there is a common 1-mismatch variant
(`TCCAATCACCTC`). An exact check would discard roughly a quarter of all reads.
They are declared in the config so the layout is documented and so a future
anchor-based locator has something to search for; `enabled: false` is what
reproduces the original.

### Adapting to a different chemistry

Suppose a custom chemistry with: a **12 bp** ADT tag on **R2** starting at base
**5**; a **9 bp** UMI at the **end** of R2; and only **two** barcode rounds, on
**R1**, of **6 bp** and **10 bp**, drawn from two different plates. Nothing but
the config changes:

```yaml
inputs:
  whitelists:
    plate_a:                       # 6bp round
      csv: /path/to/plate_a.csv
      sequence_column: sequence
      well_column: well
      edit_distance_json: null     # null => exact matching only
    plate_b:                       # 10bp round
      csv: /path/to/plate_b.csv
      sequence_column: sequence
      well_column: well
      edit_distance_json: null

read_structure:
  coordinates: one_based
  adt_element: adt
  umi_element: umi
  elements:
    adt:  { read: R2, start:  5, length: 12, whitelist: adt_panel, obs_label: ADT }
    umi:  { read: R2, start: 21, length:  9, obs_label: UMI }
    cb1:  { read: R1, start:  3, length:  6, whitelist: plate_a, obs_label: CB1 }
    cb2:  { read: R1, start: 15, length: 10, whitelist: plate_b, obs_label: CB2 }
  linkers: []
  cell_barcode:
    order: [cb1, cb2]                          # most significant first
    name_template: "{sublibrary}_{cb1}_{cb2}"

matching:
  cell_barcode:
    max_edit_distance: 0           # no precomputed dictionaries available
```

Everything else follows automatically: the minimum acceptable read length per
read, the packing radix and bit widths, `obs` column names (`CB1_well`,
`CB2_sequence`, …), the number of plate heatmaps, and which reads get a length
guard. Run `signalseq adt validate` and it will print the layout it derived and
confirm the barcode lengths agree with the whitelist files.

Two constraints to be aware of:

- **Elements on the same read may not overlap** unless one sets
  `allow_overlap: true`. Overlap is nearly always a typo.
- **(cell space × panel size × UMI) must fit in 64 bits.** Validation reports the
  packing, e.g. `cell space 99 x 96 x 96 = 912,384 (20 bits), 14 ADTs (4 bits),
  10bp UMI (20 bits) = 44/64 bits`. Long UMIs with large whitelists are what
  push against this.

---

## Config reference

Any section may be omitted, in which case the defaults below apply — and those
defaults reproduce the pipeline this package was ported from. `read_structure`
and `inputs` have no useful defaults and are required.

### `run`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `name` | str | *required* | Label for the run; recorded in provenance. |
| `output_dir` | path | *required* | Where everything is written. Must resolve outside the repo. |
| `threads` | int | `-1` | `-1` = all cores. Only reaches PHATE's `n_jobs`; the read scan is single-threaded by design (the GIL serializes per-record work, so a second reader thread adds no throughput) and decompression runs in child processes. |
| `overwrite` | bool | `false` | When false, refuses to write into a non-empty `output_dir`. |

### `inputs`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `sublibraries[]` | list | *required* | One entry per sublibrary: `name`, `r1`, `r2`. Names must be unique. |
| `whitelists.<id>` | map | *required* | Named barcode whitelists, referenced from `read_structure.elements.*.whitelist`. |
| `whitelists.<id>.csv` | path | *required* | Whitelist table. |
| `whitelists.<id>.sequence_column` | str | `sequence` | Column holding barcode sequences. All must be the same length. |
| `whitelists.<id>.well_column` | str \| null | `well` | Column holding well IDs. Falls back to the sequence when null. |
| `whitelists.<id>.stype_column` | str \| null | `null` | Optional sample-type column; colours the reads-per-barcode plot. |
| `whitelists.<id>.edit_distance_json` | path \| null | `null` | Parse precomputed edit-distance dictionary. Null means exact matching only. |
| `adt_panel.csv` | path | *required* | Antibody panel table. |
| `adt_panel.barcode_column` | str | `Barcode` | Column holding tag sequences. |
| `adt_panel.name_column` | str | `Antigen` | Column that becomes `var_names`. Must be unique. |
| `rt_layout.csv` | path | *required* | Well → sample-name layout. |
| `rt_layout.well_column` | str | `Well_ID` | Well ID column. |
| `rt_layout.sample_column` | str | `Sample_name` | Sample-name column, parsed per `sample_annotation`. |

Declare **one whitelist entry per barcode round**, even when rounds share a
file — Parse v3 uses the same 96-well ligation plate for both ligation rounds, so
its `bc2_round` and `bc3_round` point at the same CSV and JSON. Repeated files
are parsed once and cached, so this costs nothing and means a chemistry with
distinct plates per round needs no code change.

### `read_structure`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `coordinates` | enum | `one_based` | Only accepted value. Starts are 1-based and inclusive. |
| `adt_element` | str | `adt` | Which element is the antibody tag. |
| `umi_element` | str | `umi` | Which element is the UMI. |
| `elements.<id>.read` | enum | *required* | `R1` or `R2`. |
| `elements.<id>.start` | int | *required* | 1-based first base. |
| `elements.<id>.length` | int | *required* | Length in bases. Must match the whitelist file's barcode length. |
| `elements.<id>.whitelist` | str \| null | `null` | Whitelist name, or `adt_panel` for the tag. Required for barcode rounds. |
| `elements.<id>.obs_label` | str \| null | element id | Prefix for `obs` columns (`<label>_well`, `<label>_sequence`). |
| `elements.<id>.allow_overlap` | bool | `false` | Permit overlapping this element with another. |
| `linkers[]` | list | `[]` | Constant sequences: `name`, `read`, `start`, `length`, `sequence`, `max_mismatches`, `enabled`. |
| `cell_barcode.order` | list | *required* | Barcode rounds, most significant first. Defines the packing radix and the `obs_names` order. |
| `cell_barcode.name_template` | str | `{sublibrary}_{round}_…` | Template for `obs_names`. May reference `{sublibrary}` and any round by id or label. |

### `matching`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `adt.max_mismatches` | int | `1` | Substitution budget for the tag dictionary. 0–2. |
| `adt.drop_ambiguous` | bool | `true` | Discard a mutant reachable from two different tags. |
| `cell_barcode.max_edit_distance` | int | `2` | Cap on distances read from the edit-distance dictionaries. Parse's production pipeline caps at 2. |
| `cell_barcode.census.offset_reads` | int | `3000000` | Where the census sample starts. |
| `cell_barcode.census.n_reads` | int | `1000000` | How many reads the census samples. |
| `cell_barcode.census.perfect_cell_threshold` | float | `0.92` | Cumulative read fraction defining the combo-count threshold. |
| `cell_barcode.census.min_count_threshold` | int | `5` | Floor for that threshold. |
| `cell_barcode.census.accept` | enum | `strictly_greater` | Whether a combo must exceed (`strictly_greater`) or merely reach (`greater_or_equal`) the threshold. |
| `umi.method` | enum | `directional` | `directional` (UMI-tools style) or `exact`. |
| `umi.max_mismatches` | int | `1` | UMI mismatch budget. 0 behaves like `exact`. |
| `umi.directional_count_ratio` | num | `2` | Absorb UMI *j* into seed *i* when `count_j <= ratio * count_i - 1`. |

**How barcode correction works.** A first pass over a sample of reads counts how
often each combination of *exactly matching* barcode rounds occurs, and derives a
count threshold from that distribution. Then, per read: if every round matches
exactly, accept immediately. Otherwise gather each round's candidates at the
smallest edit distance that hits, and accept only if **exactly one** combination
of candidates clears the threshold — two or more is ambiguous and the read is
dropped. Set `census.n_reads` and `offset_reads` so the window lies inside your
data; if the sample lands past the end of the file the census is empty and only
exact matches will ever be accepted.

### `filtering`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `min_umi_per_cell` | num \| null | `null` | Fixed UMI floor. Null uses the detected knee. The knee is computed and plotted either way. |
| `max_umi_per_cell` | num \| null | `null` | Optional upper cap, for excluding likely multiplets. |
| `min_adts_detected` | int | `2` | Minimum distinct ADTs a barcode must show. |
| `knee.exclude_top_n` | int | `3` | Top ranks skipped when fitting the log-log chord, where the curve is steep and noisy. |

Cell calling is the UMI threshold **and** the min-ADTs floor, combined. The knee
is found by the maximum-perpendicular-distance-to-chord method on the log-log
barcode-rank curve.

### `qc`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `saturation.enabled` | bool | `true` | Compute the downsampling curve. |
| `saturation.fractions` | list | `0.1 … 1.0` | Depths to subsample to. Ascending, each in (0, 1]. |
| `saturation.seed` | int | `0` | RNG seed, so the curve is reproducible. |
| `analysis.enabled` | bool | `true` | Run CLR + PCA + PHATE. False skips it and drops the `muon`/`phate` requirement. |
| `analysis.n_pcs` | int | `10` | Principal components; capped at `n_vars - 1`. |
| `analysis.pca_svd_solver` | str | `arpack` | Passed to scanpy. |
| `analysis.pca_random_state` | int | `0` | Seed for the PCA solver, pinned explicitly. ARPACK's result depends on its starting vector, so leaving this to the library default lets a `scanpy`/`scipy` upgrade shift the PCA and, through it, PHATE. `0` matches scanpy's current default, so setting it changes nothing today. |
| `analysis.phate.*` | map | `knn 15, decay 40, t auto, random_state 42` | Passed to PHATE, plus `n_jobs` from `run.threads`. |
| `plots.enabled` | bool | `true` | Write figures. |
| `plots.dpi` | int | `150` | Figure resolution. |
| `plots.style` | str | `seaborn-v0_8-whitegrid` | Matplotlib style; ignored if unavailable. |
| `plots.plate_rows` / `plate_cols` | list | `A–H` / `1–12` | Plate geometry for the well heatmaps. |
| `plots.grid_columns` | int | `4` | Columns in the per-ADT panel grids. |
| `plots.color_percentiles` | list | `[5, 95]` | Colour limits for per-ADT panels, as percentiles. |
| `plots.well_heatmap_cmaps` | list | `[Blues, Greens, Oranges]` | One colormap per round, cycled. |
| `plots.stype_colors` | map | `{}` | Colour per sample-type value in the reads-per-barcode plot. |
| `plots.treatment_order` | list | `[]` | Condition ordering by treatment. |
| `plots.culture_colors` | map | `{}` | Bar colours per culture type. |
| `plots.unassigned_label` | str | `Unassigned` | Excluded from the culture legend. |

### `sample_annotation`

Turns the layout's sample-name column into `obs` annotation. Cosmetic: it never
affects counts.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `join_on_round` | str | first round | Which barcode round carries sample identity; joined on `<label>_well`. |
| `culture_map` | map | `{}` | Maps a raw first token to a display name. |
| `treatment.control_token` | str | `DMSO` | Token in position 2 marking a vehicle control. |
| `treatment.control_label` | str | = token | Label for controls. |
| `treatment.treated_label` | str | `Treated` | Label for treated wells. |
| `unassigned_label` | str | `Unassigned` | Used for wells absent from the layout. |
| `condition_template` | str | `{culture_type}_{timepoint_hrs}hr_{treatment}` | How `condition` is composed. |

Sample names are parsed as three underscore-separated tokens:
`{culture}_{timepoint}_{anything}` for treated wells, or
`{culture}_{control_token}_{timepoint}` for controls. Anything else gets
treatment `Unknown` and a missing timepoint, rather than failing the run.

### `outputs`, `performance`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `outputs.raw_anndata` | bool | `true` | Write `<name>_raw.h5ad`. |
| `outputs.filtered_anndata` | bool | `true` | Write `<name>_filtered.h5ad`. |
| `outputs.saturation_table` | bool | `true` | Write `<name>_saturation.csv`. |
| `outputs.qc_stats` | bool | `true` | Write `<name>_qc_stats.json`. |
| `outputs.figures` | bool | `true` | Write the `figures/` PNGs. |
| `outputs.aggregate.combined_anndata` | bool | `true` | Write `combined/combined_filtered.h5ad`. |
| `outputs.aggregate.combined_qc_stats` | bool | `true` | Write `combined/combined_qc_stats.json`. |
| `outputs.aggregate.copy_best_saturation` | bool | `true` | Copy the best sublibrary's saturation curve into `combined/`. |
| `performance.chunk_rows` | int | `25000000` | Rows per preallocated numpy buffer in the read scan. |
| `performance.gzip_binaries` | list | `[libdeflate-gzip, gzip]` | Decompressors to try, in order. |

---

## Outputs

Per sublibrary, under `<output_dir>/<sublibrary>/`:

| File | Contents |
|---|---|
| `<name>_raw.h5ad` | **Every candidate barcode** — before cell calling. `X` and `layers['counts']` are UMI-collapsed molecule counts, cells × ADTs. |
| `<name>_filtered.h5ad` | **Called cells only.** `layers['counts']` keeps raw counts; `X` and `layers['clr']` hold CLR values, with `obsm['X_pca']` and `obsm['X_phate']`, when analysis is enabled. |
| `<name>_saturation.csv` | Downsampling curve: reads used, mean reads/cell, median UMIs/cell, median ADTs/cell, sequencing saturation. |
| `<name>_qc_stats.json` | The read funnel, matrix summary, knee, per-round well counts, cells per condition. |
| `resolved_config.yaml` | The fully merged config actually used, defaults included. |
| `provenance.json` | Package version, git commit and dirty flag, dependency versions, Python and platform, the derived read structure, timestamp. |

`obs` carries `<label>_well` and `<label>_sequence` per barcode round,
`cell_id_packed`, `total_counts`, `n_adts_detected`, and the sample annotation
(`sample_name`, `culture_type`, `timepoint_hrs`, `treatment`, `condition`).
`var` carries the panel columns plus `total_counts`, `n_cells_detected`,
`detection_rate`.

Aggregated output lands in `<output_dir>/combined/`, with CLR/PCA/PHATE rerun
jointly on the pooled raw counts so cells from different sublibraries are
directly comparable.

### The figures

| Figure | What it shows |
|---|---|
| `funnel` | Reads surviving each stage: total → valid ADT tag → fully assigned, with percentages. The first place to look when yields are low. |
| `reads_per_<round>` | Reads per barcode in that round, coloured by sample type. Reveals dead or over-represented wells. |
| `knee` | Log-log barcode-rank vs UMI curve, with the detected knee and the threshold actually applied. |
| `qc_prefilter` / `qc_postfilter` | Six panels: UMI distribution (with thresholds), ADTs detected, UMI vs ADTs, total counts per ADT, detection rate per ADT, per-ADT count distributions. |
| `wells_prefilter` / `wells_postfilter` | Plate heatmaps of cells per well, one per barcode round. Spatial artefacts show up here. |
| `cells_per_sample` | Cells per experimental condition, coloured by culture type. |
| `qc_by_sample` | Boxplots of UMIs and ADTs detected per cell, by condition — checks no condition is an outlier. |
| `saturation_curve` | Complexity vs depth, and sequencing saturation vs depth. Flattening means more reads would add little. |
| `clr_normalized` | Per-ADT CLR distributions. Bimodality indicates a discriminating marker. |
| `pca_variance`, `pca_scatter`, `pca_per_adt` | Variance explained; PC1/PC2 coloured by QC; PC1/PC2 coloured by each ADT. |
| `phate_qc`, `phate_per_adt` | PHATE embedding coloured by QC metrics, and by each ADT. |
| `violin_all_adts` | CLR expression per ADT, side by side. |
| `adt_correlation` | ADT–ADT Pearson correlation on CLR values, annotated. |
| `adt_heatmap_by_sample` | Mean CLR expression per condition × ADT. |
| `umi_by_sublibrary` | *(combined only)* UMI distributions overlaid per sublibrary, to spot an outlier sublibrary. |

Note that `wells_*` heatmaps assume a plate geometry from
`qc.plots.plate_rows`/`plate_cols`; a whitelist larger than that grid will have
entries that do not appear on the heatmap, though they are still counted
everywhere else.

---

## Troubleshooting

**`run.output_dir resolves to … which is inside the signalseq repository`**
Point `output_dir` somewhere outside the checkout. See *Outputs stay outside the
repository*.

**`run.output_dir is not empty`**
A previous run is there. Use a fresh directory, or set `run.overwrite: true`.

**`inputs.whitelists.X (file.csv) holds 8bp barcodes but read_structure.elements.Y.length is 7`**
The declared length disagrees with the file. Fix whichever is wrong — usually the
config. The same check applies to `adt_panel`.

**`the read structure needs 58bp but 1000/1000 of the first reads are shorter (min 45bp). Element(s) running past the end: rt: R2 51..58`**
An element runs off the end of the reads. Either the offsets are wrong for this
chemistry, or the FASTQs are trimmed. Remember `start` is **1-based**: the first
base is 1, not 0.

**`read_structure: elements overlap on the same read`**
Two elements' spans intersect — nearly always a typo in a `start` or `length`. If
genuinely intended, set `allow_overlap: true` on one.

**`This chemistry needs 68 bits to pack (cell, ADT, UMI) into one integer, over the 64-bit limit`**
The combined cell space, panel size and UMI length don't fit. Shorten the UMI
element or reduce whitelist sizes.

**`qc.analysis.enabled is true, which needs the optional analysis extras`**
Run `pip install -e '.[analysis]'`, or set `qc.analysis.enabled: false`.

**Everything validates but almost no reads are assigned.**
Check the `funnel` figure. If `valid_adt` is high but `valid_all` is low, the cell
barcode is the problem — confirm the round offsets and that
`matching.cell_barcode.census` samples a window that actually exists in your file
(with `offset_reads: 3000000`, a file with fewer than ~4M reads gives an empty
census, so only exact barcode matches are ever accepted). If `valid_adt` itself is
low, check the ADT offsets and that the panel file matches the antibodies used.

**`--sublibrary is required when the config lists more than one`**
Either name one with `--sublibrary`, or use `run-multi` to process them all.

**`N duplicate cell barcodes across sublibraries after prefixing`**
`cell_barcode.name_template` must include `{sublibrary}`, or barcodes from
different sublibraries collide during aggregation.

---

## Differences from the original pipeline

This package was ported from an earlier script-driven pipeline and verified
bit-identical to it on a full sublibrary (224M read pairs): the raw and filtered
count matrices, the CLR matrix, the complete read funnel, the knee, and the
saturation curve all matched exactly.

Two defects were found during that port, and **both are now fixed
unconditionally** — there is no flag to restore the old behaviour. Equivalence
with the original was demonstrated first, at full scale, so the fixes are known
to change only what they claim to. If you need to reproduce a pre-fix result,
check out commit `d225860` or earlier, which still carried the opt-out.

1. **`valid_cell_barcode` double-counted `valid_all`.** The original applied the
   ADT-tag gate before barcode correction, so correction never saw a read whose
   tag failed, and the two counters were incremented at the same point — making
   them always equal and `valid_cell_barcode` meaningless. Fixed, it reports what
   its name says. **Affects `processing_stats.valid_cell_barcode` only**; no
   matrix, cell call or other counter moves.

   Measured on a 224,465,769-read sublibrary: `valid_cell_barcode` reports
   163,753,195 where the original reported 157,029,873 (**+6,723,322, +4.28%**) —
   the reads whose cell barcode resolves but whose ADT tag did not match. Every
   other funnel counter, the knee, the cells called, the per-condition counts and
   both count matrices are unchanged. Because correction now runs on all reads
   rather than the ~96% passing the ADT gate, the scan costs about **3% more
   time** (802s vs 778s) and no extra pass over the FASTQ.

   **Consequence to be aware of:** `valid_cell_barcode` is no longer equal to
   `valid_all`, and `overall_valid_rate` is still computed from `valid_all`, so
   that rate is unchanged. Any downstream script that assumed the two counters
   were interchangeable needs checking.

2. **An unparseable timepoint rendered as the literal `<NA>`.** The original built
   `condition` by concatenating a pandas `Int64` cast to `str`, so a missing
   timepoint produced strings like `MonoCulture_<NA>hr_SN-38`. Fixed, it uses
   `sample_annotation.unassigned_label`. **Affects `obs.condition` and anything
   grouped by it**; counts are untouched. This is latent for a layout whose rows
   all parse cleanly — on the sublibrary above it changed nothing, since every
   layout row parsed. It is insurance, not a correction to existing results.

A third change is not a defect fix but guards against a real failure seen during
the port: `qc.analysis.pca_random_state` is now passed explicitly instead of
relying on scanpy's default. It defaults to `0`, which is scanpy's current
default, so it changes nothing today.

The package also writes a reads-per-barcode array for **every** declared barcode
round (`uns['reads_per_<round>']`), where the original kept only the first, and
generates `uns['barcode_structure']` from the read structure rather than
hand-writing it, so it cannot drift from the actual offsets.

---

## Development

```bash
pip install -e '.[dev]'
pytest                                   # synthetic FASTQs, no real data needed
```

Tests build their own FASTQs in-test against two deliberately different read
structures, so extraction is exercised under more than one chemistry rather than
only the one this package was ported from.

To check a run against a reference run:

```bash
python scripts/compare_outputs.py --a /path/to/run_a --b /path/to/run_b
```

It compares barcode and feature sets, requires count matrices to be **exactly**
identical after aligning row and column order, compares `obs`/`var` within a
tight float tolerance, checks the read-assignment funnel, and compares the
numbers behind the figures rather than the PNG bytes (which are not
reproducible). PCA and PHATE components are compared up to a per-component sign
flip, which carries no meaning.

## License

See [LICENSE](LICENSE).
