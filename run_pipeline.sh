#!/usr/bin/env bash
# Full pipeline run: process each ADT sublibrary sequentially, then aggregate.
# Sequential by design -- see plan doc for the memory rationale (peak ~6-9GB
# per sublibrary job on a 16GB machine, dominated by the UMI-dedup np.unique
# sort step; running sublibraries concurrently would risk swap thrashing).

set -euo pipefail

PY="${PY:-/PATH/TO/CONDA/envs/parse_adt/bin/python}"  # PLACEHOLDER: set PY=... or edit
DATA_DIR="${DATA_DIR:-../data/2026_parse_v3_run_1/adt}"
OUT_DIR="${OUT_DIR:-outs}"
# First-pass permissive cell calling: fixed 250 UMI/cell floor (not the
# automatic knee) since downstream filtering will be done jointly with the
# RNA modality; min-ADTs floor dropped to 1 for the same reason.
MIN_UMI_PER_CELL="${MIN_UMI_PER_CELL:-250}"
MIN_ADTS_FLOOR="${MIN_ADTS_FLOOR:-1}"

cd "$(dirname "$0")"

# macOS ships bash 3.2 (no associative arrays), so use parallel arrays instead.
NAMES=(Sublibrary_3 Sublibrary_4 Sublibrary_6 Sublibrary_7)
SAMPLES=(S5 S6 S7 S8)

SUB_DIRS=()
for i in "${!NAMES[@]}"; do
  name="${NAMES[$i]}"
  sample="${SAMPLES[$i]}"
  r1="${DATA_DIR}/ADT_${name}_${sample}_R1_001.fastq.gz"
  r2="${DATA_DIR}/ADT_${name}_${sample}_R2_001.fastq.gz"
  out="${OUT_DIR}/${name}"

  echo "=== Processing ${name} ==="
  "$PY" process_sublibrary.py \
    --r1 "$r1" \
    --r2 "$r2" \
    --name "$name" \
    --output-dir "$out" \
    --min-umi-per-cell "$MIN_UMI_PER_CELL" \
    --min-adts-floor "$MIN_ADTS_FLOOR"

  SUB_DIRS+=(--sublibrary-dir "$out")
done

echo "=== Aggregating ${#NAMES[@]} sublibraries ==="
"$PY" aggregate_sublibraries.py "${SUB_DIRS[@]}" --output-dir "${OUT_DIR}/combined"

echo "=== Pipeline complete. See ${OUT_DIR}/combined/ ==="
