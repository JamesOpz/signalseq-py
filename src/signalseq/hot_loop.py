"""The per-read scan: FASTQ pair -> flat numpy arrays of (cell_id, adt_idx, umi).

One row per read that passes ADT-tag matching and cell-barcode correction. No
per-read string concatenation and no nested-dict UMI tracking: cell identity is
a packed integer and hits accumulate into preallocated numpy buffers, flushed
into a chunk list and concatenated once at the end.

Performance note: this loop slices the reads directly rather than going through
`readstructure.Locator`. That is deliberate -- at hundreds of millions of reads
an extra bound-method call per element per read is not free, and fixed offsets
need no logic. A chemistry with floating offsets should route through the
Locator seam (see readstructure.FixedOffsetLocator) and accept that cost;
correctness of everything downstream is unaffected either way.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .bc_correct import BarcodeCorrector
from .encoding import Encoder
from .fastq_io import paired_fastq_records
from .readstructure import ReadStructure
from .whitelists import AdtPanel


def guarded_reads(read_structure: ReadStructure) -> tuple[str, ...]:
    """Reads that need an explicit length guard.

    A barcode round or the UMI silently produces garbage from a truncated slice,
    so those reads are checked. The ADT read is deliberately NOT checked: a short
    tag simply misses the panel dictionary and the read is dropped, which is what
    the original pipeline does, and adding a guard would move reads from
    "unmatched ADT" into a too-short counter and change the funnel.
    """
    reads = [el.read for el in read_structure.rounds]
    if read_structure.umi.read not in reads:
        reads.append(read_structure.umi.read)
    # Preserve R1-before-R2 ordering for stable key order.
    return tuple(r for r in ("R1", "R2") if r in reads)


def new_stats(read_structure: ReadStructure) -> dict:
    """The funnel counters, in the order they are populated.

    Key names are the original pipeline's, kept verbatim so `*_qc_stats.json`
    stays comparable across the port:

      <read>_too_short  that read was shorter than the layout needs. For Parse
                        v3 this is the single key `r2_too_short`, since the UMI
                        and all three rounds sit on R2.
      valid_adt         the ADT tag matched the panel
      valid_cell_barcode / valid_all
                        see the note in scan_sublibrary -- in the ported
                        behaviour these are necessarily equal
      bc_exact_triple   every barcode round matched at edit distance 0.
                        "triple" is historical: it counts all rounds, however
                        many the chemistry declares.
    """
    stats = {"total_reads": 0}
    for read in guarded_reads(read_structure):
        stats[f"{read.lower()}_too_short"] = 0
    stats.update({
        "valid_adt": 0,
        "valid_cell_barcode": 0,
        "valid_all": 0,
        "bc_exact_triple": 0,
    })
    return stats


def _new_chunk(rows: int, encoder: Encoder):
    return (
        np.empty(rows, dtype=encoder.cell_dtype),
        np.empty(rows, dtype=encoder.adt_dtype),
        np.empty(rows, dtype=encoder.umi_dtype),
    )


def scan_sublibrary(r1_path, r2_path,
                    read_structure: ReadStructure,
                    panel: AdtPanel,
                    whitelists: Mapping,
                    corrector: BarcodeCorrector,
                    encoder: Encoder,
                    chunk_rows: int,
                    gzip_binaries: Sequence[str],
                    fix_valid_cell_barcode_counter: bool = False,
                    max_reads: int | None = None):
    """Stream one sublibrary's FASTQ pair.

    Returns (cell_id, adt_idx, umi_packed, stats, reads_per_round) where the
    three arrays are parallel with one row per valid read, and reads_per_round
    maps each barcode round's name to an int64 array of post-correction read
    counts per whitelist entry.
    """
    rounds = read_structure.rounds
    barcode_read = rounds[0].read
    stats = new_stats(read_structure)
    guards = [
        (read, read_structure.min_length(read), f"{read.lower()}_too_short")
        for read in guarded_reads(read_structure)
    ]

    reads_per_round = {
        el.name: np.zeros(whitelists[el.name].n, dtype=np.int64) for el in rounds
    }
    round_counters = [reads_per_round[el.name] for el in rounds]

    # Hoist everything the inner loop touches.
    adt_el = read_structure.adt
    adt_start, adt_end = adt_el.start0, adt_el.end0
    umi_el = read_structure.umi
    umi_start, umi_end = umi_el.start0, umi_el.end0
    adt_lookup = panel.correction_dict.get
    correct = corrector.correct
    pack_umi = encoder.pack_umi
    multipliers = encoder.radix_multipliers
    umi_read = umi_el.read

    active_linkers = [l for l in read_structure.linkers if l.enabled]
    if active_linkers:
        stats["linker_mismatch"] = 0

    chunks = []
    cur_cell, cur_adt, cur_umi = _new_chunk(chunk_rows, encoder)
    ptr = 0

    for r1_seq, r2_seq in paired_fastq_records(
        r1_path, r2_path, gzip_binaries=gzip_binaries, max_reads=max_reads
    ):
        stats["total_reads"] += 1

        barcode_seq = r2_seq if barcode_read == "R2" else r1_seq
        adt_seq = r1_seq if adt_el.read == "R1" else r2_seq
        umi_seq = r1_seq if umi_read == "R1" else r2_seq

        too_short = False
        for read, needed, key in guards:
            if len(r1_seq if read == "R1" else r2_seq) < needed:
                stats[key] += 1
                too_short = True
                break
        if too_short:
            continue

        # A read failing a linker check is malformed, so that gate precedes
        # everything else. (No effect on a chemistry with no enabled linkers,
        # which includes Parse v3.)
        if active_linkers:
            seqs = {"R1": r1_seq, "R2": r2_seq}
            if not all(l.matches(seqs[l.read]) for l in active_linkers):
                stats["linker_mismatch"] += 1
                continue

        # The two orderings below differ ONLY in when barcode correction runs
        # relative to the ADT-tag gate, and therefore in what
        # `valid_cell_barcode` can observe.
        #
        # Original: the ADT gate comes first, so correction never sees a read
        # whose tag failed, and valid_cell_barcode is incremented at the same
        # point as valid_all -- making the two necessarily equal and the former
        # meaningless.
        #
        # Fixed: correction runs first, so valid_cell_barcode counts every read
        # whose barcode rounds resolve, independent of its ADT tag, which is
        # what the name says. Correction is the expensive step, but the ADT gate
        # only rejects a few percent of reads, so running it on all of them
        # rather than on the survivors costs a few percent -- far less than the
        # second FASTQ pass this used to take.
        if fix_valid_cell_barcode_counter:
            combo, is_exact = correct(barcode_seq)
            if combo is not None:
                stats["valid_cell_barcode"] += 1

            adt_idx = adt_lookup(adt_seq[adt_start:adt_end])
            if adt_idx is None:
                continue
            stats["valid_adt"] += 1

            if combo is None:
                continue
        else:
            adt_idx = adt_lookup(adt_seq[adt_start:adt_end])
            if adt_idx is None:
                continue
            stats["valid_adt"] += 1

            combo, is_exact = correct(barcode_seq)
            if combo is None:
                continue
            stats["valid_cell_barcode"] += 1

        if is_exact:
            stats["bc_exact_triple"] += 1
        stats["valid_all"] += 1

        cell_total = 0
        for idx, mult, counter in zip(combo, multipliers, round_counters):
            cell_total += idx * mult
            counter[idx] += 1

        if ptr == chunk_rows:
            chunks.append((cur_cell, cur_adt, cur_umi))
            cur_cell, cur_adt, cur_umi = _new_chunk(chunk_rows, encoder)
            ptr = 0

        cur_cell[ptr] = cell_total
        cur_adt[ptr] = adt_idx
        cur_umi[ptr] = pack_umi(umi_seq[umi_start:umi_end])
        ptr += 1

    chunks.append((cur_cell[:ptr], cur_adt[:ptr], cur_umi[:ptr]))

    if len(chunks) > 1:
        cell_id_arr = np.concatenate([c[0] for c in chunks])
        adt_idx_arr = np.concatenate([c[1] for c in chunks])
        umi_arr = np.concatenate([c[2] for c in chunks])
    else:
        cell_id_arr, adt_idx_arr, umi_arr = chunks[0]

    return cell_id_arr, adt_idx_arr, umi_arr, stats, reads_per_round
