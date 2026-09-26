"""Fast paired-FASTQ streaming.

Decompression is offloaded to a child process per file (libdeflate-gzip if
available, else gzip, else Python's gzip module), so the two decompressors run on
separate cores while the single Python thread does the barcode work. R1 and R2
are read alternately from one thread to keep pairing in lockstep: a second
Python thread reading independently would not add throughput, because the GIL
still serializes the per-record work, but would add synchronization complexity.

Which binaries to try is config-driven (`performance.gzip_binaries`).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Iterator, Sequence

DEFAULT_GZIP_BINARIES = ("libdeflate-gzip", "gzip")

# A decompressor killed by SIGPIPE when we stop reading early is expected, not
# an error: -13 on macOS/Linux when the consumer closes the pipe first.
_SIGPIPE_RETURNCODE = -13


def _decompressor_stream(path: str | Path, gzip_binaries: Sequence[str] = DEFAULT_GZIP_BINARIES):
    """Return (readable binary stream, child process or None) for a FASTQ file."""
    path = str(path)
    if not path.endswith(".gz"):
        return open(path, "rb"), None

    for binary in gzip_binaries:
        exe = shutil.which(binary)
        if exe:
            proc = subprocess.Popen(
                [exe, "-dc", path],
                stdout=subprocess.PIPE,
                bufsize=1 << 20,
            )
            return proc.stdout, proc

    import gzip
    return gzip.open(path, "rb"), None


class FastqStream:
    """A line-oriented FASTQ reader that also cleans up its decompressor."""

    def __init__(self, path, gzip_binaries: Sequence[str] = DEFAULT_GZIP_BINARIES):
        self.path = path
        self.stream, self.proc = _decompressor_stream(path, gzip_binaries)

    def readline(self):
        return self.stream.readline()

    def close(self):
        self.stream.close()
        if self.proc is not None:
            ret = self.proc.wait()
            if ret not in (0, None, _SIGPIPE_RETURNCODE):
                raise IOError(f"decompressor for {self.path} exited with code {ret}")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def iter_sequences(path, gzip_binaries: Sequence[str] = DEFAULT_GZIP_BINARIES,
                   offset_reads: int = 0, max_reads: int | None = None) -> Iterator[bytes]:
    """Yield the sequence line of each record from one FASTQ."""
    stream = FastqStream(path, gzip_binaries)
    try:
        readline = stream.readline
        for _ in range(offset_reads):
            for _ in range(4):
                if not readline():
                    return
        yielded = 0
        while max_reads is None or yielded < max_reads:
            readline()                          # @header
            seq = readline().rstrip(b"\n")
            readline()                          # +
            readline()                          # quality
            if not seq:
                return
            yield seq
            yielded += 1
    finally:
        stream.close()


def paired_fastq_records(r1_path, r2_path,
                         gzip_binaries: Sequence[str] = DEFAULT_GZIP_BINARIES,
                         offset_reads: int = 0,
                         max_reads: int | None = None) -> Iterator[tuple[bytes, bytes]]:
    """Yield (r1_seq, r2_seq) per read pair, sequence lines only.

    `offset_reads` skips that many records from the start (used by the
    barcode-correction census); `max_reads` caps how many are yielded after it.
    """
    r1 = FastqStream(r1_path, gzip_binaries)
    r2 = FastqStream(r2_path, gzip_binaries)
    try:
        r1_readline = r1.readline
        r2_readline = r2.readline

        for _ in range(offset_reads):
            for _ in range(4):
                if not r1_readline() or not r2_readline():
                    return

        yielded = 0
        while max_reads is None or yielded < max_reads:
            r1_readline()                        # @header
            r1_seq = r1_readline().rstrip(b"\n")
            r1_readline()                        # +
            r1_readline()                        # quality

            r2_readline()                        # @header
            r2_seq = r2_readline().rstrip(b"\n")
            r2_readline()                        # +
            r2_readline()                        # quality

            if not r1_seq or not r2_seq:
                return

            yield r1_seq, r2_seq
            yielded += 1
    finally:
        r1.close()
        r2.close()


def peek_read_lengths(path, n_reads: int = 1000,
                      gzip_binaries: Sequence[str] = DEFAULT_GZIP_BINARIES) -> list[int]:
    """Sequence lengths of the first `n_reads` records, for config validation."""
    return [len(seq) for seq in iter_sequences(path, gzip_binaries, max_reads=n_reads)]
