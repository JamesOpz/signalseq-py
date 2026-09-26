"""Fast paired-FASTQ streaming.

Decompression is offloaded to a `libdeflate-gzip` (or plain `gzip`) child
process per file, so the two decompressors run on separate cores while the
single Python thread does the barcode-matching work -- pairing order is kept
in lockstep by reading R1 and R2 alternately from one thread; a second Python
thread reading independently would not add throughput (CPython's GIL still
serializes the per-record work) but would add synchronization complexity.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from . import config


def _decompressor_stream(path: str | Path):
    """Return a readable binary stream for a (possibly gzipped) FASTQ file.
    Prefers `libdeflate-gzip -dc` (faster than Python's gzip module), falls
    back to plain `gzip -dc`, then to Python's built-in gzip if neither
    binary is on PATH. Returns (stream, proc_or_None)."""
    path = str(path)
    if not path.endswith(".gz"):
        return open(path, "rb"), None

    for binary in (config.LIBDEFLATE_GZIP_BIN, "gzip"):
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


class _FastqStream:
    """Thin wrapper closing both the pipe and its owning subprocess."""

    def __init__(self, path):
        self.stream, self.proc = _decompressor_stream(path)

    def readline(self):
        return self.stream.readline()

    def close(self):
        self.stream.close()
        if self.proc is not None:
            ret = self.proc.wait()
            if ret not in (0, None, -13):  # -13 = SIGPIPE from early consumer stop, harmless
                raise IOError(f"decompressor exited with code {ret}")


def paired_fastq_records(r1_path, r2_path, offset_reads: int = 0, max_reads: int | None = None):
    """Yield (r1_seq: bytes, r2_seq: bytes) for each read pair, seq lines only
    (trailing newline stripped). `offset_reads` skips that many records from
    the start (used for the barcode-correction census sample); `max_reads`
    caps the number of records yielded after the offset.
    """
    r1 = _FastqStream(r1_path)
    r2 = _FastqStream(r2_path)
    try:
        r1_readline = r1.readline
        r2_readline = r2.readline

        for _ in range(offset_reads):
            for _ in range(4):
                if not r1_readline() or not r2_readline():
                    return

        yielded = 0
        while max_reads is None or yielded < max_reads:
            r1_readline()  # @header
            r1_seq = r1_readline().rstrip(b"\n")
            r1_readline()  # +
            r1_readline()  # quality

            r2_readline()  # @header
            r2_seq = r2_readline().rstrip(b"\n")
            r2_readline()  # +
            r2_readline()  # quality

            if not r1_seq or not r2_seq:
                return

            yield r1_seq, r2_seq
            yielded += 1
    finally:
        r1.close()
        r2.close()
