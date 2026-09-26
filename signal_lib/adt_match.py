"""ADT antibody-tag (R1, 15bp) matching against the 14-plex panel.

Custom to this CITE-seq-style protocol -- there is no RNA-side equivalent to
match against, so this keeps the pilot notebook's simple exact + 1bp-Hamming-
mutant dictionary (ambiguous mutants dropped), just built once up front.
"""

from . import config


def match_adt(tag: bytes, adt_whitelist) -> int | None:
    """O(1) dict lookup. Returns the ADT index, or None if unmatched."""
    return adt_whitelist.correction_dict.get(tag)
