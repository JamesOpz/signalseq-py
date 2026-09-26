"""Where each element sits in the reads, and how to pull it out.

This module owns the ONE place where the config's 1-based, inclusive-start
coordinates are converted to Python's 0-based half-open slices:
`Element.from_mapping` does `start0 = start - 1`, and nothing else in the
package subtracts 1 from a coordinate.

Extraction goes through a locator object rather than a bare slice. Today the
only implementation is `FixedOffsetLocator` (constant offsets, which is what
Parse v3 and the original pipeline use). A chemistry whose barcodes float --
variable-length adapters, indels in a linker -- can add an anchor-based locator
that searches for a linker and returns an offset-corrected slice, without
touching the read scan or anything downstream of it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

READS = ("R1", "R2")


class ConfigError(ValueError):
    """Raised for any malformed or self-inconsistent config."""


# --------------------------------------------------------------------------
# Elements
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Element:
    """One extracted stretch of a read (ADT tag, UMI, or a cell-barcode round).

    `start0` is 0-based; the config's `start` is 1-based. This is the only
    class that performs that conversion.
    """

    name: str
    read: str
    start0: int
    length: int
    whitelist: str | None = None
    allow_overlap: bool = False
    obs_label: str | None = None

    @classmethod
    def from_mapping(cls, name: str, m: Mapping) -> "Element":
        if not isinstance(m, Mapping):
            raise ConfigError(f"read_structure.elements.{name}: expected a mapping, got {type(m).__name__}")
        for required in ("read", "start", "length"):
            if required not in m:
                raise ConfigError(f"read_structure.elements.{name}: missing required key '{required}'")

        read = str(m["read"]).upper()
        if read not in READS:
            raise ConfigError(
                f"read_structure.elements.{name}.read: expected one of {READS}, got {m['read']!r}"
            )

        start = m["start"]
        length = m["length"]
        if not isinstance(start, int) or isinstance(start, bool):
            raise ConfigError(f"read_structure.elements.{name}.start: expected an integer, got {start!r}")
        if not isinstance(length, int) or isinstance(length, bool):
            raise ConfigError(f"read_structure.elements.{name}.length: expected an integer, got {length!r}")
        if start < 1:
            raise ConfigError(
                f"read_structure.elements.{name}.start: coordinates are 1-based, so start must be >= 1, got {start}"
            )
        if length < 1:
            raise ConfigError(f"read_structure.elements.{name}.length: must be >= 1, got {length}")

        wl = m.get("whitelist")
        label = m.get("obs_label")
        return cls(
            name=name,
            read=read,
            start0=start - 1,          # <-- the single 1-based -> 0-based conversion
            length=length,
            whitelist=None if wl is None else str(wl),
            allow_overlap=bool(m.get("allow_overlap", False)),
            obs_label=None if label is None else str(label),
        )

    # -- derived -----------------------------------------------------------
    @property
    def label(self) -> str:
        """Prefix for this element's columns in `adata.obs`.

        Defaults to the element name. Parse v3 sets it explicitly so the
        columns keep the original pipeline's names (`RT_well`, `R2_well`,
        `R3_well`) even though the elements are named rt/bc2/bc3.
        """
        return self.obs_label or self.name

    @property
    def end0(self) -> int:
        """Exclusive 0-based end, i.e. the minimum read length that contains it."""
        return self.start0 + self.length

    @property
    def start1(self) -> int:
        """1-based start, for echoing coordinates back in the same units the user wrote."""
        return self.start0 + 1

    @property
    def slice(self) -> slice:
        return slice(self.start0, self.end0)

    def extract(self, seq: bytes) -> bytes:
        return seq[self.start0:self.end0]

    def describe(self) -> str:
        return f"{self.name}: {self.read} {self.start1}..{self.end0} (length {self.length})"


@dataclass(frozen=True)
class Linker:
    """A constant sequence between barcode blocks.

    The original Parse v3 pipeline skips linkers without inspecting them, so
    these ship with `enabled=False` and exist to (a) document the layout and
    (b) give an anchor-based locator something to search for later.
    """

    name: str
    read: str
    start0: int
    length: int
    sequence: bytes | None
    max_mismatches: int | None
    enabled: bool

    @classmethod
    def from_mapping(cls, m: Mapping) -> "Linker":
        name = str(m.get("name", "linker"))
        base = Element.from_mapping(f"<linker {name}>", m)
        seq = m.get("sequence")
        seq_b = None if seq is None else str(seq).upper().encode("ascii")
        if seq_b is not None and len(seq_b) != base.length:
            raise ConfigError(
                f"read_structure.linkers[{name}]: sequence is {len(seq_b)}bp "
                f"but length is declared as {base.length}"
            )
        mm = m.get("max_mismatches")
        enabled = bool(m.get("enabled", False))
        if enabled and seq_b is None:
            raise ConfigError(
                f"read_structure.linkers[{name}]: enabled: true requires a 'sequence' to compare against"
            )
        if enabled and mm is None:
            raise ConfigError(
                f"read_structure.linkers[{name}]: enabled: true requires 'max_mismatches' "
                f"(use 0 for an exact match)"
            )
        return cls(
            name=name, read=base.read, start0=base.start0, length=base.length,
            sequence=seq_b, max_mismatches=mm, enabled=enabled,
        )

    @property
    def end0(self) -> int:
        return self.start0 + self.length

    @property
    def start1(self) -> int:
        return self.start0 + 1

    def matches(self, seq: bytes) -> bool:
        """True if this linker is absent-but-disabled, or present within tolerance."""
        if not self.enabled:
            return True
        observed = seq[self.start0:self.end0]
        if len(observed) != self.length:
            return False
        mismatches = sum(a != b for a, b in zip(observed, self.sequence))
        return mismatches <= self.max_mismatches


# --------------------------------------------------------------------------
# Locators -- the future-proofing seam
# --------------------------------------------------------------------------

class Locator(Protocol):
    """Turns a read into the bytes for one element.

    Implementations must be cheap: this runs once per element per read.
    """

    def locate(self, seq: bytes) -> bytes | None:
        ...


@dataclass(frozen=True)
class FixedOffsetLocator:
    """Constant offsets -- the only behaviour the original pipeline has.

    Returns None when the read is too short to contain the element, so callers
    can count that separately from a whitelist miss.
    """

    element: Element

    def locate(self, seq: bytes) -> bytes | None:
        if len(seq) < self.element.end0:
            return None
        return seq[self.element.start0:self.element.end0]


# --------------------------------------------------------------------------
# The whole structure
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ReadStructure:
    """All elements plus the rule for combining barcode rounds into a cell ID."""

    elements: Mapping[str, Element]
    linkers: Sequence[Linker]
    cell_barcode_order: Sequence[str]
    name_template: str
    adt_element: str
    umi_element: str

    # -- construction ------------------------------------------------------
    @classmethod
    def from_mapping(cls, m: Mapping) -> "ReadStructure":
        coords = str(m.get("coordinates", "one_based"))
        if coords != "one_based":
            raise ConfigError(
                f"read_structure.coordinates: only 'one_based' is supported, got {coords!r}. "
                f"Starts are 1-based and inclusive."
            )

        raw_elements = m.get("elements") or {}
        if not raw_elements:
            raise ConfigError("read_structure.elements: at least one element is required")
        elements = {name: Element.from_mapping(name, d) for name, d in raw_elements.items()}

        linkers = tuple(Linker.from_mapping(d) for d in (m.get("linkers") or []))

        cb = m.get("cell_barcode") or {}
        order = tuple(str(x) for x in (cb.get("order") or []))
        if not order:
            raise ConfigError("read_structure.cell_barcode.order: required, e.g. [rt, bc2, bc3]")
        missing = [r for r in order if r not in elements]
        if missing:
            raise ConfigError(
                f"read_structure.cell_barcode.order names unknown element(s) {missing}; "
                f"defined elements are {sorted(elements)}"
            )
        for r in order:
            if elements[r].whitelist is None:
                raise ConfigError(
                    f"read_structure.elements.{r}: cell-barcode rounds need a 'whitelist', "
                    f"since they are matched against one"
                )

        adt_name = str(m.get("adt_element", "adt"))
        umi_name = str(m.get("umi_element", "umi"))
        for role, name in (("adt_element", adt_name), ("umi_element", umi_name)):
            if name not in elements:
                raise ConfigError(
                    f"read_structure.{role}: '{name}' is not a defined element "
                    f"(defined: {sorted(elements)})"
                )
        if elements[adt_name].whitelist is None:
            raise ConfigError(
                f"read_structure.elements.{adt_name}: the ADT element needs a 'whitelist' "
                f"pointing at the ADT panel"
            )

        template = str(cb.get("name_template") or _default_template(order))
        return cls(
            elements=elements, linkers=linkers, cell_barcode_order=order,
            name_template=template, adt_element=adt_name, umi_element=umi_name,
        )

    # -- accessors ---------------------------------------------------------
    def __getitem__(self, name: str) -> Element:
        return self.elements[name]

    @property
    def adt(self) -> Element:
        return self.elements[self.adt_element]

    @property
    def umi(self) -> Element:
        return self.elements[self.umi_element]

    @property
    def rounds(self) -> tuple[Element, ...]:
        """Cell-barcode round elements, in the configured combining order."""
        return tuple(self.elements[r] for r in self.cell_barcode_order)

    def locator(self, name: str) -> Locator:
        return FixedOffsetLocator(self.elements[name])

    def min_length(self, read: str) -> int:
        """Shortest read that contains every element (and enabled linker) on `read`.

        This is what the original pipeline hardcoded as `min_r2_len = RT_END`;
        deriving it means a re-ordered chemistry stays correct for free.
        """
        read = read.upper()
        ends = [e.end0 for e in self.elements.values() if e.read == read]
        ends += [l.end0 for l in self.linkers if l.read == read and l.enabled]
        return max(ends, default=0)

    def elements_on(self, read: str) -> tuple[Element, ...]:
        read = read.upper()
        return tuple(sorted(
            (e for e in self.elements.values() if e.read == read),
            key=lambda e: e.start0,
        ))

    def overlaps(self) -> list[tuple[Element, Element]]:
        """Pairs of elements on the same read whose spans intersect.

        Overlap is almost always a typo, but a chemistry could legitimately
        nest one element inside another, so an element can opt out with
        `allow_overlap: true` and be excluded here.
        """
        found = []
        for read in READS:
            on_read = self.elements_on(read)
            for i, a in enumerate(on_read):
                for b in on_read[i + 1:]:
                    if a.allow_overlap or b.allow_overlap:
                        continue
                    if a.start0 < b.end0 and b.start0 < a.end0:
                        found.append((a, b))
        return found

    def diagram(self) -> str:
        """Human-readable layout, echoed by `validate` and written to provenance."""
        lines = []
        for read in READS:
            on_read = self.elements_on(read)
            if not on_read:
                continue
            lines.append(f"{read} (needs >= {self.min_length(read)}bp):")
            cursor = 0
            for e in on_read:
                if e.start0 > cursor:
                    gap = e.start0 - cursor
                    label = _linker_label(self.linkers, read, cursor, gap)
                    lines.append(f"  {cursor + 1:>4}..{e.start0:<4} {gap:>3}bp  {label}")
                lines.append(f"  {e.start1:>4}..{e.end0:<4} {e.length:>3}bp  {e.name}")
                cursor = max(cursor, e.end0)
        return "\n".join(lines)


def _default_template(order: Sequence[str]) -> str:
    return "{sublibrary}_" + "_".join("{" + r + "}" for r in order)


def _linker_label(linkers: Sequence[Linker], read: str, start0: int, length: int) -> str:
    for l in linkers:
        if l.read == read and l.start0 == start0 and l.length == length:
            state = f"checked, <= {l.max_mismatches} mismatches" if l.enabled else "not checked"
            return f"{l.name} ({state})"
    return "(unused)"
