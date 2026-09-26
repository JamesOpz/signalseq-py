"""Coordinate handling: the 1-based -> 0-based conversion and layout checks."""

import pytest

from signalseq.readstructure import ConfigError, Element, FixedOffsetLocator, Linker, ReadStructure


def test_start_is_one_based_and_converted_once():
    el = Element.from_mapping("umi", {"read": "R2", "start": 1, "length": 10})
    assert el.start0 == 0
    assert el.end0 == 10
    assert el.start1 == 1
    assert el.slice == slice(0, 10)


def test_parse_v3_offsets_match_the_original_zero_based_values():
    """The ported Parse v3 coordinates must land on the original's offsets."""
    declared = {
        "adt": ({"read": "R1", "start": 1, "length": 15}, (0, 15)),
        "umi": ({"read": "R2", "start": 1, "length": 10}, (0, 10)),
        "bc3": ({"read": "R2", "start": 11, "length": 8}, (10, 18)),
        "bc2": ({"read": "R2", "start": 31, "length": 8}, (30, 38)),
        "rt": ({"read": "R2", "start": 51, "length": 8}, (50, 58)),
    }
    for name, (mapping, expected) in declared.items():
        el = Element.from_mapping(name, mapping)
        assert (el.start0, el.end0) == expected, name


def test_extract_pulls_the_declared_bases():
    el = Element.from_mapping("bc3", {"read": "R2", "start": 11, "length": 8})
    seq = bytes(range(ord("A"), ord("A") + 60))
    assert el.extract(seq) == seq[10:18]


def test_zero_or_negative_start_is_rejected():
    with pytest.raises(ConfigError, match="1-based"):
        Element.from_mapping("umi", {"read": "R2", "start": 0, "length": 10})


def test_unknown_read_is_rejected():
    with pytest.raises(ConfigError, match="R1"):
        Element.from_mapping("umi", {"read": "R3", "start": 1, "length": 10})


def test_non_integer_coordinates_are_rejected():
    with pytest.raises(ConfigError, match="integer"):
        Element.from_mapping("umi", {"read": "R2", "start": 1.5, "length": 10})


def _structure(elements, **extra):
    base = {
        "coordinates": "one_based",
        "elements": elements,
        "cell_barcode": {"order": ["rt"]},
    }
    base.update(extra)
    return ReadStructure.from_mapping(base)


def test_min_length_is_derived_not_hardcoded():
    """The original hardcoded min_r2_len = 58; it must fall out of the layout."""
    rs = _structure({
        "adt": {"read": "R1", "start": 1, "length": 15, "whitelist": "adt_panel"},
        "umi": {"read": "R2", "start": 1, "length": 10},
        "bc3": {"read": "R2", "start": 11, "length": 8, "whitelist": "lig"},
        "bc2": {"read": "R2", "start": 31, "length": 8, "whitelist": "lig"},
        "rt": {"read": "R2", "start": 51, "length": 8, "whitelist": "rt"},
    }, cell_barcode={"order": ["rt", "bc2", "bc3"]})
    assert rs.min_length("R2") == 58
    assert rs.min_length("R1") == 15


def test_overlapping_elements_are_detected():
    rs = _structure({
        "adt": {"read": "R1", "start": 1, "length": 15, "whitelist": "adt_panel"},
        "umi": {"read": "R2", "start": 1, "length": 10},
        "rt": {"read": "R2", "start": 8, "length": 8, "whitelist": "rt"},
    })
    overlaps = rs.overlaps()
    assert len(overlaps) == 1
    assert {a.name for a in overlaps[0]} == {"umi", "rt"}


def test_allow_overlap_opts_out():
    rs = _structure({
        "adt": {"read": "R1", "start": 1, "length": 15, "whitelist": "adt_panel"},
        "umi": {"read": "R2", "start": 1, "length": 10, "allow_overlap": True},
        "rt": {"read": "R2", "start": 8, "length": 8, "whitelist": "rt"},
    })
    assert rs.overlaps() == []


def test_barcode_round_without_whitelist_is_rejected():
    with pytest.raises(ConfigError, match="whitelist"):
        _structure({
            "adt": {"read": "R1", "start": 1, "length": 15, "whitelist": "adt_panel"},
            "umi": {"read": "R2", "start": 1, "length": 10},
            "rt": {"read": "R2", "start": 51, "length": 8},
        })


def test_cell_barcode_order_must_name_real_elements():
    with pytest.raises(ConfigError, match="unknown element"):
        _structure({
            "adt": {"read": "R1", "start": 1, "length": 15, "whitelist": "adt_panel"},
            "umi": {"read": "R2", "start": 1, "length": 10},
            "rt": {"read": "R2", "start": 51, "length": 8, "whitelist": "rt"},
        }, cell_barcode={"order": ["rt", "nope"]})


def test_only_one_based_coordinates_accepted():
    with pytest.raises(ConfigError, match="one_based"):
        ReadStructure.from_mapping({
            "coordinates": "zero_based",
            "elements": {"rt": {"read": "R2", "start": 1, "length": 8, "whitelist": "rt"}},
            "cell_barcode": {"order": ["rt"]},
        })


def test_obs_label_defaults_to_element_name():
    el = Element.from_mapping("rt", {"read": "R2", "start": 51, "length": 8})
    assert el.label == "rt"
    labelled = Element.from_mapping("rt", {"read": "R2", "start": 51, "length": 8, "obs_label": "RT"})
    assert labelled.label == "RT"


# -- locators --------------------------------------------------------------

def test_fixed_offset_locator_returns_none_when_read_too_short():
    el = Element.from_mapping("rt", {"read": "R2", "start": 51, "length": 8})
    locator = FixedOffsetLocator(el)
    assert locator.locate(b"A" * 57) is None
    assert locator.locate(b"A" * 58) == b"A" * 8


# -- linkers ---------------------------------------------------------------

def test_disabled_linker_always_matches():
    linker = Linker.from_mapping({
        "name": "l1", "read": "R2", "start": 19, "length": 12,
        "sequence": "ATGAGGGGTCAG", "enabled": False,
    })
    assert linker.matches(b"X" * 40) is True


def test_enabled_linker_respects_mismatch_budget():
    linker = Linker.from_mapping({
        "name": "l1", "read": "R2", "start": 1, "length": 4,
        "sequence": "ACGT", "max_mismatches": 1, "enabled": True,
    })
    assert linker.matches(b"ACGT") is True
    assert linker.matches(b"ACGA") is True     # 1 mismatch
    assert linker.matches(b"ACAA") is False    # 2 mismatches


def test_enabled_linker_needs_a_sequence_and_budget():
    with pytest.raises(ConfigError, match="sequence"):
        Linker.from_mapping({"name": "l", "read": "R2", "start": 1, "length": 4, "enabled": True})
    with pytest.raises(ConfigError, match="max_mismatches"):
        Linker.from_mapping({
            "name": "l", "read": "R2", "start": 1, "length": 4,
            "sequence": "ACGT", "enabled": True,
        })


def test_linker_sequence_length_must_agree():
    with pytest.raises(ConfigError, match="declared"):
        Linker.from_mapping({
            "name": "l", "read": "R2", "start": 1, "length": 8, "sequence": "ACGT",
        })
