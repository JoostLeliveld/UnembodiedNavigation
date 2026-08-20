from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import json

import pytest


REPO = Path(__file__).resolve().parents[2]
STUDY = REPO / "experiments/reconfiguration_holdout"


def _load(name: str, path: Path):
    sys.path.insert(0, str(STUDY))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


L2 = _load("choose_second_layout_test", STUDY / "choose_second_layout.py")
CONTRACT = _load("freeze_l2_capture_contract_test", STUDY / "freeze_l2_capture_contract.py")


def test_eligibility_enumeration_is_frozen_and_complete():
    segments = L2.LAYOUT1.rack_segments(L2.LAYOUT1.BASE_WORLD)
    spec = L2.eligibility_spec(segments)

    assert spec["candidate_count"] == 27
    assert spec["eligible_layout_count"] == 216_000
    assert spec["eligible_layout_enumeration_sha256"] == (
        "753766cf2c3fb2120e235d9e6c8dcc7f8689a57972f8f07fd0381df3b2c742c5"
    )
    assert (STUDY / "layouts/L2_eligibility.json").read_text(encoding="utf-8") == L2.render(spec)


def test_external_seed_maps_to_one_balanced_layout_deterministically():
    segments = L2.LAYOUT1.rack_segments(L2.LAYOUT1.BASE_WORLD)
    groups = L2.eligible_groups(segments)
    eligibility = L2.eligibility_spec(segments)

    first, metadata_first = L2.select_layout_names(eligibility, groups, "abcd" * 32)
    second, metadata_second = L2.select_layout_names(eligibility, groups, "ABCD" * 32)

    assert first == second
    assert metadata_first == metadata_second
    assert len(first) == 12
    selected = {name: segment for name, segment in (
        (str(value["name"]), value) for value in segments
    )}
    strata = [L2.segment_stratum(selected[name]) for name in first]
    assert {stratum: strata.count(stratum) for stratum in L2.STRATA} == {
        stratum: 2 for stratum in L2.STRATA
    }


def test_beacon_seed_requires_and_preserves_public_provenance(tmp_path):
    path = tmp_path / "pulse.json"
    record = {
        "source_url": "https://beacon.nist.gov/beacon/2.0/pulse/time/123",
        "retrieved_at_utc": "2030-01-01T00:00:01Z",
        "pulse": {"pulse": {"outputValue": "AB" * 64, "timeStamp": "2030-01-01T00:00:00Z"}},
    }
    path.write_text(json.dumps(record), encoding="utf-8")

    seed, provenance = L2.beacon_seed(path)

    assert seed == "ab" * 64
    assert provenance["source_type"] == "NIST Randomness Beacon 2.0"
    assert provenance["record"] == record

    record["source_url"] = "https://example.com/not-public"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="beacon.nist.gov"):
        L2.beacon_seed(path)


def test_exact_membership_rejects_missing_extra_and_duplicate_rows():
    expected = ["A|1.00000000|2.00000000|0.00000000", "B|1.00000000|2.00000000|0.00000000"]
    CONTRACT.exact_membership(list(expected), expected, label="complete")

    with pytest.raises(ValueError, match="missing=1"):
        CONTRACT.exact_membership(expected[:1], expected, label="missing")
    with pytest.raises(ValueError, match="extra=1"):
        CONTRACT.exact_membership(expected + ["C|1|2|0"], expected, label="extra")
    with pytest.raises(ValueError, match="duplicate"):
        CONTRACT.exact_membership(expected + [expected[0]], expected, label="duplicate")


def test_l0_source_defines_exact_registered_membership():
    keys, metadata = CONTRACT.l0_expected_membership()

    assert len(keys) == CONTRACT.EXPECTED_COUNT == 15_072
    assert metadata["unique_position_count"] == 942
    assert metadata["membership_sha256"] == CONTRACT.membership_digest(keys)
