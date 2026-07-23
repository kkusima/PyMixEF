from __future__ import annotations

from pymixef.ir import ModelIR
from pymixef.provenance import (
    RunManifest,
    environment_snapshot,
    fingerprint_data,
    fingerprint_model_ir,
)


def test_fingerprints_are_order_stable() -> None:
    left = fingerprint_data({"b": [3, 4], "a": [1, 2]})
    right = fingerprint_data({"a": [1, 2], "b": [3, 4]})
    assert left == right


def test_manifest_captures_thread_settings() -> None:
    manifest = RunManifest.capture(
        model_ir={"x": 1},
        data={"y": [1]},
        engine="reference",
        method="test",
    )
    assert "thread_settings" in manifest.environment
    assert manifest.model_ir_hash.startswith("sha256:")
    assert environment_snapshot()["packages"]["numpy"]
    assert manifest.source["build_id"] == "unrecorded"
    assert manifest.source["git_commit"] == "unrecorded"


def test_model_ir_fingerprint_matches_object_and_dict_payload() -> None:
    model_ir = ModelIR(name="non-ASCII café model", response="y")
    assert fingerprint_model_ir(model_ir) == fingerprint_model_ir(model_ir.to_dict())
