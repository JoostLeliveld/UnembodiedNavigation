"""Acceptance tests for the monocular depth adapter.

The five acceptance criteria are covered here:

1. deterministic inference on a frozen image set
2. correct output dimensions and depth convention
3. invalid pixels explicitly marked
4. batch inference works for all four cameras
5. runtime and memory use recorded

Most of them are checked through a deterministic stub backend, so the whole file
runs in about a second and stays in the default suite. The stub goes through the
exact same adapter code path as a real network — batching, valid masks, timing,
memory, flip-consistency, storage — so what is being tested is the adapter, not
the stub.

The same criteria against the real networks need a GPU and several minutes, so
those tests are opt-in::

    MONODEPTH_GPU_TESTS=1 python3 -m pytest tests/perception/test_monocular_depth_adapter.py

There is also a contract test that the adapter package cannot reach ground
truth, simulator depth, or the camera calibration code, enforced by parsing its
imports rather than by trusting the docstrings.
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
STUDY = REPO / "experiments" / "monocular_depth_adapter"
for _p in (str(STUDY),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import frozen_set as fs  # noqa: E402
from monodepth import (  # noqa: E402
    CameraIntrinsics,
    DepthConvention,
    DepthRequest,
    MonocularDepthAdapter,
    available_models,
    build_valid_mask,
    conventions,
    storage,
    uncertainty as unc,
)
from monodepth.backends import base as backend_base  # noqa: E402
from monodepth.determinism import array_fingerprint  # noqa: E402
from monodepth.types import BackendInfo  # noqa: E402

GPU_TESTS = os.environ.get("MONODEPTH_GPU_TESTS") == "1"
gpu_only = pytest.mark.skipif(
    not GPU_TESTS, reason="set MONODEPTH_GPU_TESTS=1 to run the real models (needs a GPU, minutes)"
)


# --------------------------------------------------------------------------- stub
class StubBackend(backend_base.DepthBackend):
    """A backend whose output depends only on the image and intrinsics.

    Deterministic by construction, and it plants a block of NaN plus a block of
    negatives so the valid-mask behaviour has something real to catch.
    """

    family = "stub"

    def __init__(self, model_name="stub_metric", device="cpu", dtype="float32",
                 convention=DepthConvention.METRIC_Z):
        super().__init__(device=device, dtype=dtype)
        self.model_name = model_name
        self._convention = convention
        self.calls: list[int] = []

    def load(self):
        self._loaded = True

    def info(self):
        return BackendInfo(
            backend=self.family, model_name=self.model_name, checkpoint="stub://v1",
            checkpoint_revision="deadbeef", convention=self._convention,
            provides_native_confidence=True, uses_intrinsics=True,
            native_input_size=(64, 64), device=self.device, torch_dtype=self.dtype,
            parameter_count=7, library_versions={"stub": "1.0"}, notes="test double",
        )

    def infer_batch(self, images, intrinsics):
        self.calls.append(len(images))
        h, w = self._check_uniform_shape(images)
        out = []
        for img, intr in zip(images, intrinsics):
            depth = (img.astype(np.float64).mean(axis=2) / 255.0 * 5.0 + 1.0)
            depth *= intr.fx / 640.0                      # a real dependence on K
            depth[:4, :4] = np.nan                        # model says: no idea here
            depth[-4:, -4:] = -1.0                        # and: impossible value here
            out.append(backend_base.RawDepthOutput(
                depth=depth.astype(np.float32),
                native_confidence=np.full((h, w), 0.5, dtype=np.float32),
                extras={"stub": True, "radius_m": depth.astype(np.float32)},
            ))
        return out


@pytest.fixture()
def stub_adapter(monkeypatch):
    """An adapter wired to the stub, registered under a real-looking name."""
    backends = sys.modules["monodepth.backends"]
    monkeypatch.setitem(backends.MODEL_FAMILIES, "stub", ("stub_metric", "stub_relative"))
    monkeypatch.setitem(backends._REGISTRARS, "stub", lambda: None)
    monkeypatch.setitem(backends._FACTORIES, "stub_metric",
                        lambda **kw: StubBackend("stub_metric", **kw))
    monkeypatch.setitem(backends._FACTORIES, "stub_relative",
                        lambda **kw: StubBackend("stub_relative",
                                                 convention=DepthConvention.RELATIVE_DEPTH, **kw))

    def make(model_name="stub_metric", **kwargs):
        kwargs.setdefault("device", "cpu")
        return MonocularDepthAdapter(model_name, **kwargs).load()

    return make


def _requests(n=4, width=64, height=48, seed=0):
    rng = np.random.default_rng(seed)
    reqs = []
    for i in range(n):
        img = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
        intr = CameraIntrinsics(fx=640.0 + i, fy=640.0 + i, cx=width / 2, cy=height / 2,
                                width=width, height=height)
        reqs.append(DepthRequest(f"frame{i}", img, intr))
    return reqs


# ------------------------------------------------------ acceptance 2: dims + units
def test_output_has_the_input_dimensions(stub_adapter):
    adapter = stub_adapter()
    reqs = _requests(3, width=80, height=60)
    preds = adapter.predict(reqs)
    assert len(preds) == 3
    for req, pred in zip(reqs, preds):
        assert pred.depth.shape == (60, 80)
        assert pred.valid.shape == pred.depth.shape
        assert pred.depth.dtype == np.float32
        assert pred.image_id == req.image_id


def test_depth_convention_is_carried_and_not_silently_converted(stub_adapter):
    metric = stub_adapter("stub_metric").predict(_requests(1))[0]
    relative = stub_adapter("stub_relative").predict(_requests(1))[0]
    assert metric.convention is DepthConvention.METRIC_Z
    assert metric.convention.is_metric and metric.convention.unit == "m"
    assert relative.convention is DepthConvention.RELATIVE_DEPTH
    assert not relative.convention.is_metric and relative.convention.unit == "unitless"
    # same pixels, different declared meaning: the adapter must not have rescaled
    np.testing.assert_array_equal(metric.depth, relative.depth)


def test_metadata_states_the_convention_and_the_checkpoint(stub_adapter):
    meta = stub_adapter().predict(_requests(1))[0].metadata()
    assert meta["convention"] == "metric_z"
    assert meta["unit"] == "m"
    assert meta["larger_is_nearer"] is False
    assert meta["model"]["checkpoint"] == "stub://v1"
    assert meta["model"]["model_name"] == "stub_metric"


def test_inverse_depth_is_flagged_as_larger_is_nearer():
    assert DepthConvention.INVERSE_DEPTH.larger_is_nearer
    assert not DepthConvention.METRIC_Z.larger_is_nearer
    assert not DepthConvention.EUCLIDEAN_RANGE.larger_is_nearer
    assert not DepthConvention.RELATIVE_DEPTH.larger_is_nearer


def test_range_and_axis_depth_differ_by_the_secant_of_the_off_axis_angle():
    intr = CameraIntrinsics(fx=640.0, fy=640.0, cx=640.0, cy=360.0, width=1280, height=720)
    z = np.full((720, 1280), 10.0, dtype=np.float32)
    r = conventions.z_to_euclidean(z, intr)
    assert r[360, 640] == pytest.approx(10.0, abs=1e-3)         # on the axis: equal
    expected = 10.0 * np.sqrt(1 + (640 / 640) ** 2 + (360 / 640) ** 2)
    assert r[0, 0] == pytest.approx(expected, rel=1e-4)          # corner: ~1.5x
    np.testing.assert_allclose(conventions.euclidean_to_z(r, intr), z, rtol=1e-5)


def test_non_metric_conversion_is_refused_rather_than_guessed():
    intr = CameraIntrinsics(fx=1.0, fy=1.0, cx=0.5, cy=0.5, width=2, height=2)
    with pytest.raises(ValueError, match="scene anchor"):
        conventions.convert(np.ones((2, 2)), DepthConvention.RELATIVE_DEPTH,
                            DepthConvention.METRIC_Z, intr)
    with pytest.raises(ValueError, match="scene anchor"):
        conventions.convert(np.ones((2, 2)), DepthConvention.INVERSE_DEPTH,
                            DepthConvention.METRIC_Z, intr)


# ---------------------------------------------------- acceptance 3: invalid pixels
def test_invalid_pixels_are_marked_not_dropped(stub_adapter):
    pred = stub_adapter().predict(_requests(1, width=64, height=48))[0]
    assert not pred.valid[:4, :4].any(), "NaN block must be marked invalid"
    assert not pred.valid[-4:, -4:].any(), "negative block must be marked invalid"
    assert pred.valid[20:30, 20:30].all(), "ordinary pixels must stay valid"
    assert pred.depth.shape == pred.valid.shape, "invalid pixels stay in place, masked"
    assert 0.0 < pred.valid_fraction < 1.0


def test_valid_mask_rules_per_convention():
    depth = np.array([[np.nan, -1.0], [0.0, 5.0]], dtype=np.float32)
    metric = build_valid_mask(depth, DepthConvention.METRIC_Z)
    np.testing.assert_array_equal(metric, [[False, False], [False, True]])

    # zero inverse depth means infinitely far, which is a legal reading
    inverse = build_valid_mask(depth, DepthConvention.INVERSE_DEPTH)
    np.testing.assert_array_equal(inverse, [[False, False], [True, True]])

    clamped = build_valid_mask(depth, DepthConvention.METRIC_Z, max_metric_m=4.0)
    np.testing.assert_array_equal(clamped, [[False, False], [False, False]])

    extra = build_valid_mask(depth, DepthConvention.METRIC_Z,
                             extra_invalid=np.array([[False, False], [False, True]]))
    assert not extra.any()


# ------------------------------------------------------- acceptance 1: determinism
def test_repeated_inference_is_bit_identical(stub_adapter):
    adapter = stub_adapter()
    reqs = _requests(3)
    first = adapter.predict(reqs)
    second = adapter.predict(reqs)
    for a, b in zip(first, second):
        assert array_fingerprint(a.depth) == array_fingerprint(b.depth)
        assert array_fingerprint(a.valid) == array_fingerprint(b.valid)


def test_batch_size_does_not_change_the_numbers(stub_adapter):
    reqs = _requests(4)
    one = stub_adapter(batch_size=1).predict(reqs)
    four = stub_adapter(batch_size=4).predict(reqs)
    for a, b in zip(one, four):
        assert array_fingerprint(a.depth) == array_fingerprint(b.depth)


def test_determinism_flags_are_recorded(stub_adapter):
    cfg = stub_adapter().determinism_config
    assert cfg["cudnn_deterministic"] is True
    assert cfg["cudnn_benchmark"] is False
    assert cfg["allow_tf32"] is False
    assert cfg["cublas_workspace_config"] == ":4096:8"


# ------------------------------------------------------- acceptance 4: four cameras
def test_batch_inference_over_all_four_cameras(stub_adapter):
    """One call, four cameras, four different calibrations, order preserved."""
    frames = fs.load_frames(role="batch_plumbing_only")
    cameras = sorted({f.camera_id for f in frames})
    assert cameras == ["A", "B", "C", "D"], f"frozen set must cover four cameras, got {cameras}"

    one_pose = sorted({f.source_sample_id for f in frames})[0]
    quad = [f for f in frames if f.source_sample_id == one_pose]
    assert len(quad) == 4

    # Stand-in imagery keeps this test off the disk and off the GPU; the four
    # real calibrations are the part that matters here.
    reqs = [
        DepthRequest(f.frame_id,
                     np.full((48, 64, 3), 40 + 30 * i, dtype=np.uint8),
                     CameraIntrinsics(fx=f.intrinsics["fx"], fy=f.intrinsics["fy"],
                                      cx=32.0, cy=24.0, width=64, height=48))
        for i, f in enumerate(quad)
    ]
    adapter = stub_adapter(batch_size=4)
    preds = adapter.predict(reqs)

    assert [p.image_id for p in preds] == [r.image_id for r in reqs]
    assert {p.image_id.split("_")[1] for p in preds} == {"camA", "camB", "camC", "camD"}
    assert all(p.timing.batch_size == 4 for p in preds), "all four went through one forward"
    assert all(p.depth.shape == (48, 64) for p in preds)


def test_mixed_image_sizes_are_grouped_not_rejected(stub_adapter):
    small = _requests(2, width=32, height=24, seed=1)
    large = _requests(2, width=64, height=48, seed=2)
    preds = stub_adapter(batch_size=4).predict(small + large)
    assert [p.depth.shape for p in preds] == [(24, 32), (24, 32), (48, 64), (48, 64)]


# --------------------------------------------------- acceptance 5: runtime + memory
def test_runtime_and_memory_are_recorded(stub_adapter):
    pred = stub_adapter().predict(_requests(1))[0]
    t = pred.timing
    assert t.forward_s > 0.0
    assert t.total_s >= t.forward_s
    assert t.batch_size == 1
    assert np.isfinite(pred.memory.host_rss_mib) and pred.memory.host_rss_mib > 0
    meta = pred.metadata()
    assert set(meta["timing"]) >= {"preprocess_s", "forward_s", "postprocess_s",
                                   "uncertainty_s", "total_s"}
    assert set(meta["memory"]) >= {"gpu_peak_allocated_mib", "gpu_peak_reserved_mib",
                                   "host_rss_mib", "weights_mib"}


def test_model_identity_is_recorded(stub_adapter):
    info = stub_adapter().info
    assert info.model_name == "stub_metric"
    assert info.checkpoint == "stub://v1"
    assert info.checkpoint_revision == "deadbeef"
    assert info.parameter_count == 7
    assert info.library_versions["stub"] == "1.0"


# --------------------------------------------------------------------- uncertainty
def test_native_confidence_is_returned_when_the_model_has_one(stub_adapter):
    pred = stub_adapter(uncertainty="native").predict(_requests(1))[0]
    assert pred.uncertainty_kind == unc.NATIVE
    assert pred.uncertainty is not None
    assert pred.native_confidence is not None


def test_flip_consistency_costs_a_second_pass_and_returns_a_spread(stub_adapter):
    adapter = stub_adapter(uncertainty="native+flip")
    pred = adapter.predict(_requests(1))[0]
    assert pred.uncertainty_kind == unc.FLIP
    assert pred.uncertainty is not None
    assert pred.uncertainty.shape == pred.depth.shape
    assert pred.uncertainty_detail["unit"] == "m"
    assert adapter._backend.calls == [1, 1], "one forward for the image, one for its mirror"
    assert pred.timing.uncertainty_s > 0.0, "the extra pass must show up in the cost"
    assert pred.timing.total_s >= pred.timing.forward_s + pred.timing.uncertainty_s


def test_flip_consistency_affine_aligns_only_the_non_metric_models():
    valid = np.ones((8, 8), dtype=bool)
    a = np.linspace(1, 5, 64).reshape(8, 8)
    b = 2.0 * a + 3.0                                   # same shape, different gauge

    _, metric_detail = unc.flip_consistency(a, b, valid, DepthConvention.METRIC_Z)
    assert metric_detail["affine_aligned"] is False
    assert metric_detail["median_spread"] > 0.1, "a metric scale error is a real error"

    spread, rel_detail = unc.flip_consistency(a, b, valid, DepthConvention.RELATIVE_DEPTH)
    assert rel_detail["affine_aligned"] is True
    assert rel_detail["median_spread"] == pytest.approx(0.0, abs=1e-9), \
        "for relative depth, an affine difference is gauge, not disagreement"
    assert np.isfinite(spread).all()


def test_uncertainty_is_absent_and_says_why_when_the_model_has_no_confidence(stub_adapter, monkeypatch):
    adapter = stub_adapter(uncertainty="native")
    monkeypatch.setattr(adapter._backend, "info", lambda: BackendInfo(
        backend="stub", model_name="stub_metric", checkpoint="stub://v1",
        checkpoint_revision=None, convention=DepthConvention.METRIC_Z,
        provides_native_confidence=False, uses_intrinsics=True, native_input_size=None))
    monkeypatch.setattr(adapter._backend, "infer_batch", lambda images, intrinsics: [
        backend_base.RawDepthOutput(depth=np.ones(i.shape[:2], dtype=np.float32))
        for i in images])
    pred = adapter.predict(_requests(1))[0]
    assert pred.uncertainty is None
    assert pred.uncertainty_kind is None
    assert pred.uncertainty_detail["available"] is False
    assert "no native confidence head" in pred.uncertainty_detail["reason"]


def test_temporal_disagreement_needs_one_camera_and_one_model(stub_adapter):
    preds = stub_adapter().predict(_requests(3))
    with pytest.raises(ValueError, match="mixed intrinsics"):
        unc.temporal_disagreement(preds)          # the stub requests vary fx on purpose

    same = _requests(3)
    fixed = CameraIntrinsics(fx=640.0, fy=640.0, cx=32.0, cy=24.0, width=64, height=48)
    same = [DepthRequest(r.image_id, r.image, fixed) for r in same]
    spread, detail = unc.temporal_disagreement(stub_adapter().predict(same))
    assert spread.shape == (48, 64)
    assert detail["n_frames"] == 3
    assert detail["method"] == unc.TEMPORAL
    assert np.isnan(spread[:4, :4]).all(), "pixels invalid in every frame stay unknown"


# ------------------------------------------------------------------------- storage
def test_saved_and_reloaded_predictions_are_identical(stub_adapter, tmp_path):
    pred = stub_adapter(uncertainty="native+flip").predict(_requests(1))[0]
    npz, sidecar = storage.save_prediction(pred, tmp_path)
    assert npz.exists() and sidecar.exists()

    back = storage.load_prediction(sidecar)
    assert array_fingerprint(back.depth) == array_fingerprint(pred.depth)
    assert array_fingerprint(back.valid) == array_fingerprint(pred.valid)
    assert back.convention is pred.convention
    assert back.model.checkpoint == pred.model.checkpoint
    assert back.intrinsics == pred.intrinsics
    assert back.timing.total_s == pytest.approx(pred.timing.total_s)
    assert set(back.extra_arrays) == set(pred.extra_arrays)


def test_a_prediction_without_a_recorded_convention_is_refused(stub_adapter, tmp_path):
    pred = stub_adapter().predict(_requests(1))[0]
    _, sidecar = storage.save_prediction(pred, tmp_path)
    import json

    meta = json.loads(sidecar.read_text())
    del meta["convention"]
    sidecar.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="refusing to guess"):
        storage.load_prediction(sidecar)


# --------------------------------------------------------------------- frozen set
def test_frozen_set_manifest_is_intact():
    problems = fs.verify()
    assert problems == [], f"frozen set changed under us: {problems}"


def test_frozen_set_covers_both_roles_and_four_cameras():
    frames = fs.load_frames()
    dev = [f for f in frames if f.role == "method_development"]
    plumbing = [f for f in frames if f.role == "batch_plumbing_only"]
    assert dev, "need method-development frames"
    assert {f.world for f in dev} == {"warehouse_aws.world.sdf"}, \
        "method development happens in the original warehouse only"
    assert {f.camera_id for f in plumbing} == {"A", "B", "C", "D"}
    assert {f.world for f in plumbing} == {"warehouse_full_4cam.world.sdf"}
    assert all(len(f.sha256) == 64 for f in frames)


def test_frozen_frames_carry_usable_intrinsics():
    for frame in fs.load_frames():
        intr = frame.camera_intrinsics()
        assert intr.width == frame.width and intr.height == frame.height
        assert intr.fx > 0 and intr.fy > 0


# ---------------------------------------------------------------- the "must not" list
FORBIDDEN_IMPORTS = {
    # repo packages: the adapter must not be able to reach the pipeline around it
    "unav_common", "experiments", "reliability", "planning", "perception", "state", "sim",
    "geometry_visibility", "campaign_metrics", "frozen_set", "metrics", "common",
    # ROS: predictions are saved to disk at this stage, not published
    "rclpy", "rosidl_runtime_py", "sensor_msgs", "cv_bridge", "std_msgs", "geometry_msgs",
}


def _adapter_modules():
    return sorted((STUDY / "monodepth").rglob("*.py"))


def test_adapter_package_imports_nothing_from_the_repo_or_from_ros():
    offenders = []
    for path in _adapter_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module] if node.level == 0 and node.module else []
            else:
                continue
            for name in names:
                root = (name or "").split(".")[0]
                if root in FORBIDDEN_IMPORTS:
                    offenders.append(f"{path.relative_to(STUDY)}: {name}")
    assert offenders == [], (
        "the depth adapter reached outside its own package: " + "; ".join(offenders)
    )


def test_backend_signature_offers_no_route_to_truth_or_geometry():
    import inspect

    params = list(inspect.signature(backend_base.DepthBackend.infer_batch).parameters)
    assert params == ["self", "images", "intrinsics"], (
        "a backend may only ever see an image and its intrinsics; "
        f"got {params}"
    )
    request_fields = set(DepthRequest.__dataclass_fields__)
    assert request_fields == {"image_id", "image", "intrinsics", "source_path", "image_sha256"}


def test_adapter_never_touches_ground_truth_or_simulator_depth_paths():
    needles = ("gt_", "truth", "oracle", "depth_stack", "eval_", "keep_out", "keep_in")
    offenders = []
    for path in _adapter_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                lowered = node.value.lower()
                # docstrings state what is forbidden; only short literals are code-like
                if len(node.value) < 120 and any(n in lowered for n in needles):
                    offenders.append(f"{path.relative_to(STUDY)}: {node.value!r}")
    assert offenders == [], "suspicious literals in the adapter: " + "; ".join(offenders)


def test_registry_lists_the_three_model_families_without_importing_them():
    models = available_models()
    assert "dav2_metric_indoor_large" in models
    assert "metric3d_v2_vit_small" in models
    assert "unidepth_v2_vits14" in models
    from monodepth.backends import MODEL_FAMILIES, family_of

    assert set(MODEL_FAMILIES) == {"depth_anything_v2", "metric3d_v2", "unidepth_v2"}
    assert family_of("unidepth_v2_vitl14") == "unidepth_v2"
    with pytest.raises(KeyError):
        family_of("not_a_model")


def test_adapter_rejects_a_bad_uncertainty_mode():
    with pytest.raises(ValueError, match="uncertainty must be one of"):
        MonocularDepthAdapter("dav2_metric_indoor_small", uncertainty="vibes")


def test_request_rejects_an_image_that_does_not_match_its_calibration():
    intr = CameraIntrinsics(fx=1.0, fy=1.0, cx=1.0, cy=1.0, width=8, height=8)
    with pytest.raises(ValueError, match="intrinsics declare"):
        DepthRequest("bad", np.zeros((4, 4, 3), dtype=np.uint8), intr)
    with pytest.raises(ValueError, match="uint8"):
        DepthRequest("bad", np.zeros((8, 8, 3), dtype=np.float32), intr)


# ------------------------------------------------------------------ real models
@gpu_only
@pytest.mark.parametrize("model_name", ["dav2_metric_indoor_small",
                                        "metric3d_v2_vit_small",
                                        "unidepth_v2_vits14"])
def test_real_model_is_deterministic_and_declares_its_convention(model_name):
    frames = fs.load_frames(role="method_development")[:2]
    reqs = fs.to_requests(frames)
    with MonocularDepthAdapter(model_name, batch_size=2, uncertainty="native") as adapter:
        first = adapter.predict(reqs)
        second = adapter.predict(reqs)
        info = adapter.info

    assert info.convention in tuple(DepthConvention)
    for a, b in zip(first, second):
        assert array_fingerprint(a.depth) == array_fingerprint(b.depth)
        assert a.depth.shape == (frames[0].height, frames[0].width)
        assert a.valid.any()
        assert a.timing.forward_s > 0
        assert a.memory.gpu_peak_allocated_mib > 0


@gpu_only
def test_real_batch_over_the_four_cameras():
    frames = [f for f in fs.load_frames(role="batch_plumbing_only")
              if f.source_sample_id == sorted({x.source_sample_id for x in
                                               fs.load_frames(role="batch_plumbing_only")})[0]]
    assert {f.camera_id for f in frames} == {"A", "B", "C", "D"}
    with MonocularDepthAdapter("dav2_metric_indoor_small", batch_size=4,
                               uncertainty="none") as adapter:
        preds = adapter.predict(fs.to_requests(frames))
    assert len(preds) == 4
    assert all(p.depth.shape == (720, 1280) for p in preds)
    assert all(p.valid.any() for p in preds)
