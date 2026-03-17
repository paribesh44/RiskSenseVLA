"""Unit tests for the synthetic hazard dataset generation pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from risksense_vla.synthetic.io_export import DatasetWriter
from risksense_vla.synthetic.renderers import ProceduralRenderer, get_renderer
from risksense_vla.synthetic.scene_config import (
    DEFAULT_ACTION_TEMPLATES,
    HAZARD_TEMPLATES,
    ROOM_PRESETS,
    SceneConfig,
    build_scene_configs,
)
from risksense_vla.synthetic.sequence_engine import (
    AnnotatedFrame,
    AnnotatedSequence,
    SequenceEngine,
)


# ---------------------------------------------------------------------------
# Scene config tests
# ---------------------------------------------------------------------------


def test_scene_config_defaults():
    cfg = SceneConfig()
    assert cfg.room_type == "kitchen"
    assert cfg.num_frames == 24
    assert cfg.resolution == (640, 480)
    assert cfg.fps == 24
    assert cfg.camera_angles == ["front"]


@pytest.mark.parametrize("room", list(ROOM_PRESETS.keys()))
def test_scene_config_room_presets(room: str):
    preset = ROOM_PRESETS[room]
    assert "typical_objects" in preset
    assert "bg_color" in preset
    assert "lighting_options" in preset
    assert len(preset["typical_objects"]) >= 2
    cfg = SceneConfig(room_type=room, object_classes=preset["typical_objects"])
    assert cfg.room_type == room


def test_build_scene_configs_count():
    configs = build_scene_configs(num_scenes=20, seed=42)
    assert len(configs) == 20


def test_build_scene_configs_covers_all_hazards():
    configs = build_scene_configs(num_scenes=50, seed=0)
    seen = {cfg.hazard_templates[0] for cfg in configs if cfg.hazard_templates}
    assert seen == set(HAZARD_TEMPLATES.keys())


def test_build_scene_configs_rare_ratio():
    from risksense_vla.synthetic.scene_config import RARE_HAZARD_TYPES

    configs = build_scene_configs(num_scenes=100, rare_event_ratio=0.3, seed=7)
    rare_count = sum(
        1 for cfg in configs
        if cfg.hazard_templates and cfg.hazard_templates[0] in RARE_HAZARD_TYPES
    )
    assert rare_count >= 25


# ---------------------------------------------------------------------------
# Sequence engine tests
# ---------------------------------------------------------------------------


def test_sequence_engine_frame_count():
    cfg = SceneConfig(num_frames=16, hazard_templates=["spill_risk"])
    engine = SequenceEngine(seed=99)
    seqs = engine.generate(cfg, "test_001")
    assert len(seqs) == 1
    assert len(seqs[0].frames) >= 1
    assert len(seqs[0].frames) <= 16


def test_sequence_engine_multi_angle():
    cfg = SceneConfig(
        num_frames=8,
        hazard_templates=["clutter"],
        camera_angles=["front", "left", "right"],
    )
    engine = SequenceEngine(seed=10)
    seqs = engine.generate(cfg, "test_multi")
    assert len(seqs) == 3
    assert {s.camera_angle for s in seqs} == {"front", "left", "right"}


def test_sequence_engine_hoi_labels():
    cfg = SceneConfig(
        num_frames=12,
        hazard_templates=["sharp_tool_contact"],
        action_templates=["hold", "cut", "carry"],
        object_classes=["person", "knife"],
    )
    engine = SequenceEngine(seed=5)
    seqs = engine.generate(cfg, "test_hoi")
    for frame in seqs[0].frames:
        assert "subject" in frame.hoi
        assert "action" in frame.hoi
        assert "object" in frame.hoi
        assert "confidence" in frame.hoi
        assert 0.0 <= frame.hoi["confidence"] <= 1.0


def test_sequence_engine_hazard_scores_ramp():
    cfg = SceneConfig(num_frames=24, hazard_templates=["hot_surface_contact"])
    engine = SequenceEngine(seed=33)
    seqs = engine.generate(cfg, "ramp_test")
    scores = [f.hazard_score["score"] for f in seqs[0].frames]
    assert scores[-1] >= scores[0], "Hazard scores should generally increase"


def test_sequence_engine_hazard_coverage():
    """Generating 50+ scenes covers all hazard types."""
    configs = build_scene_configs(num_scenes=60, seed=1)
    engine = SequenceEngine(seed=1)
    seen_hazards: set[str] = set()
    for i, cfg in enumerate(configs):
        seqs = engine.generate(cfg, f"cov_{i:04d}")
        for seq in seqs:
            for frame in seq.frames:
                ht = frame.metadata.get("hazard_type", "")
                if ht:
                    seen_hazards.add(ht)
    assert seen_hazards == set(HAZARD_TEMPLATES.keys())


def test_sequence_engine_rare_events():
    from risksense_vla.synthetic.scene_config import RARE_HAZARD_TYPES

    configs = build_scene_configs(num_scenes=50, rare_event_ratio=0.3, seed=77)
    engine = SequenceEngine(seed=77)
    rare_count = 0
    for i, cfg in enumerate(configs):
        seqs = engine.generate(cfg, f"rare_{i:04d}")
        for seq in seqs:
            if any(
                f.metadata.get("hazard_type") in RARE_HAZARD_TYPES
                for f in seq.frames
            ):
                rare_count += 1
    assert rare_count >= 10


# ---------------------------------------------------------------------------
# Renderer tests
# ---------------------------------------------------------------------------


def test_procedural_renderer_output_shape():
    cfg = SceneConfig(resolution=(320, 240), hazard_templates=["clutter"])
    frame = AnnotatedFrame(
        frame_idx=0,
        objects=[
            {"label": "person", "bbox_xyxy": [10, 10, 60, 80], "track_id": "t0"},
            {"label": "knife", "bbox_xyxy": [100, 50, 140, 90], "track_id": "t1"},
        ],
        hoi={"subject": "person", "action": "hold", "object": "knife", "confidence": 0.8},
        hazard_score={
            "subject": "person", "action": "hold", "object": "knife",
            "score": 0.3, "severity": "low", "explanation": "test",
        },
    )
    renderer = ProceduralRenderer()
    img = renderer.render_frame(cfg, frame)
    assert img.shape == (240, 320, 3)
    assert img.dtype == np.uint8


def test_procedural_renderer_occlusion():
    cfg = SceneConfig(resolution=(160, 120))
    base_frame = AnnotatedFrame(
        frame_idx=0,
        objects=[{"label": "person", "bbox_xyxy": [10, 10, 50, 50], "track_id": "t0"}],
        hoi={"subject": "person", "action": "move", "object": "person", "confidence": 0.5},
        hazard_score={
            "subject": "person", "action": "move", "object": "person",
            "score": 0.1, "severity": "low", "explanation": "test",
        },
        occluded=False,
    )
    occ_frame = AnnotatedFrame(
        frame_idx=1,
        objects=base_frame.objects,
        hoi=base_frame.hoi,
        hazard_score=base_frame.hazard_score,
        occluded=True,
    )
    renderer = ProceduralRenderer()
    img_normal = renderer.render_frame(cfg, base_frame)
    img_occ = renderer.render_frame(cfg, occ_frame)
    assert not np.array_equal(img_normal, img_occ), "Occluded frame should differ"


def test_get_renderer_factory():
    r = get_renderer("procedural")
    assert isinstance(r, ProceduralRenderer)


# ---------------------------------------------------------------------------
# Dataset writer tests
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_output(tmp_path: Path) -> Path:
    return tmp_path / "synth_output"


def _make_sequence(num_frames: int = 4) -> tuple[AnnotatedSequence, list[np.ndarray]]:
    cfg = SceneConfig(
        resolution=(64, 48),
        hazard_templates=["spill_risk"],
        num_frames=num_frames,
    )
    engine = SequenceEngine(seed=42)
    seqs = engine.generate(cfg, "writer_test")
    renderer = ProceduralRenderer()
    seq = seqs[0]
    frames_bgr = [renderer.render_frame(cfg, f) for f in seq.frames]
    return seq, frames_bgr


def test_dataset_writer_creates_files(tmp_output: Path):
    seq, frames_bgr = _make_sequence()
    writer = DatasetWriter(tmp_output)
    writer.write_sequence(seq, frames_bgr)

    frame_dir = tmp_output / "frames" / seq.scene_id
    assert frame_dir.exists()
    pngs = list(frame_dir.glob("frame_*.png"))
    assert len(pngs) == len(frames_bgr)

    mp4 = tmp_output / "videos" / f"{seq.scene_id}.mp4"
    assert mp4.exists()

    ann = tmp_output / "annotations.jsonl"
    assert ann.exists()
    lines = ann.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 1


def test_annotations_schema(tmp_output: Path):
    seq, frames_bgr = _make_sequence()
    writer = DatasetWriter(tmp_output)
    writer.write_sequence(seq, frames_bgr)

    ann = tmp_output / "annotations.jsonl"
    for line in ann.read_text(encoding="utf-8").strip().splitlines():
        rec = json.loads(line)
        assert "scene_id" in rec
        assert "hazard_event" in rec
        assert "hazard_severity" in rec
        assert isinstance(rec["frames"], list)
        for frame in rec["frames"]:
            assert "frame_id" in frame
            assert "frame_idx" in frame
            assert "objects" in frame
            assert "hoi" in frame
            assert "hazard_score" in frame
            hoi = frame["hoi"]
            assert "subject" in hoi
            assert "action" in hoi
            assert "object" in hoi
            assert "confidence" in hoi
            hs = frame["hazard_score"]
            assert 0.0 <= hs["score"] <= 1.0
            assert hs["severity"] in {"low", "medium", "high"}


def test_mp4_frame_count(tmp_output: Path):
    seq, frames_bgr = _make_sequence(8)
    writer = DatasetWriter(tmp_output)
    writer.write_sequence(seq, frames_bgr)

    mp4 = tmp_output / "videos" / f"{seq.scene_id}.mp4"
    cap = cv2.VideoCapture(str(mp4))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert count == len(frames_bgr)


def test_legacy_jsonl(tmp_output: Path):
    seq, _ = _make_sequence()
    writer = DatasetWriter(tmp_output)
    legacy_path = tmp_output / "hazards.jsonl"
    writer.write_legacy_jsonl([seq], legacy_path)

    assert legacy_path.exists()
    lines = legacy_path.read_text(encoding="utf-8").strip().splitlines()
    rec = json.loads(lines[0])
    assert "scene_id" in rec
    assert "hazard_event" in rec
    assert "hazard_severity" in rec
    assert "frames" in rec


# ---------------------------------------------------------------------------
# Export format compatibility tests
# ---------------------------------------------------------------------------


def test_export_hoigen_format(tmp_output: Path):
    """Exported HOIGen JSON is loadable by HOIGenRawDataset."""
    from risksense_vla.hoi.datasets import HOIGenRawDataset

    seq, frames_bgr = _make_sequence()
    writer = DatasetWriter(tmp_output)
    writer.write_sequence(seq, frames_bgr)

    ann = tmp_output / "annotations.jsonl"
    records = [
        json.loads(line)
        for line in ann.read_text(encoding="utf-8").strip().splitlines()
    ]

    samples = []
    for rec in records:
        frames_out = []
        for frame in rec["frames"]:
            hoi = frame["hoi"]
            frames_out.append({
                "frame_idx": frame["frame_idx"],
                "hois": [
                    {
                        "subject": hoi["subject"],
                        "action": hoi["action"],
                        "object": hoi["object"],
                    }
                ],
            })
        samples.append({"video_id": rec["scene_id"], "frames": frames_out})

    hoigen_path = tmp_output / "hoigen_test.json"
    hoigen_path.write_text(json.dumps({"samples": samples}), encoding="utf-8")

    action_vocab = DEFAULT_ACTION_TEMPLATES + [
        "hold", "cut", "pour", "open", "touch_hot_surface",
        "carry", "drop", "close", "push", "pull", "move", "inspect", "interact",
    ]
    action_vocab = list(dict.fromkeys(action_vocab))
    ds = HOIGenRawDataset(str(hoigen_path), action_vocab=action_vocab)
    assert len(ds) >= 1
    item = ds[0]
    assert "object_embedding" in item
    assert "current_action_idx" in item


def test_export_temporal_jsonl(tmp_output: Path):
    """Exported temporal JSONL is loadable by TemporalHOIPreprocessedDataset."""
    from risksense_vla.hoi.datasets import TemporalHOIPreprocessedDataset

    seq, frames_bgr = _make_sequence(8)
    writer = DatasetWriter(tmp_output)
    writer.write_sequence(seq, frames_bgr)

    ann = tmp_output / "annotations.jsonl"
    records = [
        json.loads(line)
        for line in ann.read_text(encoding="utf-8").strip().splitlines()
    ]

    temporal_path = tmp_output / "temporal_test.jsonl"
    window = 4
    with temporal_path.open("w", encoding="utf-8") as f:
        for rec in records:
            frames = rec["frames"]
            for start in range(0, len(frames), window):
                chunk = frames[start : start + window]
                hois = []
                for frame in chunk:
                    hoi = frame["hoi"]
                    hois.append({
                        "subject": hoi["subject"],
                        "action": hoi["action"],
                        "object": hoi["object"],
                        "frame_idx": frame["frame_idx"],
                    })
                entry = {
                    "video_id": rec["scene_id"],
                    "start_frame": chunk[0]["frame_idx"],
                    "end_frame": chunk[-1]["frame_idx"],
                    "hois": hois,
                }
                f.write(json.dumps(entry) + "\n")

    action_vocab = DEFAULT_ACTION_TEMPLATES + [
        "hold", "cut", "pour", "open", "touch_hot_surface",
        "carry", "drop", "close", "push", "pull", "move", "inspect", "interact",
    ]
    action_vocab = list(dict.fromkeys(action_vocab))
    ds = TemporalHOIPreprocessedDataset(str(temporal_path), action_vocab=action_vocab)
    assert len(ds) >= 1
    item = ds[0]
    assert "object_embedding" in item
    assert "future_action_indices" in item
