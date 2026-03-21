"""Comprehensive unit tests for evaluation metrics (THC, HAA, RME, FPS)."""

from __future__ import annotations


from risksense_vla.eval.metrics import (
    SequenceMetrics,
    aggregate_sequences,
    evaluate_sequence,
    frame_metrics,
    hazard_lead_time,
    hazard_anticipation_accuracy,
    risk_weighted_memory_efficiency,
    temporal_hoi_consistency,
    _safe_mean,
)


# ── _safe_mean ────────────────────────────────────────────────────────


class TestSafeMean:
    def test_empty(self) -> None:
        assert _safe_mean([]) == 0.0

    def test_single(self) -> None:
        assert _safe_mean([5.0]) == 5.0

    def test_multiple(self) -> None:
        assert abs(_safe_mean([1.0, 3.0]) - 2.0) < 1e-9


# ── THC ───────────────────────────────────────────────────────────────


class TestTHC:
    def test_empty_records(self) -> None:
        assert temporal_hoi_consistency([]) == 0.0

    def test_single_record(self) -> None:
        assert temporal_hoi_consistency([{"hois": [{"action": "cut", "confidence": 0.9}]}]) == 0.0

    def test_all_consistent(self) -> None:
        records = [
            {"hois": [{"action": "cut", "confidence": 0.9}]},
            {"hois": [{"action": "cut", "confidence": 0.8}]},
            {"hois": [{"action": "cut", "confidence": 0.7}]},
        ]
        assert temporal_hoi_consistency(records) == 1.0

    def test_all_inconsistent(self) -> None:
        records = [
            {"hois": [{"action": "cut", "confidence": 0.9}]},
            {"hois": [{"action": "hold", "confidence": 0.8}]},
            {"hois": [{"action": "pour", "confidence": 0.7}]},
        ]
        assert temporal_hoi_consistency(records) == 0.0

    def test_mixed_consistency(self) -> None:
        records = [
            {"hois": [{"action": "cut", "confidence": 0.9}]},
            {"hois": [{"action": "cut", "confidence": 0.8}]},
            {"hois": [{"action": "hold", "confidence": 0.7}]},
        ]
        assert abs(temporal_hoi_consistency(records) - 0.5) < 1e-9

    def test_skips_predicted_hois(self) -> None:
        records = [
            {"hois": [{"action": "cut", "confidence": 0.9}]},
            {"hois": [{"action": "hold", "confidence": 0.99, "predicted": True}]},
            {"hois": [{"action": "cut", "confidence": 0.8}]},
        ]
        assert temporal_hoi_consistency(records) == 1.0

    def test_empty_hois_skipped(self) -> None:
        records = [
            {"hois": [{"action": "cut", "confidence": 0.9}]},
            {"hois": []},
            {"hois": [{"action": "cut", "confidence": 0.7}]},
        ]
        assert temporal_hoi_consistency(records) == 1.0

    def test_no_observed_hois_at_all(self) -> None:
        records = [
            {"hois": [{"action": "cut", "confidence": 0.9, "predicted": True}]},
            {"hois": [{"action": "hold", "confidence": 0.8, "predicted": True}]},
        ]
        assert temporal_hoi_consistency(records) == 0.0


# ── HAA ───────────────────────────────────────────────────────────────


class TestHAA:
    def test_no_hazards(self) -> None:
        records = [{"frame_id": i, "hazards": [], "hois": []} for i in range(10)]
        assert hazard_anticipation_accuracy(records) == 0.0


class TestHazardLeadTime:
    def test_empty(self) -> None:
        out = hazard_lead_time([], [])
        assert out["mean"] == 0.0
        assert out["median"] == 0.0
        assert out["distribution"] == []
        assert out["missing_pre_event"] == 0

    def test_signed(self) -> None:
        events = [{"frame_id": 20, "track_id": "knife", "action": "cut"}]
        predictions = [{"source_frame_id": 25, "track_id": "knife", "predicted_action": "cut"}]
        out = hazard_lead_time(events, predictions)
        assert out["samples"] == 1
        assert out["mean"] == -5.0
        assert out["missing_pre_event"] == 1

    def test_uses_closest_pre_event_prediction_per_event(self) -> None:
        events = [{"frame_id": 20, "track_id": "knife_1", "action": "cut"}]
        predictions = [
            {"source_frame_id": 3, "track_id": "knife_1", "predicted_action": "cut"},
            {"source_frame_id": 11, "track_id": "knife_1", "predicted_action": "cut"},
            {"source_frame_id": 18, "track_id": "knife_1", "predicted_action": "cut"},
        ]
        out = hazard_lead_time(events, predictions)
        assert out["distribution"] == [2]

    def test_no_prediction_for_key_records_none(self) -> None:
        events = [{"frame_id": 20, "track_id": "knife_1", "action": "cut"}]
        predictions = [{"source_frame_id": 10, "track_id": "other_track", "predicted_action": "cut"}]
        out = hazard_lead_time(events, predictions)
        assert out["samples"] == 0
        assert out["distribution"] == [None]

    def test_all_anticipated(self) -> None:
        records = [
            {"frame_id": 0, "hazards": [], "hois": [{"predicted": True}]},
            {"frame_id": 5, "hazards": [{"score": 0.9, "action": "cut"}], "hois": []},
        ]
        assert hazard_anticipation_accuracy(records, lead_frames=10) == 1.0

    def test_none_anticipated(self) -> None:
        records = [
            {"frame_id": 100, "hazards": [{"score": 0.9, "action": "cut"}], "hois": []},
        ]
        assert hazard_anticipation_accuracy(records) == 0.0

    def test_boundary_exact_lead(self) -> None:
        records = [
            {"frame_id": 0, "hazards": [], "hois": [{"predicted": True}]},
            {"frame_id": 25, "hazards": [{"score": 0.9, "action": "cut"}], "hois": []},
        ]
        assert hazard_anticipation_accuracy(records, lead_frames=25) == 1.0

    def test_beyond_lead_window(self) -> None:
        records = [
            {"frame_id": 0, "hazards": [], "hois": [{"predicted": True}]},
            {"frame_id": 26, "hazards": [{"score": 0.9, "action": "cut"}], "hois": []},
        ]
        assert hazard_anticipation_accuracy(records, lead_frames=25) == 0.0

    def test_custom_hazard_threshold(self) -> None:
        records = [
            {"frame_id": 0, "hazards": [], "hois": [{"predicted": True}]},
            {"frame_id": 5, "hazards": [{"score": 0.5, "action": "cut"}], "hois": []},
        ]
        assert hazard_anticipation_accuracy(records, hazard_threshold=0.3) == 1.0
        assert hazard_anticipation_accuracy(records, hazard_threshold=0.7) == 0.0

    def test_below_threshold_not_counted(self) -> None:
        records = [
            {"frame_id": 0, "hazards": [{"score": 0.3, "action": "cut"}], "hois": []},
        ]
        assert hazard_anticipation_accuracy(records) == 0.0


# ── RME ───────────────────────────────────────────────────────────────


class TestRME:
    def test_empty_records(self) -> None:
        assert risk_weighted_memory_efficiency([]) == 0.0

    def test_no_hazards_gives_zero(self) -> None:
        records = [
            {"hazards": [], "attention_allocation": {"obj_0": 1.0}},
        ]
        assert risk_weighted_memory_efficiency(records) == 0.0

    def test_high_risk_high_compute(self) -> None:
        records = [
            {"hazards": [{"score": 1.0}], "attention_allocation": {"obj_0": 1.0}},
        ]
        result = risk_weighted_memory_efficiency(records)
        assert abs(result - 1.0) < 1e-6

    def test_mixed(self) -> None:
        records = [
            {"hazards": [{"score": 0.5}], "attention_allocation": {"obj_0": 0.5}},
        ]
        result = risk_weighted_memory_efficiency(records)
        assert abs(result - 0.25) < 1e-6

    def test_clipped_to_unit(self) -> None:
        result = risk_weighted_memory_efficiency([
            {"hazards": [{"score": 0.8}], "attention_allocation": {"a": 0.9}},
        ])
        assert 0.0 <= result <= 1.0


# ── FPS / evaluate_sequence ──────────────────────────────────────────


class TestEvaluateSequence:
    def test_zero_latency(self) -> None:
        records = [{"frame_id": 0, "hois": [], "hazards": [], "detections": [], "latency_ms": {}}]
        sm = evaluate_sequence(records)
        assert sm.fps == 0.0

    def test_normal_latency(self) -> None:
        records = [
            {"frame_id": i, "hois": [], "hazards": [], "detections": [],
             "latency_ms": {"perception": 20.0, "memory": 5.0}}
            for i in range(5)
        ]
        sm = evaluate_sequence(records)
        assert sm.fps > 0
        assert sm.latency_ms > 0

    def test_custom_thresholds(self) -> None:
        records = [
            {"frame_id": 0, "hois": [{"predicted": True}], "hazards": [],
             "detections": [], "latency_ms": {"p": 10.0}},
            {"frame_id": 5, "hois": [], "hazards": [{"score": 0.5, "action": "cut"}],
             "detections": [], "latency_ms": {"p": 10.0}},
        ]
        sm1 = evaluate_sequence(records, hazard_threshold=0.3)
        sm2 = evaluate_sequence(records, hazard_threshold=0.8)
        assert sm1.haa >= sm2.haa


# ── aggregate_sequences ──────────────────────────────────────────────


class TestAggregateSequences:
    def test_empty(self) -> None:
        result = aggregate_sequences([])
        assert result["THC"] == 0.0
        assert result["FPS"] == 0.0

    def test_single(self) -> None:
        sm = SequenceMetrics(thc=0.8, haa=0.6, rme=0.3, fps=30.0, latency_ms=33.3, detection_map=0.7)
        result = aggregate_sequences([sm])
        assert abs(result["THC"] - 0.8) < 1e-6

    def test_averaging(self) -> None:
        s1 = SequenceMetrics(thc=0.8, haa=0.6, rme=0.2, fps=30.0, latency_ms=33.3, detection_map=0.5)
        s2 = SequenceMetrics(thc=0.4, haa=0.2, rme=0.4, fps=20.0, latency_ms=50.0, detection_map=0.3)
        result = aggregate_sequences([s1, s2])
        assert abs(result["THC"] - 0.6) < 1e-6
        assert abs(result["HAA"] - 0.4) < 1e-6


# ── frame_metrics ─────────────────────────────────────────────────────


class TestFrameMetrics:
    def test_basic(self) -> None:
        rec = {
            "frame_id": 5,
            "detections": [{"confidence": 0.9}],
            "hois": [{"action": "cut"}],
            "hazards": [{"score": 0.8}],
            "latency_ms": {"perception": 10.0, "memory": 5.0},
        }
        fm = frame_metrics(rec)
        assert fm.frame_id == 5
        assert fm.num_detections == 1
        assert fm.num_hois == 1
        assert fm.max_hazard == 0.8
        assert abs(fm.avg_latency_ms - 7.5) < 1e-6

    def test_empty_record(self) -> None:
        fm = frame_metrics({})
        assert fm.frame_id == -1
        assert fm.num_detections == 0
        assert fm.max_hazard == 0.0
        assert fm.avg_latency_ms == 0.0


# ── Reproducibility ──────────────────────────────────────────────────


class TestReproducibility:
    def _make_records(self, seed: int) -> list[dict]:
        import numpy as np
        rng = np.random.RandomState(seed)
        records = []
        for i in range(50):
            records.append({
                "frame_id": i,
                "hois": [{"action": "cut" if rng.random() > 0.3 else "hold",
                           "confidence": float(rng.uniform(0.5, 1.0))}],
                "hazards": [{"score": float(rng.uniform(0.0, 1.0)), "action": "cut"}]
                if rng.random() > 0.5 else [],
                "attention_allocation": {"obj_0": float(rng.uniform(0.4, 1.0))},
                "detections": [{"confidence": float(rng.uniform(0.5, 1.0))}],
                "latency_ms": {"perception": float(rng.uniform(10, 50))},
            })
        return records

    def test_deterministic_across_calls(self) -> None:
        r1 = self._make_records(42)
        r2 = self._make_records(42)
        sm1 = evaluate_sequence(r1)
        sm2 = evaluate_sequence(r2)
        assert sm1.thc == sm2.thc
        assert sm1.haa == sm2.haa
        assert sm1.rme == sm2.rme
