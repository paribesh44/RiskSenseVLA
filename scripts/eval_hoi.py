#!/usr/bin/env python3
"""Evaluate predictive HOI logs for current and 1-3s future accuracy."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

_LOG = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pred-log-jsonl", default="outputs/hoi_inference.jsonl")
    p.add_argument("--gt-jsonl", default=None, help="Optional ground-truth JSONL for action/embedding evaluation.")
    p.add_argument("--report-json", default="outputs/hoi_eval.json")
    return p.parse_args()


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing JSONL: {path}")
    records: list[dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _text_embedding(text: str, emb_dim: int = 256) -> torch.Tensor:
    seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
    g = torch.Generator()
    g.manual_seed(seed)
    emb = torch.randn((emb_dim,), generator=g, dtype=torch.float32)
    return emb / (torch.linalg.norm(emb) + 1e-8)


def _extract_gt_actions(gt_records: list[dict[str, Any]]) -> dict[int, list[str]]:
    actions_by_frame: dict[int, list[str]] = defaultdict(list)
    for rec in gt_records:
        if "frame_id" in rec:
            frame_id = int(rec.get("frame_id", -1))
            hois = rec.get("hois", rec.get("hoi_current", []))
            for hoi in hois if isinstance(hois, list) else []:
                if isinstance(hoi, dict):
                    action = str(hoi.get("action", "")).strip()
                    if action:
                        actions_by_frame[frame_id].append(action)
            continue
        hois = rec.get("hois", [])
        for hoi in hois if isinstance(hois, list) else []:
            if not isinstance(hoi, dict):
                continue
            frame_idx = int(hoi.get("frame_idx", rec.get("start_frame", -1)))
            action = str(hoi.get("action", "")).strip()
            if action:
                actions_by_frame[frame_idx].append(action)
    return actions_by_frame


def _top_current_action(pred_record: dict[str, Any]) -> str:
    hois = pred_record.get("hoi_current", [])
    if not isinstance(hois, list) or not hois:
        return ""
    best = max(hois, key=lambda x: float(x.get("confidence", 0.0)))
    return str(best.get("action", ""))


def _top_future_actions(pred_record: dict[str, Any]) -> dict[int, str]:
    labels = pred_record.get("hoi_future_action_labels", [])
    if not isinstance(labels, list) or not labels:
        return {}
    horizon = max((len(row) for row in labels if isinstance(row, list)), default=0)
    out: dict[int, str] = {}
    for step in range(horizon):
        vals = []
        for row in labels:
            if isinstance(row, list) and step < len(row):
                label = str(row[step]).strip()
                if label:
                    vals.append(label)
        if vals:
            out[step + 1] = Counter(vals).most_common(1)[0][0]
    return out


def _future_embedding_cosine(
    pred_record: dict[str, Any],
    gt_actions_by_frame: dict[int, list[str]],
) -> dict[int, float]:
    frame_id = int(pred_record.get("frame_id", -1))
    raw = pred_record.get("hoi_future_embeddings", [])
    if not isinstance(raw, list) or not raw:
        return {}
    pred = torch.as_tensor(raw, dtype=torch.float32)
    if pred.ndim != 3 or pred.shape[1] == 0:
        return {}
    out: dict[int, float] = {}
    for step in range(pred.shape[1]):
        gt_actions = gt_actions_by_frame.get(frame_id + step + 1, [])
        if not gt_actions:
            continue
        gt_vec = _text_embedding(gt_actions[0], emb_dim=pred.shape[2])
        pred_step = pred[:, step, :]
        pred_step = pred_step / (torch.linalg.norm(pred_step, dim=1, keepdim=True) + 1e-8)
        cos = torch.matmul(pred_step, gt_vec)
        out[step + 1] = float(cos.mean().item())
    return out


def evaluate(pred_records: list[dict[str, Any]], gt_records: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    latencies = [float(r.get("latency_ms", {}).get("total", 0.0)) for r in pred_records]
    avg_latency = _safe_mean(latencies)
    fps = (1000.0 / avg_latency) if avg_latency > 0 else 0.0
    report: dict[str, Any] = {
        "frames": len(pred_records),
        "latency_ms": avg_latency,
        "fps": fps,
        "avg_current_hoi_per_frame": _safe_mean(
            [float(len(r.get("hoi_current", []))) for r in pred_records]
        ),
    }
    if not gt_records:
        return report

    gt_actions = _extract_gt_actions(gt_records)
    current_hits = 0
    current_total = 0
    future_hits_by_h: dict[int, int] = defaultdict(int)
    future_total_by_h: dict[int, int] = defaultdict(int)
    cosine_by_h: dict[int, list[float]] = defaultdict(list)

    for rec in pred_records:
        frame_id = int(rec.get("frame_id", -1))
        top_cur = _top_current_action(rec)
        if top_cur:
            gt_now = set(gt_actions.get(frame_id, []))
            if gt_now:
                current_total += 1
                if top_cur in gt_now:
                    current_hits += 1

        for h, pred_action in _top_future_actions(rec).items():
            gt_f = set(gt_actions.get(frame_id + h, []))
            if gt_f:
                future_total_by_h[h] += 1
                if pred_action in gt_f:
                    future_hits_by_h[h] += 1

        cos = _future_embedding_cosine(rec, gt_actions)
        for h, value in cos.items():
            cosine_by_h[h].append(value)

    report["current_action_top1"] = float(current_hits / current_total) if current_total else 0.0
    report["future_action_top1_by_horizon"] = {
        str(h): float(future_hits_by_h[h] / future_total_by_h[h]) if future_total_by_h[h] else 0.0
        for h in sorted(set(future_total_by_h) | set(future_hits_by_h))
    }
    report["future_embedding_cosine_by_horizon"] = {
        str(h): _safe_mean(vals) for h, vals in sorted(cosine_by_h.items(), key=lambda x: x[0])
    }
    return report


def main() -> None:
    args = parse_args()
    pred_records = _load_jsonl(args.pred_log_jsonl)
    gt_records = _load_jsonl(args.gt_jsonl) if args.gt_jsonl else None
    report = evaluate(pred_records, gt_records)
    out = Path(args.report_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _LOG.info("%s", json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
