#!/usr/bin/env python3
"""Predictive HOI training with raw/preprocessed dataset modes, QAT support."""

from __future__ import annotations

import argparse
import logging
import json
from pathlib import Path

import torch
from torch.utils.data import random_split

from risksense_vla.config import load_config
from risksense_vla.hoi import (
    HICODetRawDataset,
    HOIGenRawDataset,
    PredictiveHOINet,
    PredictiveHOIModule,
    TemporalHOIPreprocessedDataset,
    build_hoi_dataloader,
    evaluate_predictive_hoi,
    save_predictive_hoi_checkpoint,
    train_predictive_hoi,
)
from risksense_vla.runtime import pick_backend
from risksense_vla.train import apply_qat

_LOG = logging.getLogger(__name__)


def _default_actions(cfg: dict) -> list[str]:
    hoi_cfg = cfg.get("hoi", {}) if isinstance(cfg, dict) else {}
    if isinstance(hoi_cfg, dict):
        actions = hoi_cfg.get("actions")
        if isinstance(actions, list) and actions:
            return [str(x) for x in actions]
    return PredictiveHOIModule().action_vocab


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--backend-config", default="configs/backend_mps.yaml")
    p.add_argument("--dataset-mode", choices=["raw", "preprocessed"], default="preprocessed")
    p.add_argument("--dataset-name", choices=["hoigen", "hico"], default="hoigen")
    p.add_argument("--annotation-json", default=None, help="Raw HOIGen/HICO annotation JSON path.")
    p.add_argument("--preprocessed-jsonl", default=None, help="Temporal JSONL path.")
    p.add_argument("--val-preprocessed-jsonl", default=None, help="Optional held-out validation JSONL path.")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--horizon-seconds", type=int, default=None)
    p.add_argument("--output", default="trained_models/hoi/checkpoint.pt")
    return p.parse_args()


def _build_dataset(args: argparse.Namespace, action_vocab: list[str], emb_dim: int, horizon: int):
    if args.dataset_mode == "preprocessed":
        if not args.preprocessed_jsonl:
            raise ValueError("--preprocessed-jsonl is required when --dataset-mode=preprocessed.")
        train_ds = TemporalHOIPreprocessedDataset(
            args.preprocessed_jsonl,
            action_vocab=action_vocab,
            emb_dim=emb_dim,
            horizon_seconds=horizon,
        )
        if args.val_preprocessed_jsonl:
            val_ds = TemporalHOIPreprocessedDataset(
                args.val_preprocessed_jsonl,
                action_vocab=action_vocab,
                emb_dim=emb_dim,
                horizon_seconds=horizon,
            )
            return train_ds, val_ds
        return train_ds, None

    if not args.annotation_json:
        raise ValueError("--annotation-json is required when --dataset-mode=raw.")
    if args.dataset_name == "hoigen":
        dataset = HOIGenRawDataset(
            args.annotation_json, action_vocab=action_vocab, emb_dim=emb_dim, horizon_seconds=horizon
        )
    else:
        dataset = HICODetRawDataset(
            args.annotation_json, action_vocab=action_vocab, emb_dim=emb_dim, horizon_seconds=horizon
        )
    return dataset, None


def _split_dataset(dataset, val_split: float, seed: int):
    n = len(dataset)
    if n < 2 or val_split <= 0:
        return dataset, dataset
    val_n = max(1, int(round(n * val_split)))
    train_n = max(1, n - val_n)
    if train_n + val_n > n:
        val_n = n - train_n
    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [train_n, val_n], generator=gen)
    return train_ds, val_ds


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    cfg = load_config(args.config, args.backend_config)
    backend = pick_backend(cfg.get("runtime", {}).get("backend", "mps"))
    device = torch.device(backend.device)
    emb_dim = int(cfg.get("perception", {}).get("embedding_dim", 256))
    horizon = int(args.horizon_seconds or cfg.get("hazard", {}).get("future_horizon_seconds", 3))
    horizon = max(1, min(3, horizon))
    action_vocab = _default_actions(cfg)

    dataset, explicit_val = _build_dataset(args, action_vocab, emb_dim, horizon)
    if len(dataset) == 0:
        raise ValueError("Training dataset is empty after parsing.")
    if explicit_val is None:
        train_ds, val_ds = _split_dataset(dataset, val_split=args.val_split, seed=args.seed)
    else:
        train_ds, val_ds = dataset, explicit_val

    train_loader = build_hoi_dataloader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = build_hoi_dataloader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = PredictiveHOINet(emb_dim=emb_dim, num_actions=len(action_vocab), horizon_seconds=horizon).to(device)
    model = apply_qat(model, cfg)

    use_amp = bool(cfg.get("runtime", {}).get("mixed_precision", True)) and backend.device != "cpu"
    hist = train_predictive_hoi(
        model,
        train_loader,
        epochs=args.epochs,
        lr=args.learning_rate,
        device=str(device),
        use_amp=use_amp,
    )
    metrics = evaluate_predictive_hoi(model, val_loader, device=str(device))
    _LOG.info(
        "[hoi] final loss=%.4f current_top1=%.4f future_top1=%.4f future_emb_cos=%.4f",
        hist["loss"][-1],
        metrics["current_top1"],
        metrics["future_top1"],
        metrics["future_embedding_cosine"],
    )

    out = Path(args.output)
    meta = {
        "config": cfg,
        "dataset_mode": args.dataset_mode,
        "dataset_name": args.dataset_name,
        "metrics": metrics,
        "history": hist,
        "horizon_seconds": horizon,
    }
    save_predictive_hoi_checkpoint(out, model, action_vocab=action_vocab, extra=meta)

    log_path = out.parent / "train_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _LOG.info("Saved %s", out)


if __name__ == "__main__":
    main()
