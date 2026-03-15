from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from assr.config import load_config
from assr.data.dataset import ASSRDataset, assr_collate
from assr.engine.evaluator import evaluate_model, evaluate_scale_stability
from assr.models.assr import ASSR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument(
        "--scales",
        type=str,
        default="",
        help="Optional scale sweep for SAS/SCE, e.g. 1.5,2,3,4,5.5,6",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    device = torch.device(
        cfg.device if cfg.device == "cpu" or torch.cuda.is_available() else "cpu"
    )

    model = ASSR(
        cfg=cfg.model,
        s2_channels=cfg.data.s2_channels,
        s1_channels=cfg.data.s1_channels,
        use_s1=cfg.data.use_s1,
    ).to(device)
    state = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=False)
    model.eval()

    ds = ASSRDataset(
        manifest_path=cfg.data.test_manifest,
        data_cfg=cfg.data,
        degradation_cfg=cfg.degradation,
        training=False,
    )
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=assr_collate,
    )
    metrics = evaluate_model(
        model=model,
        loader=loader,
        device=device,
        use_amp=cfg.amp,
        max_batches=args.max_batches if args.max_batches > 0 else None,
    )

    scales = []
    if args.scales.strip():
        scales = [float(x.strip()) for x in args.scales.split(",") if x.strip()]
    if len(scales) > 0:
        sas_sce = evaluate_scale_stability(
            model=model,
            loader=loader,
            scales=scales,
            device=device,
            use_amp=cfg.amp,
            max_batches=args.max_batches if args.max_batches > 0 else None,
        )
        metrics.update(sas_sce)

    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")


if __name__ == "__main__":
    main()
