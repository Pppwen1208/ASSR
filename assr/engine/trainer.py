from __future__ import annotations

import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from assr.config.schema import ASSRConfig
from assr.data.dataset import ASSRDataset, assr_collate
from assr.engine.ema import EMA
from assr.engine.evaluator import evaluate_model
from assr.losses.adversarial import hinge_d_loss, hinge_g_loss
from assr.losses.perceptual import VGGPerceptualLoss
from assr.losses.reconstruction import ASSRReconstructionLoss
from assr.losses.scheduler import LossScheduler
from assr.models.assr import ASSR
from assr.models.discriminator import SNUNetDiscriminator
from assr.utils.io import ensure_dir, save_checkpoint


def _make_loader(cfg: ASSRConfig, manifest_path: str, training: bool) -> DataLoader:
    ds = ASSRDataset(
        manifest_path=manifest_path,
        data_cfg=cfg.data,
        degradation_cfg=cfg.degradation,
        training=training,
    )
    bs = cfg.train.batch_size if training else 1
    collate_fn = None if training else assr_collate
    return DataLoader(
        ds,
        batch_size=bs,
        shuffle=training,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=training,
        collate_fn=collate_fn,
    )


def _build_scheduler(optimizer: torch.optim.Optimizer, cfg: ASSRConfig) -> LambdaLR:
    total = cfg.train.total_steps
    warmup = cfg.train.warmup_steps

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return float(step + 1) / float(max(warmup, 1))
        progress = (step - warmup) / float(max(total - warmup, 1))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def _to_batch(sample: dict, device: torch.device) -> dict:
    s2_lr = sample["s2_lr"].unsqueeze(0).to(device)
    s2_lr_ref = sample["s2_lr_reff"].unsqueeze(0).to(device)
    s1_lr = sample["s1_lr"]
    if s1_lr is not None:
        s1_lr = s1_lr.unsqueeze(0).to(device)
    hr = sample["s2_hr"].unsqueeze(0).to(device)
    scale = sample["scale"].view(1).to(device)
    text_embed = sample["text_embed"].unsqueeze(0).to(device)
    text_mask = sample["text_mask"].unsqueeze(0).to(device)
    return {
        "s2_lr": s2_lr,
        "s2_lr_ref": s2_lr_ref,
        "s1_lr": s1_lr,
        "s2_hr": hr,
        "scale": scale,
        "text_embed": text_embed,
        "text_mask": text_mask,
        "resize_meta": [sample["resize_meta"]],
    }


def train(cfg: ASSRConfig) -> ASSR:
    device = torch.device(
        cfg.device if cfg.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    out_dir = ensure_dir(cfg.train.out_dir)

    train_loader = _make_loader(cfg, cfg.data.train_manifest, training=True)
    val_loader = _make_loader(cfg, cfg.data.val_manifest, training=False)

    model = ASSR(
        cfg=cfg.model,
        s2_channels=cfg.data.s2_channels,
        s1_channels=cfg.data.s1_channels,
        use_s1=cfg.data.use_s1,
    ).to(device)

    recon_loss = ASSRReconstructionLoss(cfg.train)
    perceptual = VGGPerceptualLoss().to(device)

    optimizer_g = Adam(
        model.parameters(),
        lr=cfg.train.lr,
        betas=tuple(cfg.train.betas),
        weight_decay=cfg.train.weight_decay,
    )
    scheduler_g = _build_scheduler(optimizer_g, cfg)

    use_gan = bool(cfg.train.use_gan)
    disc = None
    optimizer_d = None
    if use_gan:
        disc = SNUNetDiscriminator(in_channels=3).to(device)
        optimizer_d = Adam(
            disc.parameters(),
            lr=cfg.train.lr,
            betas=tuple(cfg.train.betas),
            weight_decay=cfg.train.weight_decay,
        )

    scaler = torch.amp.GradScaler(enabled=cfg.amp and device.type == "cuda")
    ema = EMA(model, decay=cfg.train.ema_decay)
    loss_scheduler = LossScheduler(
        lambda_perc_max=cfg.train.lambda_perc_max,
        lambda_adv_max=cfg.train.lambda_adv_max,
        t0=cfg.train.ramp_start,
        tau=cfg.train.ramp_tau,
    )

    step = 0
    pbar = tqdm(total=cfg.train.total_steps, desc="train")
    while step < cfg.train.total_steps:
        for batch in train_loader:
            if step >= cfg.train.total_steps:
                break
            step += 1
            model.train()
            optimizer_g.zero_grad(set_to_none=True)
            if optimizer_d is not None:
                optimizer_d.zero_grad(set_to_none=True)

            batch_loss = 0.0
            log_pix = 0.0
            log_cons = 0.0

            for sample in batch:
                item = _to_batch(sample, device=device)
                with torch.autocast(
                    device_type=device.type,
                    enabled=cfg.amp and device.type == "cuda",
                ):
                    pred = model(
                        s2_lr=item["s2_lr"],
                        s1_lr=item["s1_lr"],
                        scale=item["scale"],
                        text_embed=item["text_embed"],
                        text_mask=item["text_mask"],
                        resize_meta=item["resize_meta"],
                        enable_risk_gate=False,
                    )["sr"]
                    if pred.shape[-2:] != item["s2_hr"].shape[-2:]:
                        pred = F.interpolate(
                            pred,
                            size=item["s2_hr"].shape[-2:],
                            mode="bicubic",
                            align_corners=False,
                            antialias=True,
                        )

                    pair_pred = None
                    pair_scale = None
                    if cfg.train.pair_scale_samples > 0:
                        pair_scale_val = random.uniform(
                            cfg.data.scale_min, cfg.data.scale_max
                        )
                        pair_scale = torch.tensor(
                            [pair_scale_val], device=device, dtype=item["scale"].dtype
                        )
                        with torch.no_grad():
                            pair_pred = model(
                                s2_lr=item["s2_lr"],
                                s1_lr=item["s1_lr"],
                                scale=pair_scale,
                                text_embed=item["text_embed"],
                                text_mask=item["text_mask"],
                                resize_meta=item["resize_meta"],
                                enable_risk_gate=False,
                            )["sr"]

                    ldict = recon_loss(
                        pred_hr=pred,
                        target_hr=item["s2_hr"],
                        lr_ref=item["s2_lr_ref"],
                        scale=item["scale"],
                        pair_pred_hr=pair_pred,
                        pair_scale=pair_scale,
                    )
                    total = ldict["total"]

                    lambda_perc, lambda_adv = loss_scheduler.weights(
                        step=step,
                        use_perceptual=bool(cfg.train.use_perceptual),
                        use_gan=bool(cfg.train.use_gan),
                    )
                    if lambda_perc > 0:
                        total = total + lambda_perc * perceptual(pred, item["s2_hr"])

                    if use_gan and disc is not None and optimizer_d is not None and lambda_adv > 0:
                        with torch.no_grad():
                            fake_detach = pred.detach()
                        d_real = disc(item["s2_hr"])
                        d_fake = disc(fake_detach)
                        d_loss = hinge_d_loss(d_real, d_fake)
                        scaler.scale(d_loss / len(batch)).backward(retain_graph=True)

                        g_adv = hinge_g_loss(disc(pred))
                        total = total + lambda_adv * g_adv

                    total = total / len(batch)

                scaler.scale(total).backward()
                batch_loss += float(total.detach().item())
                log_pix += float(ldict["pix"].item()) / len(batch)
                log_cons += float(ldict["consist"].item()) / len(batch)

            scaler.unscale_(optimizer_g)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            scaler.step(optimizer_g)
            if optimizer_d is not None:
                scaler.step(optimizer_d)
            scaler.update()
            scheduler_g.step()
            ema.update(model)

            if step % cfg.train.log_every == 0:
                lr_now = optimizer_g.param_groups[0]["lr"]
                pbar.set_postfix(
                    {
                        "loss": f"{batch_loss:.4f}",
                        "pix": f"{log_pix:.4f}",
                        "cons": f"{log_cons:.4f}",
                        "lr": f"{lr_now:.2e}",
                    }
                )

            if step % cfg.train.val_every == 0:
                metrics = evaluate_model(
                    model=ema.shadow,
                    loader=val_loader,
                    device=device,
                    use_amp=cfg.amp,
                    max_batches=32,
                )
                print(
                    f"[val@{step}] "
                    + ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
                )

            if step % cfg.train.save_every == 0:
                ckpt = {
                    "step": step,
                    "model": model.state_dict(),
                    "ema": ema.shadow.state_dict(),
                    "opt_g": optimizer_g.state_dict(),
                    "sched_g": scheduler_g.state_dict(),
                    "cfg": cfg.__dict__,
                }
                if disc is not None and optimizer_d is not None:
                    ckpt["disc"] = disc.state_dict()
                    ckpt["opt_d"] = optimizer_d.state_dict()
                save_checkpoint(out_dir / f"step_{step:07d}.pth", ckpt)

            pbar.update(1)

    pbar.close()
    save_checkpoint(Path(cfg.train.out_dir) / "final_ema.pth", {"model": ema.shadow.state_dict()})
    return ema.shadow
