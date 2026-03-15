from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DataConfig:
    train_manifest: str = "data/train_manifest.json"
    val_manifest: str = "data/val_manifest.json"
    test_manifest: str = "data/test_manifest.json"
    hr_crop_size: int = 256
    scale_min: float = 1.5
    scale_max: float = 6.0
    use_s1: bool = True
    s2_channels: int = 3
    s1_channels: int = 1
    max_text_len: int = 16
    pixelshuffle_supervision_prob: float = 0.2


@dataclass
class DegradationConfig:
    kernel_size: int = 7
    sigma_min: float = 0.6
    sigma_max: float = 1.8
    anisotropy_min: float = 1.0
    anisotropy_max: float = 1.6
    shot_noise_max: float = 0.02
    read_noise_max: float = 0.01
    compression_prob: float = 0.3
    compression_quality_min: int = 85
    compression_quality_max: int = 100
    usm_amount: float = 0.2
    usm_sigma: float = 1.0


@dataclass
class ModelConfig:
    channels: int = 64
    rrdb_blocks: int = 36
    growth_channels: int = 32
    num_heads: int = 8
    attn_dropout: float = 0.1
    token_dropout: float = 0.1
    max_text_len: int = 16
    text_embed_dim: int = 768
    adaln_eps: float = 1e-6
    fourier_bands: int = 8
    meta_kernel_size: int = 3
    ff_expand: int = 2
    rgb_out_channels: int = 3
    pixelshuffle_scale: int = 4


@dataclass
class TrainConfig:
    batch_size: int = 4
    total_steps: int = 200000
    warmup_steps: int = 3000
    lr: float = 2e-4
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    log_every: int = 50
    val_every: int = 2000
    save_every: int = 5000
    out_dir: str = "runs/assr"
    lambda_consist: float = 0.2
    lambda_lr: float = 1.0
    lambda_pair: float = 0.2
    use_perceptual: bool = False
    use_gan: bool = False
    lambda_perc_max: float = 0.005
    lambda_adv_max: float = 0.0025
    ramp_start: int = 120000
    ramp_tau: int = 20000
    pair_scale_samples: int = 1


@dataclass
class InferConfig:
    tile_size: int = 256
    hr_overlap: int = 16
    batch_size: int = 1
    auto_mixed_precision: bool = True


@dataclass
class ASSRConfig:
    seed: int = 3407
    device: str = "cuda"
    num_workers: int = 4
    amp: bool = True
    data: DataConfig = field(default_factory=DataConfig)
    degradation: DegradationConfig = field(default_factory=DegradationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    infer: InferConfig = field(default_factory=InferConfig)

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> "ASSRConfig":
        return ASSRConfig(
            seed=raw.get("seed", 3407),
            device=raw.get("device", "cuda"),
            num_workers=raw.get("num_workers", 4),
            amp=raw.get("amp", True),
            data=DataConfig(**raw.get("data", {})),
            degradation=DegradationConfig(**raw.get("degradation", {})),
            model=ModelConfig(**raw.get("model", {})),
            train=TrainConfig(**raw.get("train", {})),
            infer=InferConfig(**raw.get("infer", {})),
        )
