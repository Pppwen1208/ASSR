# ASSR

Multimodal arbitrary-scale super-resolution for Sentinel-1 / Sentinel-2 imagery.

Core components:
- Scale-conditioned degradation
- Dynamic reliability-gated S1/S2 fusion
- LR-only semantic prior (offline captions/tokens)
- RRDB trunk + text/scale conditioning
- PixelShuffle x4 + meta-upsampler (`s in [1.5, 6]`)
- Cross-scale consistency training/evaluation

## Repository Layout

```text
ASSR/
  assr/                 # core package
    config/             # config schema + loader
    data/               # dataset + degradation + text token adapter
    engine/             # training/evaluation loops
    inference/          # tiled inference
    losses/             # reconstruction/perceptual/adversarial losses
    metrics/            # image + scale metrics
    models/             # ASSR backbone, attention, upsampler, discriminator
    utils/              # io/random/misc helpers
  configs/
    assr_default.yaml   # default experiment config
  data/
    train_manifest.json
    val_manifest.json
    test_manifest.json
    split.json          # paper-aligned split protocol metadata
  weights/
    assr.pth            # place your checkpoint here
  scripts/
    train.py
    train.sh
    train.ps1
    evaluate.py
    infer.py
    generate_captions.py
  requirements.txt
```

## Environment and Dependencies

Recommended:
- Python 3.10+
- CUDA-capable PyTorch runtime (optional but recommended for training)

Install:

```bash
python -m venv .venv
. .venv/Scripts/activate
python -m pip install -r requirements.txt
```

Dependencies from `requirements.txt`:
- Core runtime: `numpy`, `Pillow`, `PyYAML`, `tqdm`
- Deep learning: `torch`, `torchvision`
- Metrics/vision: `lpips`, `opencv-python-headless`, `scikit-image`
- Offline captions: `transformers`, `accelerate`

## Configuration

Default training/inference config: `configs/assr_default.yaml`

Important fields:
- Data manifests: `data.train_manifest`, `data.val_manifest`, `data.test_manifest`
- Scale range: `data.scale_min`, `data.scale_max`
- Training outputs: `train.out_dir`
- Inference tile settings: `infer.tile_size`, `infer.hr_overlap`

Section 3.5 training-related hyperparameter mapping:
- Optimizer Adam with `beta1=0.9`, `beta2=0.999`:
  `train.betas` in `configs/assr_default.yaml`
- Warm-up steps:
  `train.warmup_steps` in `configs/assr_default.yaml`
- Gradient clipping:
  `train.grad_clip` in `configs/assr_default.yaml`
- EMA decay rate:
  `train.ema_decay` in `configs/assr_default.yaml`
- Cosine LR decay:
  implemented in `assr/engine/trainer.py::_build_scheduler` (cosine schedule after warm-up)

Main entry for paper-default training remains:
- config: `configs/assr_default.yaml`
- launcher: `scripts/train.py`

## Data Preparation

### 1) Prepare split files

Required files:
- `data/train_manifest.json`
- `data/val_manifest.json`
- `data/test_manifest.json`
- `data/split.json`

`data/split.json` stores split protocol metadata aligned with the paper text.

### 2) Prepare manifest entries

Each manifest is a JSON list. Each item is one sample.

Minimal sample:

```json
{
  "id": "sample_0001",
  "s2_hr": "data/s2/sample_0001.png",
  "s1_hr": "data/s1/sample_0001.png",
  "text_tokens": [12, 88, 109, 3]
}
```

Optional supported fields:
- `valid_mask`, `cloud_mask`, `coherence_mask`
- `text_embed_path` (e.g., `.npy`) or `text_embed`

### 3) Optional: offline caption generation

```bash
python scripts/generate_captions.py \
  --manifest data/train_manifest.json \
  --output data/train_manifest_captioned.json \
  --image-key s2_lr
```

The caption script is protocol-fixed (Table A2 style):
- model: `Salesforce/blip-image-captioning-base`
- prompt: `Describe this image in one short sentence.`
- decoding: `do_sample=False`, `num_beams=3`, `max_new_tokens=16`
- image preprocessing: RGB + resize to `384x384`
- post-processing: lowercase + whitespace stripping
- fallback caption: `a low-resolution satellite image`

## Weights

Put your checkpoint at:
- `weights/assr.pth`

## Workflow

### Training

```bash
python scripts/train.py --config configs/assr_default.yaml
```

One-command wrappers (same logic, same config):

```bash
bash scripts/train.sh
```

```powershell
powershell -ExecutionPolicy Bypass -File scripts/train.ps1
```

### Evaluation

```bash
python scripts/evaluate.py --config configs/assr_default.yaml --checkpoint weights/assr.pth
```

Default metrics include:
- `PSNR`, `SSIM`, `LPIPS`, `Edge-F1`
- Scale-consistency metrics including `SCE` (default enabled)

Useful options:
- `--max-batches N`
- `--skip-scale-metrics`
- `--scales "1.5,2,3,4,5.5,6"`
- `--save-json runs/assr/eval_metrics.json`

### Inference

```bash
python scripts/infer.py --config configs/assr_default.yaml --checkpoint weights/assr.pth --input demo_s2.png --output out.png --scale 4
```

Optional inputs/outputs:
- `--s1 <path>`: aligned S1 LR input
- `--text <json/npy/or_token_ids>`: semantic input
- `--risk-output <path>`: save risk map
- `--enable-risk-gate --risk-gate-strength 0.35`

## Notes

- Training scale sampling follows `U[1.5, 6]` with explicit x4 supervision probability.
- Large-image inference uses tiling with Hann blending.
- Scale metrics include SAS/SCE-style projection consistency evaluation.
