# ASSR

Multimodal arbitrary-scale super-resolution for Sentinel-1 / Sentinel-2 imagery with:

- scale-conditioned degradation
- dynamic reliability-gated S1/S2 fusion
- LR-only semantic prior (offline caption/tokens)
- RRDB trunk + text/scale conditioning
- PixelShuffle x4 + meta-upsampler (`s in [1.5, 6]`)
- cross-scale consistency training/evaluation

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
    manifest_example.json
  scripts/
    train.py
    evaluate.py
    infer.py
    generate_captions.py
  requirements.txt
```

## Environment

```bash
python -m venv .venv
. .venv/Scripts/activate
python -m pip install -r requirements.txt
```

## Configuration

Default config: `configs/assr_default.yaml`.

## Data Manifest

Each sample is a JSON object. Minimal example:

```json
{
  "id": "sample_0001",
  "s2_hr": "data/s2/sample_0001.png",
  "s1_hr": "data/s1/sample_0001.png",
  "text_tokens": [12, 88, 109, 3]
}
```

Optional fields (already supported by loader):

- `valid_mask`, `cloud_mask`, `coherence_mask`
- `text_embed_path` (e.g., `.npy`) or `text_embed`

Template file: `data/manifest_example.json`.

## Offline Caption Generation

```bash
python scripts/generate_captions.py \
  --manifest data/train_manifest.json \
  --output data/train_manifest_captioned.json \
  --model-id Salesforce/blip-image-captioning-base \
  --image-key s2_lr \
  --max-tokens 16
```

## Entrypoints

```bash
python scripts/train.py --config configs/assr_default.yaml
python scripts/evaluate.py --config configs/assr_default.yaml --checkpoint runs/assr/final_ema.pth
python scripts/infer.py --config configs/assr_default.yaml --checkpoint runs/assr/final_ema.pth --input demo_s2.png --output out.png --scale 4
```

## Notes

- Training scale sampling follows `U[1.5, 6]` with explicit x4 supervision probability.
- Large-image inference uses tiling with Hann blending.
- Scale metrics include SAS/SCE-style projection consistency evaluation.
