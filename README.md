# DAAM Attention Heatmap Visualization Demo

Visualizes **cross-attention heatmaps** for abstract human attributes on a portrait image using [DAAM] (https://github.com/castorini/daam).

Given one input portrait, the script runs three prompts — *trustworthy*, *smart*, *dominant* — and produces side-by-side comparisons showing where in the image each concept is spatially attended to.

> This project was fully generated via claude code agent. No manual code was written.

---

## How it works
'''
Input image
    │
    ▼
Encode to latent  ──►  Add noise (strength=0.4)
                              │
                              ▼
                   Denoising loop (30 steps)
                   guided by text prompt
                   ┌──────────────────────┐
                   │  DAAM traces every   │
                   │  cross-attention op  │
                   └──────────────────────┘
                              │
                              ▼
              compute_global_heat_map()
              compute_word_heat_map(concept)
                              │
                              ▼
              Normalize → jet colormap → blend
                              │
                              ▼
              outputs/{concept}_comparison.png
```

---

## Project structure

```
home/
├── attention_map.py      # Main script
├── requirements.txt      # Pinned dependencies
├── face_example.jpg      # Example input portrait
└── outputs/              # Generated results (created at runtime)
    ├── trustworthy_comparison.png
    ├── smart_comparison.png
    └── dominant_comparison.png
```

### `requirements.txt`

Exact package versions from the verified working `att_map` conda environment. Pinned because `daam==0.2.0` has a hard dependency on `diffusers==0.21.2`, which in turn requires older versions of `transformers` and `huggingface_hub`.

---

## Setup

### 1. Create conda environment

```bash
conda create -n your_env python=3.10 -y
conda activate your_env
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `huggingface_hub` in this environment is newer than what `diffusers==0.21.2` expects. The compatibility shim in `attention_map.py` handles this automatically — no manual patching needed.

### 3. (First run) Download the model

The script downloads `runwayml/stable-diffusion-v1-5` (~4 GB) from Hugging Face on first run. To set a custom cache directory:

```bash
export HF_HOME=/path/to/your/cache
```

---

## Usage

```bash
conda activate att_map
python attention_map.py /path/to/portrait.jpg
```

Example:

```bash
python attention_map.py face_example.jpg
```

---

## Output

Three side-by-side comparison images are saved under `outputs/`:

| File | Content |
|---|---|
| `trustworthy_comparison.png` | Input image vs. "trustworthy" attention heatmap |
| `smart_comparison.png` | Input image vs. "smart" attention heatmap |
| `dominant_comparison.png` | Input image vs. "dominant" attention heatmap |

Each image shows the original portrait on the left and the jet-colormap attention overlay on the right. Warmer colors (red/yellow) indicate regions the model attends to most strongly for that concept.

---

## Key parameters

All parameters are defined at the top of `attention_map.py` and can be changed directly:

| Parameter | Default | Description |
|---|---|---|
| `STRENGTH` | `0.4` | img2img noise strength. Lower = closer to original image |
| `NUM_STEPS` | `30` | Denoising steps. More steps = more attention data collected |
| `GUIDANCE_SCALE` | `7.5` | How strongly the prompt guides generation |
| `SEED` | `42` | Fixed seed ensures all three concepts use identical noise |
| `MODEL_ID` | `runwayml/stable-diffusion-v1-5` | Can be replaced with a local model path |

---

## Dependencies

| Package | Version | Role |
|---|---|---|
| `torch` | 2.10.0 | Deep learning backend |
| `diffusers` | 0.21.2 | Stable Diffusion pipeline |
| `daam` | 0.2.0 | Cross-attention tracing |
| `transformers` | 4.30.2 | Text encoder (CLIP) |
| `accelerate` | 0.23.0 | Device/dtype management |
| `huggingface_hub` | 0.36.2 | Model downloading |

---

## Acknowledgements

- **DAAM** — Cross-attention tracing is built on [castorini/daam](https://github.com/castorini/daam).
- **Example image** — `face_example.jpg` is sourced from the [OMI face dataset](https://github.com/jcpeterson/omi). Please refer to the original dataset's license for usage terms.

---

## References

- Tang, Raphael, et al. "What the daam: Interpreting stable diffusion using cross attention." *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*. 2023.
- Peterson, Joshua C., et al. "Deep models of superficial face judgments." *Proceedings of the National Academy of Sciences* 119.17 (2022): e2115228119.
