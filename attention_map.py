import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ── compatibility shim ────────────────────────────────────────────────────────
try:
    from huggingface_hub import cached_download  # noqa: F401
except ImportError:
    import huggingface_hub as _hfhub
    from huggingface_hub import hf_hub_download as _dl

    def _cached_download(url_or_filename=None, *args, **kwargs):
        repo_id   = kwargs.pop("repo_id",   None)
        filename  = kwargs.pop("filename",  None)
        cache_dir = kwargs.pop("cache_dir", None)
        if repo_id and filename:
            return _dl(repo_id=repo_id, filename=filename, cache_dir=cache_dir, **kwargs)
        return url_or_filename

    _hfhub.cached_download = _cached_download
    import huggingface_hub
    huggingface_hub.cached_download = _cached_download
# ─────────────────────────────────────────────────────────────────────────────

from diffusers import StableDiffusionImg2ImgPipeline
import daam

MODEL_ID       = "runwayml/stable-diffusion-v1-5"
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE          = torch.float16 if DEVICE == "cuda" else torch.float32
STRENGTH       = 0.4
GUIDANCE_SCALE = 7.5
NUM_STEPS      = 30
SEED           = 42
OUTPUT_DIR     = Path("outputs")

PROMPTS = {
    "trustworthy": "a portrait photo of a trustworthy person",
    "smart":       "a portrait photo of a smart person",
    "dominant":    "a portrait photo of a dominant person",
}


def load_image(path: str, size: int = 512) -> Image.Image:
    return Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)


def make_overlay(base_img: Image.Image, heatmap: np.ndarray,
                 alpha: float = 0.55) -> Image.Image:
    hm = np.array(
        Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
            base_img.size, Image.BILINEAR)
    ) / 255.0
    colored = (cm.jet(hm)[:, :, :3] * 255).astype(np.uint8)
    return Image.blend(base_img, Image.fromarray(colored), alpha)


def main(image_path: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    input_img = load_image(image_path)

    print(f"\nLoading pipeline on {DEVICE} ({DTYPE}) …")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        MODEL_ID, torch_dtype=DTYPE, safety_checker=None,
    ).to(DEVICE)
    pipe.set_progress_bar_config(disable=False)

    generator = torch.Generator(device=DEVICE).manual_seed(SEED)

    for concept, prompt in PROMPTS.items():
        print(f"\n{'─'*60}\nConcept : {concept}\nPrompt  : {prompt}")
        generator.manual_seed(SEED)

        with daam.trace(pipe) as tc:
            pipe(
                prompt=prompt,
                image=input_img,
                strength=STRENGTH,
                guidance_scale=GUIDANCE_SCALE,
                num_inference_steps=NUM_STEPS,
                generator=generator,
            )
            global_hm = tc.compute_global_heat_map()
            word_hm   = global_hm.compute_word_heat_map(concept)

        heatmap_np = word_hm.heatmap.cpu().float().numpy()
        hmin, hmax = heatmap_np.min(), heatmap_np.max()
        if hmax > hmin:
            heatmap_np = (heatmap_np - hmin) / (hmax - hmin)

        overlay = make_overlay(input_img, heatmap_np)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(input_img); axes[0].set_title("Input image",           fontsize=13)
        axes[1].imshow(overlay);   axes[1].set_title(f'"{concept}" attention', fontsize=13)
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        out_path = OUTPUT_DIR / f"{concept}_comparison.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved → {out_path}")

    print(f"\nDone. Outputs in '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to input portrait image")
    args = parser.parse_args()
    if not os.path.isfile(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")
    main(args.image)
