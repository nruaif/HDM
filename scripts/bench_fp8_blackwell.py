"""
Benchmark & visual comparison on RTX PRO 4500 Blackwell:
  Row 1: BF16 (10 steps, current baseline with vae.enable_slicing())
  Row 2: FP8 (torchao Float8DynamicActivationFloat8WeightConfig) + batched VAE + 10 steps
  Row 3: FP8 + batched VAE + 8 steps
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from diffusers import DiffusionPipeline
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


def _convert_anima_comfy_transformer_sd(raw_sd):
    stripped = {
        (k[len("model.diffusion_model."):] if k.startswith("model.diffusion_model.") else k): v
        for k, v in raw_sd.items()
    }
    from diffusers.loaders.single_file_utils import convert_cosmos_transformer_checkpoint_to_diffusers

    converted = convert_cosmos_transformer_checkpoint_to_diffusers(stripped)
    return {
        (k[len("model.diffusion_model."):] if k.startswith("model.diffusion_model.") else k): v
        for k, v in converted.items()
    }


def load_pipe(use_fp8: bool = False, use_slicing: bool = False):
    pipe = DiffusionPipeline.from_pretrained(
        "CalamitousFelicitousness/Anima-1.0-Aesthetic-Diffusers",
        custom_pipeline="CalamitousFelicitousness/Anima-1.0-Aesthetic-Diffusers",
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    ckpt = hf_hub_download(
        "circlestone-labs/Anima",
        "split_files/diffusion_models/anima-turbo-v1.1.safetensors",
    )
    raw_sd = load_file(ckpt, device="cpu")
    adapter_sd, trns_raw = {}, {}
    for k, v in raw_sd.items():
        clean = k[len("model.diffusion_model."):] if k.startswith("model.diffusion_model.") else k
        if clean.startswith("llm_adapter."):
            adapter_sd[clean[len("llm_adapter."):]] = v
        else:
            trns_raw[k] = v
    pipe.llm_adapter.load_state_dict(adapter_sd, strict=False)
    pipe.transformer.load_state_dict(_convert_anima_comfy_transformer_sd(trns_raw), strict=False)
    pipe.to("cuda", dtype=torch.bfloat16)

    if use_slicing and hasattr(pipe.vae, "enable_slicing"):
        pipe.vae.enable_slicing()
    else:
        if hasattr(pipe.vae, "disable_slicing"):
            pipe.vae.disable_slicing()
        # Decode in chunks of 6 (fits easily in 32GB VRAM, much faster than slice=1)
        orig_decode = pipe.vae.decode
        def _chunked_decode(z, return_dict=True, **kw):
            if z.shape[0] <= 6:
                return orig_decode(z, return_dict=return_dict, **kw)
            outs = [orig_decode(z[i:i+6], return_dict=False, **kw)[0] for i in range(0, z.shape[0], 6)]
            cat = torch.cat(outs, dim=0)
            if not return_dict:
                return (cat,)
            from diffusers.models.autoencoders.vae import DecoderOutput
            return DecoderOutput(sample=cat)
        pipe.vae.decode = _chunked_decode

    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)

    if use_fp8:
        from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, quantize_
        quantize_(pipe.transformer, Float8DynamicActivationFloat8WeightConfig())

    for i, blk in enumerate(pipe.transformer.transformer_blocks):
        pipe.transformer.transformer_blocks[i] = torch.compile(blk, mode="default", dynamic=True)

    return pipe


def encode_webp_batch(imgs):
    out = []
    for img in imgs:
        buf = BytesIO()
        img.save(buf, "WEBP", quality=92)
        out.append(buf.getvalue())
    return out


def bench_and_sample(pipe, prompts_24, steps: int, async_webp: bool, label: str):
    w, h = 512, 768
    bs = len(prompts_24)
    torch.cuda.reset_peak_memory_stats()

    # Warmup (triggers compile)
    t_warm = time.time()
    with torch.inference_mode():
        gens = [torch.Generator(device="cuda").manual_seed(1000 + i) for i in range(bs)]
        sample_imgs = pipe(
            prompt=prompts_24,
            width=w,
            height=h,
            num_inference_steps=steps,
            guidance_scale=1.0,
            generator=gens,
        ).images
    torch.cuda.synchronize()
    warm_s = time.time() - t_warm

    # Timed: 3 batches of 24 = 72 images
    n_batches = 3
    pool = ThreadPoolExecutor(max_workers=4) if async_webp else None
    futs = []
    t0 = time.time()
    with torch.inference_mode():
        for b in range(n_batches):
            gens = [torch.Generator(device="cuda").manual_seed(2000 + b * 100 + i) for i in range(bs)]
            imgs = pipe(
                prompt=prompts_24,
                width=w,
                height=h,
                num_inference_steps=steps,
                guidance_scale=1.0,
                generator=gens,
            ).images
            if async_webp:
                futs.append(pool.submit(encode_webp_batch, imgs))
            else:
                encode_webp_batch(imgs)
    torch.cuda.synchronize()
    if pool:
        for f in futs:
            f.result()
        pool.shutdown()
    dt = time.time() - t0
    ips = (n_batches * bs) / dt
    peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"[{label}] warmup={warm_s:.1f}s | 72 imgs in {dt:.2f}s -> {ips:.2f} img/s | peak VRAM={peak_gb:.2f} GB", flush=True)
    return sample_imgs[:4], ips, peak_gb


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    # Load 24 real prompts from shard-0020
    import sys
    sys.path.insert(0, "/workspace")
    from runpod_gen_anima_wds import _format_anima_prompt

    manifest = Path("/workspace/anima_wds/manifests/shard-0020.jsonl")
    rows = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines()[:24]]
    prompts_24 = [_format_anima_prompt(r, aesthetic_mode=True) for r in rows]

    results = []

    # 1. BF16 baseline (10 steps, slicing, sync WebP)
    pipe = load_pipe(use_fp8=False, use_slicing=True)
    imgs_bf16, ips_bf16, vram_bf16 = bench_and_sample(pipe, prompts_24, steps=10, async_webp=False, label="BF16 10-step (baseline)")
    results.append((f"BF16 10-step | {ips_bf16:.2f} img/s | {vram_bf16:.1f}GB", imgs_bf16))
    del pipe
    torch.cuda.empty_cache()

    # 2. FP8 + batched VAE + async WebP (10 steps)
    pipe = load_pipe(use_fp8=True, use_slicing=False)
    imgs_fp8_10, ips_fp8_10, vram_fp8_10 = bench_and_sample(pipe, prompts_24, steps=10, async_webp=True, label="FP8 10-step + #3")
    results.append((f"FP8 10-step + #3 | {ips_fp8_10:.2f} img/s | {vram_fp8_10:.1f}GB", imgs_fp8_10))

    # 3. FP8 + batched VAE + async WebP (8 steps)
    imgs_fp8_8, ips_fp8_8, vram_fp8_8 = bench_and_sample(pipe, prompts_24, steps=8, async_webp=True, label="FP8 8-step + #3")
    results.append((f"FP8 8-step + #3 | {ips_fp8_8:.2f} img/s | {vram_fp8_8:.1f}GB", imgs_fp8_8))

    # Build 3x4 comparison grid
    cell_w, cell_h = 384, 576
    hdr = 36
    grid = Image.new("RGB", (cell_w * 4, (cell_h + hdr) * 3), (20, 20, 24))
    draw = ImageDraw.Draw(grid)
    for r_idx, (title, imgs) in enumerate(results):
        y0 = r_idx * (cell_h + hdr)
        draw.text((12, y0 + 10), title, fill=(255, 255, 255))
        for c_idx, img in enumerate(imgs):
            thumb = img.resize((cell_w, cell_h), Image.LANCZOS)
            grid.paste(thumb, (c_idx * cell_w, y0 + hdr))

    out_path = "/workspace/fp8_compare_grid.jpg"
    grid.save(out_path, "JPEG", quality=92)
    print(f"Saved comparison grid to {out_path}", flush=True)


if __name__ == "__main__":
    main()
