"""
Self-contained GPU speed benchmark for Anima-Turbo v1.1 (~512^2) on Colab / Modal / local GPUs.

Tests BF16 eager, BF16 per-block torch.compile, and optional per-block FP8 + compile
across batch sizes (auto-handling OOM), and prints img/s, img/hr, peak VRAM, and
estimated Modal images-per-$ for the detected GPU.

Usage (on any Colab or Modal GPU):
    python bench_anima_gpu.py --steps 10 --cfg 1.0 --batch_sizes 1,4,8,16,32
    python bench_anima_gpu.py --steps 10 --cfg 1.0 --batch_sizes 8,16,32 --fp8 --vae_slicing
"""

import argparse
import gc
import time
from pathlib import Path

import torch
from diffusers import DiffusionPipeline
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

# Base Modal GPU $/hr + ~$0.15/hr container CPU/RAM overhead
MODAL_HOURLY_USD = {
    "T4": 0.59 + 0.15,
    "L4": 0.80 + 0.15,
    "A10G": 1.10 + 0.15,
    "L40S": 1.95 + 0.15,
    "A100-SXM4-40GB": 2.10 + 0.15,
    "A100-PCIE-40GB": 2.10 + 0.15,
    "A100-SXM4-80GB": 2.50 + 0.15,
    "A100 80GB": 2.50 + 0.15,
    "H100": 3.95 + 0.15,
    "H200": 4.54 + 0.15,
    "B200": 6.25 + 0.15,
}

BENCH_PROMPT = (
    "year 2024, newest, masterpiece, best quality, highres, safe, 1girl, solo, "
    "konpaku youmu, touhou, silver hair, green checkered shirt, black skirt, katana, looking at viewer"
)


def lookup_modal_rate(gpu_name: str) -> float | None:
    upper = gpu_name.upper()
    for key, rate in MODAL_HOURLY_USD.items():
        if key.upper() in upper:
            return rate
    if "A100" in upper:
        return (2.50 if "80" in upper else 2.10) + 0.15
    return None


def convert_anima_comfy_transformer_sd(raw_sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    stripped = {
        (k[len("model.diffusion_model."):] if k.startswith("model.diffusion_model.") else k): v
        for k, v in raw_sd.items()
    }
    try:
        from diffusers.loaders.single_file_utils import convert_cosmos_transformer_checkpoint_to_diffusers
        converted = convert_cosmos_transformer_checkpoint_to_diffusers(stripped)
        return {
            (k[len("model.diffusion_model."):] if k.startswith("model.diffusion_model.") else k): v
            for k, v in converted.items()
        }
    except Exception:
        pass

    out = {}
    for k, v in stripped.items():
        if k.startswith("llm_adapter."):
            continue
        k = k.replace("x_embedder.proj.1.", "patch_embed.proj.")
        k = k.replace("t_embedder.1.linear_1.", "time_embed.t_embedder.linear_1.")
        k = k.replace("t_embedder.1.linear_2.", "time_embed.t_embedder.linear_2.")
        k = k.replace("t_embedding_norm.", "time_embed.norm.")
        k = k.replace("final_layer.linear.", "proj_out.")
        k = k.replace("final_layer.adaln_modulation.1.", "norm_out.linear_1.")
        k = k.replace("final_layer.adaln_modulation.2.", "norm_out.linear_2.")
        k = k.replace("blocks.", "transformer_blocks.")
        k = k.replace(".self_attn.q_proj.", ".attn1.to_q.")
        k = k.replace(".self_attn.k_proj.", ".attn1.to_k.")
        k = k.replace(".self_attn.v_proj.", ".attn1.to_v.")
        k = k.replace(".self_attn.output_proj.", ".attn1.to_out.0.")
        k = k.replace(".self_attn.q_norm.", ".attn1.norm_q.")
        k = k.replace(".self_attn.k_norm.", ".attn1.norm_k.")
        k = k.replace(".cross_attn.q_proj.", ".attn2.to_q.")
        k = k.replace(".cross_attn.k_proj.", ".attn2.to_k.")
        k = k.replace(".cross_attn.v_proj.", ".attn2.to_v.")
        k = k.replace(".cross_attn.output_proj.", ".attn2.to_out.0.")
        k = k.replace(".cross_attn.q_norm.", ".attn2.norm_q.")
        k = k.replace(".cross_attn.k_norm.", ".attn2.norm_k.")
        k = k.replace(".adaln_modulation_self_attn.1.", ".norm1.linear_1.")
        k = k.replace(".adaln_modulation_self_attn.2.", ".norm1.linear_2.")
        k = k.replace(".adaln_modulation_cross_attn.1.", ".norm2.linear_1.")
        k = k.replace(".adaln_modulation_cross_attn.2.", ".norm2.linear_2.")
        k = k.replace(".adaln_modulation_mlp.1.", ".norm3.linear_1.")
        k = k.replace(".adaln_modulation_mlp.2.", ".norm3.linear_2.")
        k = k.replace(".mlp.layer1.", ".ff.net.0.proj.")
        k = k.replace(".mlp.layer2.", ".ff.net.2.")
        out[k] = v
    return out


def load_anima_turbo(dtype: torch.dtype, vae_slicing: bool = False):
    pipe = DiffusionPipeline.from_pretrained(
        "CalamitousFelicitousness/Anima-1.0-Aesthetic-Diffusers",
        custom_pipeline="CalamitousFelicitousness/Anima-1.0-Aesthetic-Diffusers",
        trust_remote_code=True,
        torch_dtype=dtype,
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
    m_a, u_a = pipe.llm_adapter.load_state_dict(adapter_sd, strict=False)
    m_t, u_t = pipe.transformer.load_state_dict(convert_anima_comfy_transformer_sd(trns_raw), strict=False)
    assert len(m_a) == 0 and len(m_t) == 0, f"Unexpected missing keys: adapter={len(m_a)}, trns={len(m_t)}"

    pipe.to("cuda", dtype=dtype)
    if vae_slicing and hasattr(pipe.vae, "enable_slicing"):
        pipe.vae.enable_slicing()
    pipe.set_progress_bar_config(disable=True)
    return pipe


def run_one(pipe, bs: int, w: int, h: int, steps: int, cfg: float, repeats: int = 2):
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    prompts = [BENCH_PROMPT] * bs
    kwargs = dict(prompt=prompts, width=w, height=h, num_inference_steps=steps, guidance_scale=cfg)
    if cfg > 1.0:
        kwargs["negative_prompt"] = ["worst quality, low quality, blurry"] * bs

    with torch.inference_mode():
        # Warmup (includes compile if enabled)
        t_w = time.time()
        _ = pipe(**kwargs).images
        torch.cuda.synchronize()
        warmup_s = time.time() - t_w

        # Timed runs
        t0 = time.time()
        last_img = None
        for _ in range(repeats):
            last_img = pipe(**kwargs).images[0]
        torch.cuda.synchronize()
        dt = (time.time() - t0) / repeats

    peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
    return dt, bs / dt, warmup_s, peak_gb, last_img


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark Anima-Turbo v1.1 speed on any GPU")
    p.add_argument("--width", type=int, default=416)
    p.add_argument("--height", type=int, default=624)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--cfg", type=float, default=1.0)
    p.add_argument("--batch_sizes", type=str, default="1,4,8,16,32")
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    p.add_argument("--vae_slicing", action="store_true")
    p.add_argument("--fp8", action="store_true", help="Also benchmark per-block FP8 dynamic + torch.compile")
    p.add_argument("--save_sanity", type=str, default="bench_sanity.jpg")
    return p.parse_args()


def main():
    args = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    gpu_name = torch.cuda.get_device_name(0)
    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    cc = torch.cuda.get_device_capability(0)
    modal_rate = lookup_modal_rate(gpu_name)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]

    print("=" * 84)
    print(f"GPU: {gpu_name} | VRAM: {total_vram:.1f} GB | CC: {cc[0]}.{cc[1]} | torch: {torch.__version__}")
    print(f"SDPA: flash={torch.backends.cuda.flash_sdp_enabled()} cudnn={torch.backends.cuda.cudnn_sdp_enabled()}")
    print(f"Config: Anima-Turbo v1.1 | {args.width}x{args.height} | steps={args.steps} | cfg={args.cfg} | dtype={args.dtype}")
    if modal_rate:
        print(f"Modal est. rate: ${modal_rate:.2f}/hr (incl. ~$0.15/hr CPU/RAM)")
    print("=" * 84)

    pipe = load_anima_turbo(dtype=dtype, vae_slicing=args.vae_slicing)
    results = []
    best = (0.0, None)

    def sweep(mode_label: str, bs_list: list[int]):
        nonlocal best
        for bs in bs_list:
            try:
                dt, ips, warm_s, peak_gb, sample_img = run_one(
                    pipe, bs, args.width, args.height, args.steps, args.cfg, repeats=args.repeats
                )
                iph = ips * 3600
                ipd = (iph / modal_rate) if modal_rate else 0.0
                results.append((mode_label, bs, dt, ips, iph, ipd, peak_gb, warm_s))
                if ips > best[0]:
                    best = (ips, sample_img)
                rate_str = f" | {ipd:6.0f} img/$" if modal_rate else ""
                print(
                    f"{mode_label:28s} | bs={bs:2d} | {dt:5.2f}s | {ips:5.2f} img/s "
                    f"({iph:6.0f} img/hr){rate_str} | VRAM={peak_gb:4.1f}GB | warm={warm_s:4.1f}s"
                )
            except torch.cuda.OutOfMemoryError:
                print(f"{mode_label:28s} | bs={bs:2d} | OOM")
                torch.cuda.empty_cache()
                gc.collect()
                break

    # 1. Eager BF16/FP16
    sweep(f"{args.dtype.upper()} eager", batch_sizes)

    # 2. Per-block torch.compile
    for i, blk in enumerate(pipe.transformer.transformer_blocks):
        pipe.transformer.transformer_blocks[i] = torch.compile(blk, mode="default", dynamic=True)
    sweep(f"{args.dtype.upper()} per-block compile", [b for b in batch_sizes if b >= 4] or batch_sizes)

    # 3. Optional per-block FP8 dynamic + compile
    if args.fp8:
        del pipe
        torch.cuda.empty_cache()
        gc.collect()
        pipe = load_anima_turbo(dtype=dtype, vae_slicing=args.vae_slicing)
        try:
            from torchao.quantization import quantize_, float8_dynamic_activation_float8_weight
            for i, blk in enumerate(pipe.transformer.transformer_blocks):
                quantize_(blk, float8_dynamic_activation_float8_weight())
                pipe.transformer.transformer_blocks[i] = torch.compile(blk, mode="default", dynamic=True)
            sweep("FP8 dynamic per-block compile", [b for b in batch_sizes if b >= 4] or batch_sizes)
        except Exception as e:
            print(f"[skip] FP8 benchmark unavailable: {e}")

    if best[1] is not None and args.save_sanity:
        best[1].save(args.save_sanity, "JPEG", quality=85)
        print(f"\nSaved sanity image from fastest run ({best[0]:.2f} img/s) -> {Path(args.save_sanity).resolve()}")


if __name__ == "__main__":
    main()
