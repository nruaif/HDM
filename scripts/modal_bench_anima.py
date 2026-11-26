"""
Parallel Modal GPU price/performance benchmark for Anima-Turbo v1.1 (~512^2, batch=24).

Spawns L40S, RTX-PRO-6000, and H100 concurrently, benchmarks:
  1. BF16 eager (bs=24)
  2. BF16 per-block torch.compile (bs=24)
  3. FP8 dynamic per-block torch.compile (bs=24)

Prints a ranked leaderboard by img/s, img/hr, img/$, and projected images per $30.

Setup (one-time):
    pip install modal   # or: uv tool install modal
    modal setup         # authenticates your Modal account

Run:
    modal run HDM/scripts/modal_bench_anima.py
    modal run HDM/scripts/modal_bench_anima.py --batch-size 24 --steps 10 --fp8
"""

import modal

app = modal.App("anima-turbo-gpu-bench")

hf_cache = modal.Volume.from_name("anima-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "torchvision",
        "diffusers>=0.37.0",
        "transformers>=4.52.0",
        "accelerate",
        "safetensors",
        "huggingface_hub",
        "pillow",
        "sentencepiece",
        "torchao",
    )
    .env({"HF_HOME": "/cache/hf"})
)

# Modal base GPU $/hr + ~$0.15/hr container CPU/RAM overhead
GPU_RATES_USD_HR = {
    "L40S": 1.95 + 0.15,
    "RTX-PRO-6000": 3.03 + 0.15,
    "H100": 3.95 + 0.15,
}

BENCH_PROMPT = (
    "year 2024, newest, masterpiece, best quality, highres, safe, 1girl, solo, "
    "konpaku youmu, touhou, silver hair, green checkered shirt, black skirt, katana, looking at viewer"
)


def _convert_anima_comfy_transformer_sd(raw_sd):
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


def _bench_impl(gpu_label: str, batch_size: int, width: int, height: int, steps: int, cfg: float, run_fp8: bool):
    import gc
    import time
    import torch
    from diffusers import DiffusionPipeline
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    gpu_name = torch.cuda.get_device_name(0)
    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    rate = GPU_RATES_USD_HR[gpu_label]
    print(f"[{gpu_label}] Device: {gpu_name} ({total_vram:.1f} GB) | torch {torch.__version__}")

    def load_pipe():
        pipe = DiffusionPipeline.from_pretrained(
            "CalamitousFelicitousness/Anima-1.0-Aesthetic-Diffusers",
            custom_pipeline="CalamitousFelicitousness/Anima-1.0-Aesthetic-Diffusers",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
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
        if hasattr(pipe.vae, "enable_slicing"):
            pipe.vae.enable_slicing()
        pipe.set_progress_bar_config(disable=True)
        hf_cache.commit()
        return pipe

    def measure(pipe, mode: str, bs: int, repeats: int = 2):
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()
        prompts = [BENCH_PROMPT] * bs
        kwargs = dict(prompt=prompts, width=width, height=height, num_inference_steps=steps, guidance_scale=cfg)
        try:
            with torch.inference_mode():
                t_w = time.time()
                _ = pipe(**kwargs).images
                torch.cuda.synchronize()
                warm_s = time.time() - t_w

                t0 = time.time()
                for _ in range(repeats):
                    _ = pipe(**kwargs).images
                torch.cuda.synchronize()
                dt = (time.time() - t0) / repeats

            ips = bs / dt
            iph = ips * 3600.0
            ipd = iph / rate
            peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
            print(
                f"[{gpu_label}] {mode:28s} | bs={bs:2d} | {dt:5.2f}s | {ips:5.2f} img/s | "
                f"{ipd:6.0f} img/$ | VRAM={peak_gb:4.1f}GB | warm={warm_s:4.1f}s"
            )
            return {
                "gpu": gpu_label,
                "device": gpu_name,
                "mode": mode,
                "bs": bs,
                "sec_per_batch": round(dt, 2),
                "img_per_sec": round(ips, 2),
                "img_per_hr": round(iph),
                "rate_usd_hr": rate,
                "img_per_usd": round(ipd),
                "img_per_30usd": round(ipd * 30),
                "peak_vram_gb": round(peak_gb, 1),
                "warmup_s": round(warm_s, 1),
                "status": "ok",
            }
        except torch.cuda.OutOfMemoryError:
            print(f"[{gpu_label}] {mode} bs={bs}: OOM")
            return {"gpu": gpu_label, "device": gpu_name, "mode": mode, "bs": bs, "status": "OOM"}

    rows = []
    pipe = load_pipe()

    # 1. BF16 eager
    rows.append(measure(pipe, "BF16 eager", batch_size))

    # 2. BF16 per-block torch.compile
    for i, blk in enumerate(pipe.transformer.transformer_blocks):
        pipe.transformer.transformer_blocks[i] = torch.compile(blk, mode="default", dynamic=True)
    rows.append(measure(pipe, "BF16 per-block compile", batch_size))

    # 3. Optional FP8 dynamic per-block torch.compile
    if run_fp8:
        del pipe
        torch.cuda.empty_cache()
        gc.collect()
        pipe = load_pipe()
        try:
            from torchao.quantization import quantize_, float8_dynamic_activation_float8_weight
            for i, blk in enumerate(pipe.transformer.transformer_blocks):
                quantize_(blk, float8_dynamic_activation_float8_weight())
                pipe.transformer.transformer_blocks[i] = torch.compile(blk, mode="default", dynamic=True)
            rows.append(measure(pipe, "FP8 dynamic per-block compile", batch_size))
        except Exception as e:
            print(f"[{gpu_label}] FP8 skipped: {e}")

    return rows


@app.function(image=image, gpu="L40S", volumes={"/cache/hf": hf_cache}, timeout=900)
def bench_l40s(batch_size: int = 24, width: int = 416, height: int = 624, steps: int = 10, cfg: float = 1.0, fp8: bool = True):
    return _bench_impl("L40S", batch_size, width, height, steps, cfg, fp8)


@app.function(image=image, gpu="RTX-PRO-6000", volumes={"/cache/hf": hf_cache}, timeout=900)
def bench_rtx_pro_6000(batch_size: int = 24, width: int = 416, height: int = 624, steps: int = 10, cfg: float = 1.0, fp8: bool = True):
    return _bench_impl("RTX-PRO-6000", batch_size, width, height, steps, cfg, fp8)


@app.function(image=image, gpu="H100", volumes={"/cache/hf": hf_cache}, timeout=900)
def bench_h100(batch_size: int = 24, width: int = 416, height: int = 624, steps: int = 10, cfg: float = 1.0, fp8: bool = True):
    return _bench_impl("H100", batch_size, width, height, steps, cfg, fp8)


@app.local_entrypoint()
def main(batch_size: int = 24, width: int = 416, height: int = 624, steps: int = 10, cfg: float = 1.0, fp8: bool = True):
    print(f"Spawning parallel Modal benchmark on L40S, RTX-PRO-6000, and H100 (bs={batch_size}, {width}x{height}, {steps} steps, cfg={cfg}, fp8={fp8})...")
    calls = [
        bench_l40s.spawn(batch_size, width, height, steps, cfg, fp8),
        bench_rtx_pro_6000.spawn(batch_size, width, height, steps, cfg, fp8),
        bench_h100.spawn(batch_size, width, height, steps, cfg, fp8),
    ]
    all_rows = []
    for c in calls:
        all_rows.extend(c.get())

    ok_rows = sorted([r for r in all_rows if r.get("status") == "ok"], key=lambda r: -r["img_per_usd"])
    print("\n" + "=" * 106)
    print(f"MODAL PRICE/PERFORMANCE LEADERBOARD (Anima-Turbo v1.1 | {width}x{height} | bs={batch_size} | {steps} steps)")
    print("=" * 106)
    print(f"{'GPU':14s} | {'Mode':29s} | {'img/s':>6s} | {'img/hr':>7s} | {'$/hr':>5s} | {'img/$':>7s} | {'img/$30':>8s} | {'VRAM':>6s}")
    print("-" * 106)
    for r in ok_rows:
        print(
            f"{r['gpu']:14s} | {r['mode']:29s} | {r['img_per_sec']:6.2f} | "
            f"{r['img_per_hr']:7d} | ${r['rate_usd_hr']:4.2f} | {r['img_per_usd']:7d} | "
            f"{r['img_per_30usd']:8d} | {r['peak_vram_gb']:4.1f}GB"
        )
    print("=" * 106)
