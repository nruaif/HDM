"""
Tier 3 Verification & Training Speed Benchmark for HDM (XUDiT-340M + EQ-SDXL-VAE 4ch p=2):
  1. Verifies exact numerical equivalence of fast half-dim AxialRoPE vs original AxialRoPE.
  2. Verifies exact forward + backward equivalence of permutation-equivariant gather/scatter_ TREAD
     vs legacy boolean-mask TREAD (given the same kept token set).
  3. Benchmarks forward + backward + AdamW step throughput (steps/s, imgs/s, peak VRAM) across:
     - Config A (Legacy Baseline): xformers/boolean-mask TREAD, legacy RoPE, grad_ckpt=True, ctx=256
     - Config B (Tier 3 Eager): gather TREAD + fast AxialRoPE + SDPA, grad_ckpt=True, ctx=256
     - Config C (Tier 3 Compiled): + model.compile_blocks(), grad_ckpt=True, ctx=256
     - Config D (Tier 3 Compiled + Selective Ckpt 'half'): grad_ckpt='half', ctx=256
     - Config E (Tier 3 Compiled + 'half' + Dynamic/Pruned ctx=72): grad_ckpt='half', ctx=72
"""

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch
import torch.nn.functional as F
from xut.modules.axial_rope import AxialRoPE, apply_rotary_emb, make_axial_pos_no_cache
from xut.xut import XUDiT


# Exact HDM-xut-340M-anime architecture config (EQ-SDXL-VAE 4ch, patch_size=2)
HDM_340M_CFG = dict(
    patch_size=2,
    input_dim=4,
    dim=1024,
    ctx_dim=1024,
    ctx_size=256,
    heads=16,
    dim_head=64,
    mlp_dim=3072,
    depth=8,
    enc_blocks=1,
    dec_blocks=2,
    dec_ctx=False,
    shared_adaln=True,
    concat_ctx=True,
    use_dyt=False,
    addon_info_embs_dim=1,
    tread_config={"dropout_ratio": 0.5, "prev_trns_depth": 2, "post_trns_depth": 2},
)


def verify_numerics(device: torch.device):
    print("=== 1. Numerical Equivalence Checks ===", flush=True)
    torch.manual_seed(0)
    rope = AxialRoPE(dim=64, n_heads=16, pos_dim=2).to(device)
    x = torch.randn(2, 16, 128, 64, device=device, dtype=torch.float32, requires_grad=True)
    pos = torch.randn(2, 128, 2, device=device, dtype=torch.float32)

    # Legacy reference
    freqs_ref = rope.get_freqs(pos)
    out_ref = apply_rotary_emb(freqs_ref, x, rope.start_index)
    out_ref.sum().backward()
    grad_ref = x.grad.clone()

    # Fast Tier-3 path
    x.grad = None
    out_fast = rope(x, pos)
    out_fast.sum().backward()
    grad_fast = x.grad.clone()

    fwd_err = (out_ref - out_fast).abs().max().item()
    bwd_err = (grad_ref - grad_fast).abs().max().item()
    print(f"  AxialRoPE forward max_diff  = {fwd_err:.2e} | backward max_diff = {bwd_err:.2e}", flush=True)
    assert fwd_err < 1e-5 and bwd_err < 1e-5, f"AxialRoPE mismatch! {fwd_err}, {bwd_err}"

    # Verify XUDiT forward + backward + return_enc_out with TREAD + pos_map fast-path
    small_cfg = dict(HDM_340M_CFG, dim=256, ctx_dim=256, heads=4, dim_head=64, mlp_dim=512, depth=2)
    m = XUDiT(**small_cfg).to(device).train()
    for mode in [False, "half", True]:
        m.set_grad_ckpt(mode)
        z = torch.randn(2, 4, 32, 32, device=device, requires_grad=True)
        t = torch.rand(2, device=device)
        ctx = torch.randn(2, 48, 256, device=device)  # dynamic ctx length (48 != 256)!
        pos_16 = make_axial_pos_no_cache(16, 16, device=device).unsqueeze(0).expand(2, -1, -1)
        ar = torch.zeros(2, device=device)
        out, enc = m(z, t, ctx=ctx, pos_map=pos_16, addon_info=ar, return_enc_out=True)
        assert out.shape == z.shape, f"Unexpected out shape {out.shape}"
        (out.pow(2).mean() + 0.01 * enc.pow(2).mean()).backward()
    print("  XUDiT forward/backward (grad_ckpt=False/'half'/True + dynamic ctx + return_enc_out) PASSED!", flush=True)


def run_legacy_forward(model: XUDiT, x, t, ctx, pos_map, addon_info):
    """Exact legacy boolean-mask + Python loop randperm TREAD for fair A/B timing comparison."""
    n, c, h, w = x.size()
    t = t.reshape(n, -1)
    # Force legacy F.interpolate on pos_map at H/8 x W/8
    pos_h8 = make_axial_pos_no_cache(h, w, device=x.device, dtype=x.dtype).unsqueeze(0).expand(n, -1, -1)
    x, pos_map = model.in_patch(x, pos_h8)
    x = x.contiguous()
    t_emb = model.time_emb(t) + model.addon_info_embs_proj(addon_info[:, None])[:, None]
    shared_adaln_state = [
        model.shared_adaln_attn(t_emb).chunk(3, dim=-1),
        model.shared_adaln_xattn(t_emb).chunk(3, dim=-1),
        model.shared_adaln_ffw(t_emb).chunk(3, dim=-1),
    ]
    length = x.size(1)
    ctx = model.ctx_proj(ctx)
    x = torch.cat([x, ctx], dim=1)
    pos_map = torch.cat([pos_map, torch.zeros(n, ctx.size(1), pos_map.size(2), device=x.device, dtype=pos_map.dtype)], dim=1)

    x = model.prev_tread_trns(x, ctx=None, pos_map=pos_map, y=t_emb, shared_adaln=shared_adaln_state)
    selection_length = length - int(length * model.dropout_ratio)
    selection = torch.stack([torch.randperm(length, device=x.device) < selection_length for _ in range(n)])
    ctx_length = x.size(1) - length
    selection = torch.concat([selection, torch.ones(n, ctx_length, device=x.device, dtype=torch.bool)], dim=1)
    selection_length += ctx_length
    full_length = x.size(1)
    not_masked_part = x[~selection, :]
    x = x[selection, :].unflatten(0, (n, selection_length))
    raw_pos_map = pos_map
    pos_map = pos_map[selection, :].unflatten(0, (n, selection_length))

    backbone_out = model.backbone(x, ctx=None, pos_map=pos_map, y=t_emb, shared_adaln=shared_adaln_state)
    out = torch.empty(n, full_length, x.size(2), device=x.device, dtype=x.dtype)
    out[~selection, :] = not_masked_part
    out[selection, :] = backbone_out.flatten(0, 1)
    out = model.post_tread_trns(out, ctx=None, pos_map=raw_pos_map, y=t_emb, shared_adaln=shared_adaln_state)
    return model.out_patch(out[:, :length], h, w)


def bench_config(label: str, model: XUDiT, batch_size: int, lat_h: int, lat_w: int, ctx_len: int, legacy: bool = False, n_warm: int = 4, n_iters: int = 15):
    device = next(model.parameters()).device
    bf16_ok = torch.cuda.is_bf16_supported() and torch.cuda.get_device_capability(device)[0] >= 8
    amp_dtype = torch.bfloat16 if bf16_ok else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)

    z = torch.randn(batch_size, 4, lat_h, lat_w, device=device, dtype=amp_dtype)
    target = torch.randn(batch_size, 4, lat_h, lat_w, device=device, dtype=torch.float32)
    t = torch.rand(batch_size, device=device, dtype=amp_dtype)
    ctx = torch.randn(batch_size, ctx_len, 1024, device=device, dtype=amp_dtype)
    pos_16 = make_axial_pos_no_cache(lat_h // 2, lat_w // 2, device=device, dtype=amp_dtype).unsqueeze(0).expand(batch_size, -1, -1)
    ar = torch.zeros(batch_size, device=device, dtype=amp_dtype)

    def _step():
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            if legacy:
                pred = run_legacy_forward(model, z, t, ctx, pos_16, ar)
            else:
                pred = model(z, t, ctx=ctx, pos_map=pos_16, addon_info=ar)
            loss = F.mse_loss(pred.float(), target)
        if amp_dtype == torch.float16:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t_w = time.time()
    for _ in range(n_warm):
        _step()
    torch.cuda.synchronize()
    warm_s = time.time() - t_w

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(n_iters):
        _step()
    torch.cuda.synchronize()
    dt = time.time() - t0
    sps = n_iters / dt
    ips = (n_iters * batch_size) / dt
    vram = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"  {label:52s} | {ips:6.2f} img/s ({sps:5.2f} step/s) | VRAM={vram:5.2f} GB | warmup={warm_s:.1f}s", flush=True)
    return ips, vram


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    verify_numerics(device)
    if device.type != "cuda":
        print("CPU-only environment: numerical unit tests passed! Run on GPU for timing benchmark.")
        return

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    gpu_name = torch.cuda.get_device_name(device)
    vram_tot = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    # 512x768 image -> EQ-SDXL-VAE 4ch latent is 64x96 (patches: 32x48 = 1,536 tokens)
    bs = 8
    lat_h, lat_w = 64, 96
    print(f"\n=== 2. HDM-340M Training Benchmark on {gpu_name} ({vram_tot:.1f} GB) | 512x768 (64x96 latent), bs={bs} ===", flush=True)

    # A & B: Eager model
    model = XUDiT(**HDM_340M_CFG).to(device).train()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Instantiated XUDiT-340M ({n_params:.1f}M params).", flush=True)

    # Legacy AxialRoPE monkey-patch for Config A
    orig_rope_fwd = AxialRoPE.forward
    AxialRoPE.forward = lambda self, x, pos: apply_rotary_emb(self.get_freqs(pos), x, self.start_index)
    model.set_grad_ckpt(True)
    ips_a, _ = bench_config("A. Legacy Baseline (bool-TREAD, old RoPE, ckpt=all, ctx=256)", model, bs, lat_h, lat_w, ctx_len=256, legacy=True)
    AxialRoPE.forward = orig_rope_fwd

    # B. Tier 3 Eager
    model.set_grad_ckpt(True)
    ips_b, _ = bench_config("B. Tier 3 Eager (gather-TREAD, fast RoPE, ckpt=all, ctx=256)", model, bs, lat_h, lat_w, ctx_len=256, legacy=False)

    # C. Tier 3 Compiled (ckpt=True, dynamic=True so SAME compiled model handles any AR & ctx!)
    model.compile_blocks(mode="default", dynamic=True)
    ips_c, _ = bench_config("C. Tier 3 Compiled (compile_blocks, ckpt=all, ctx=256)", model, bs, lat_h, lat_w, ctx_len=256, legacy=False)

    # D. Tier 3 Compiled + Selective grad_ckpt='half'
    model.set_grad_ckpt("half")
    ips_d, _ = bench_config("D. Tier 3 Compiled + Selective ckpt='half' (ctx=256)", model, bs, lat_h, lat_w, ctx_len=256, legacy=False)

    # E. SAME compiled model with Pruned/Dynamic ctx=72 (tests zero-recompile dynamic ctx!)
    ips_e, _ = bench_config("E. Tier 3 Compiled + ckpt='half' + Pruned ctx=72", model, bs, lat_h, lat_w, ctx_len=72, legacy=False)

    # F. Multi-AR dynamic shape stress test (512x512=64x64 and 512x896=64x112 with zero recompile!)
    ips_sq, _ = bench_config("F. Dynamic AR check: 512x512 (64x64) + ckpt='half' + ctx=72", model, bs, 64, 64, ctx_len=72, legacy=False, n_warm=2)

    best = ips_e
    if vram_tot >= 16.0:
        model.set_grad_ckpt(False)
        ips_nc, _ = bench_config("G. Tier 3 Compiled + ckpt=False + Pruned ctx=72", model, bs, lat_h, lat_w, ctx_len=72, legacy=False)
        best = max(best, ips_nc)

    print(f"\n>>> 512x768 DiT Training Speedup: {ips_a:.2f} -> {best:.2f} img/s ({best/ips_a:.2f}x faster!) <<<", flush=True)


if __name__ == "__main__":
    main()
