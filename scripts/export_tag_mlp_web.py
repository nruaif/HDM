"""
Export a trained `masked_tag_mlp.pt` checkpoint (v1 untied or v2 tied) into a single
compact binary file (`masked_tag_mlp_web.bin`) for 100% client-side browser inference
in `web_tag_mlp/index.html`.

Binary format (little-endian):
  - uint32 header_len
  - UTF-8 JSON header (padded with spaces to 4-byte alignment) containing:
      version, vocab_size, d_model, d_ff, num_blocks, tied, logit_scale,
      vocab_tags, vocab_cats (0=general, 1=character, 2=copyright),
      tensors: [{name, dtype, shape, offset, nbytes}, ...]
  - Raw binary tensor payload (INT8 per-row quantized vocab weights + FP32 MLP/norm/bias)

Size for v2 (`V=73,626, d=256, tied=True`): ~25 MB raw (~17 MB gzipped).
"""

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import torch


CAT_TO_INT = {"general": 0, "character": 1, "copyright": 2}


def quantize_rows_int8(w: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric per-row INT8 quantization: w_fp32 ~= w_q * scale[:, None]."""
    arr = w.detach().float().cpu().numpy()
    max_abs = np.max(np.abs(arr), axis=1)
    scales = np.where(max_abs > 1e-8, max_abs / 127.0, 1.0).astype(np.float32)
    q = np.clip(np.round(arr / scales[:, None]), -127, 127).astype(np.int8)
    return q, scales


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to masked_tag_mlp.pt")
    parser.add_argument("--out", type=str, default="masked_tag_mlp_web.bin", help="Output .bin path")
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]
    vocab_tags = ckpt["vocab_tags"]
    vocab_cats = [CAT_TO_INT.get(c, 0) for c in ckpt["vocab_cats"]]

    emb_w = sd["emb.weight"]
    head_w = sd["head.weight"]
    V, d_model = emb_w.shape
    tied = bool(torch.equal(emb_w, head_w))
    logit_scale = float(sd["logit_scale"].item()) if "logit_scale" in sd else 1.0

    # Detect num_blocks and d_ff
    block_ids = sorted({int(k.split(".")[1]) for k in sd if k.startswith("blocks.")})
    num_blocks = len(block_ids)
    d_ff = int(sd["blocks.0.fc1.weight"].shape[0])

    print(
        f"Exporting MaskedTagMLP: V={V:,}, d_model={d_model}, d_ff={d_ff}, "
        f"blocks={num_blocks}, tied={tied}, logit_scale={logit_scale:.4f}"
    )

    payload_parts: list[bytes] = []
    tensor_meta: list[dict] = []
    cursor = 0

    def add_tensor(name: str, arr: np.ndarray):
        nonlocal cursor
        arr = np.ascontiguousarray(arr)
        # Align each tensor to 4 bytes so JS TypedArrays (Float32Array) can view directly
        pad = (4 - (cursor % 4)) % 4
        if pad:
            payload_parts.append(b"\x00" * pad)
            cursor += pad
        raw = arr.tobytes()
        tensor_meta.append(
            {
                "name": name,
                "dtype": str(arr.dtype),
                "shape": list(arr.shape),
                "offset": cursor,
                "nbytes": len(raw),
            }
        )
        payload_parts.append(raw)
        cursor += len(raw)

    # 1. Vocab embeddings (INT8 per-row quantized)
    emb_q, emb_s = quantize_rows_int8(emb_w)
    add_tensor("emb_q", emb_q)
    add_tensor("emb_scale", emb_s)
    if not tied:
        head_q, head_s = quantize_rows_int8(head_w)
        add_tensor("head_q", head_q)
        add_tensor("head_scale", head_s)

    # 2. Head bias & LayerNorms & MLP blocks (FP32 for exact accuracy)
    add_tensor("head_bias", sd["head.bias"].detach().float().cpu().numpy())
    add_tensor("in_norm_w", sd["in_norm.weight"].detach().float().cpu().numpy())
    add_tensor("in_norm_b", sd["in_norm.bias"].detach().float().cpu().numpy())
    add_tensor("out_norm_w", sd["out_norm.weight"].detach().float().cpu().numpy())
    add_tensor("out_norm_b", sd["out_norm.bias"].detach().float().cpu().numpy())

    for i in block_ids:
        add_tensor(f"b{i}_norm_w", sd[f"blocks.{i}.norm.weight"].detach().float().cpu().numpy())
        add_tensor(f"b{i}_norm_b", sd[f"blocks.{i}.norm.bias"].detach().float().cpu().numpy())
        add_tensor(f"b{i}_fc1_w", sd[f"blocks.{i}.fc1.weight"].detach().float().cpu().numpy())
        add_tensor(f"b{i}_fc1_b", sd[f"blocks.{i}.fc1.bias"].detach().float().cpu().numpy())
        add_tensor(f"b{i}_fc2_w", sd[f"blocks.{i}.fc2.weight"].detach().float().cpu().numpy())
        add_tensor(f"b{i}_fc2_b", sd[f"blocks.{i}.fc2.bias"].detach().float().cpu().numpy())

    header = {
        "version": 1,
        "vocab_size": V,
        "d_model": d_model,
        "d_ff": d_ff,
        "num_blocks": num_blocks,
        "tied": tied,
        "logit_scale": logit_scale,
        "vocab_tags": vocab_tags,
        "vocab_cats": vocab_cats,
        "tensors": tensor_meta,
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    # Pad header so payload starts on a 4-byte boundary (4-byte header_len + header_bytes)
    hpad = (4 - (len(header_bytes) % 4)) % 4
    if hpad:
        header_bytes += b" " * hpad

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(struct.pack("<I", len(header_bytes)))
        f.write(header_bytes)
        for part in payload_parts:
            f.write(part)

    mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Done! Wrote {out_path} ({mb:.2f} MB) — open web_tag_mlp/index.html and drop this file in.")


if __name__ == "__main__":
    main()
