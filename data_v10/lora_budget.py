"""LoRA storage-budget calculator for GaussianFormer.

Reads only the safetensors header of a checkpoint (no weights loaded), lists every
2-D weight (nn.Linear candidates), and prints the adapter size for a few
rank / target-set choices next to the byte cost of a Gaussian splat file, so the
break-even rule "adapter + small splat < big splat" can be evaluated in bytes.

Usage: uv run --no-sync python data_v10/lora_budget.py [model.safetensors]
"""
import json
import re
import struct
import sys
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints_v18_256/gaussianformer_final/model.safetensors"
with open(path, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(n))

dtype_bytes = {"F32": 4, "F16": 2, "BF16": 2}
total_params = 0
total_bytes = 0
linears: list[tuple[str, int, int]] = []
for name, meta in header.items():
    if name == "__metadata__":
        continue
    shape = meta["shape"]
    cnt = 1
    for s in shape:
        cnt *= s
    total_params += cnt
    total_bytes += cnt * dtype_bytes[meta["dtype"]]
    if len(shape) == 2 and name.endswith("weight"):
        linears.append((name, shape[0], shape[1]))  # (out, in)

print(f"checkpoint: {path}")
print(f"total params: {total_params/1e6:.1f} M, on-disk {total_bytes/1e6:.1f} MB, dtypes {set(m['dtype'] for k,m in header.items() if k!='__metadata__')}")
print(f"2-D weights (Linear candidates): {len(linears)}, "
      f"{sum(o*i for _,o,i in linears)/1e6:.1f} M params\n")

# group by (module-kind, shape) so the table is readable
groups: dict[tuple[str, int, int], int] = defaultdict(int)
for name, o, i in linears:
    kind = re.sub(r"\.\d+\.", ".N.", name)
    groups[(kind, o, i)] += 1
print(f"{'module (N = layer idx)':70s} {'out':>6s} {'in':>6s} {'#':>3s} {'params/each':>12s}")
for (kind, o, i), c in sorted(groups.items(), key=lambda kv: -kv[0][1] * kv[0][2] * kv[1]):
    print(f"{kind:70s} {o:6d} {i:6d} {c:3d} {o*i:12,d}")

# ---- target sets ----
def sel(pattern: str) -> list[tuple[str, int, int]]:
    return [t for t in linears if re.search(pattern, t[0])]

targets = {
    "attn qkv+out, both stages": sel(r"(attn|attention).*?(qkv|q_proj|k_proj|v_proj|to_q|to_k|to_v|out_proj|proj|o_proj|in_proj)"),
    "all Linear, both stages": linears,
}
# fall back: if the regex matched nothing, show it so we fix the pattern
for k, v in targets.items():
    print(f"\n[{k}] matched {len(v)} weights, {sum(o*i for _,o,i in v)/1e6:.1f} M base params")

# ---- splat costs (from files on disk: h5 = 14 fp32/Gaussian + 5 KB hdr; ply = 17 fp32 (adds normals)) ----
H5_PER_G, PLY_PER_G = 56, 68
print("\nsplat file cost (fp32):")
for N in (20_000, 10_000, 5_000, 2_500):
    print(f"  N={N:6d}: h5 {N*H5_PER_G/1e3:8.0f} KB   ply {N*PLY_PER_G/1e3:8.0f} KB   "
          f"fp16-h5 {N*H5_PER_G/2/1e3:8.0f} KB")
for N in (10_000, 5_000, 2_500):
    print(f"  budget freed by 20k -> {N:5d}: h5 {(20_000-N)*H5_PER_G/1e3:6.0f} KB (fp16: {(20_000-N)*H5_PER_G/2/1e3:4.0f} KB), "
          f"ply {(20_000-N)*PLY_PER_G/1e3:6.0f} KB")

print("\nadapter size = r * sum(in+out) over targets (A and B), no bias:")
print(f"{'target set':32s} {'r':>3s} {'params':>12s} {'fp32 KB':>9s} {'fp16 KB':>9s}  fits 20k->5k (h5 fp32 / fp16)?")
for k, v in targets.items():
    s = sum(o + i for _, o, i in v)
    for r in (1, 2, 4, 8, 16, 32):
        p = r * s
        f32, f16 = p * 4 / 1e3, p * 2 / 1e3
        print(f"{k:32s} {r:3d} {p:12,d} {f32:9.0f} {f16:9.0f}  "
              f"{'yes' if f32 < 15_000*H5_PER_G/1e3 else 'NO ':3s} / {'yes' if f16 < 15_000*H5_PER_G/2/1e3 else 'NO'}")
