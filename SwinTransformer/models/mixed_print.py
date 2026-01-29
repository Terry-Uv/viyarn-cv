import re
import argparse
import torch
import numpy as np

def get_state_dict(sd):
    return sd.get("state_dict", sd.get("model", sd))

def extract_wx_wy(F: torch.Tensor):
    """
    从任意包含 size==2 的维度里提取 (wx, wy)。
    常见：
      - (2, H, C)
      - (H, C, 2)
      - (..., 2, ...)
    """
    if not isinstance(F, torch.Tensor) or F.dim() < 2:
        return None, None

    if F.size(0) == 2:
        return F[0], F[1]

    for axis in range(F.dim()):
        if F.size(axis) == 2:
            return F.select(axis, 0), F.select(axis, 1)

    return None, None

def rho_from_freqs(freqs: torch.Tensor, eps=1e-12):
    wx, wy = extract_wx_wy(freqs)
    if wx is None:
        return None
    wx = wx.detach().float()
    wy = wy.detach().float()
    rho = torch.log(torch.sqrt(wx * wx + wy * wy + eps))
    return rho.reshape(-1).cpu().numpy()

def parse_stage(key: str):
    # timm/microsoft swin 常见：layers.{stage}.blocks.{i}.attn.rope_freqs
    m = re.search(r"\blayers\.(\d+)\b", key)
    if m:
        return int(m.group(1))
    # 兼容其它命名
    m = re.search(r"\bstages\.(\d+)\b", key)
    if m:
        return int(m.group(1))
    return None

def quantiles(arr, qs):
    return {q: float(np.quantile(arr, q)) for q in qs}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--key_regex", default=r"(rope_freqs$|\.rope_freqs$|^freqs$|\.freqs$)")
    ap.add_argument("--per_key", action="store_true")
    ap.add_argument("--per_stage", action="store_true", help="print per-stage stats (recommended for Swin)")
    ap.add_argument("--lo_q", type=float, default=0.50, help="recommend rho_lo quantile (default 0.50)")
    ap.add_argument("--hi_q", type=float, default=0.90, help="recommend rho_hi quantile (default 0.90)")
    args = ap.parse_args()

    sd = torch.load(args.ckpt, map_location="cpu")
    state = get_state_dict(sd)

    pat = re.compile(args.key_regex)

    all_rho = []
    by_stage = {}   # stage -> list of rho arrays
    by_key = []     # (key, rho_array)

    matched = 0
    used = 0

    for k, v in state.items():
        if not pat.search(k):
            continue
        matched += 1
        rho = rho_from_freqs(v)
        if rho is None or rho.size == 0:
            continue
        used += 1
        all_rho.append(rho)
        by_key.append((k, rho))

        st = parse_stage(k)
        if st is not None:
            by_stage.setdefault(st, []).append(rho)

    if not all_rho:
        print(f"[analyze] matched_keys={matched}, used_keys={used}. No usable freqs found. "
              f"Check key_regex or tensor shapes.")
        return

    all_rho = np.concatenate(all_rho, axis=0)

    qs_print = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.98]
    qv = quantiles(all_rho, qs_print)

    print("\n========== Swin mixed rho stats (GLOBAL) ==========")
    print(f"numel={all_rho.size} | matched_keys={matched} used_keys={used}")
    print("quantiles:", {q: qv[q] for q in qs_print})

    rho_lo = float(np.quantile(all_rho, args.lo_q))
    rho_hi = float(np.quantile(all_rho, args.hi_q))
    print(f"\n[RECOMMEND] rho_lo=q{args.lo_q:.2f}={rho_lo:+.6f}, rho_hi=q{args.hi_q:.2f}={rho_hi:+.6f}")
    print(f"[ALT] rho_hi=q0.75={qv[0.75]:+.6f} (more conservative)")

    if args.per_stage and by_stage:
        print("\n========== Swin mixed rho stats (PER STAGE) ==========")
        for st in sorted(by_stage.keys()):
            arr = np.concatenate(by_stage[st], axis=0)
            qst = quantiles(arr, [0.50, 0.75, 0.90, 0.95])
            rec_lo = float(np.quantile(arr, args.lo_q))
            rec_hi = float(np.quantile(arr, args.hi_q))
            print(f"stage={st} | n={arr.size:9d} | "
                  f"q50={qst[0.50]:+.4f} q75={qst[0.75]:+.4f} q90={qst[0.90]:+.4f} q95={qst[0.95]:+.4f} | "
                  f"RECOMM lo={rec_lo:+.4f} hi={rec_hi:+.4f}")

    if args.per_key:
        print("\n========== Swin mixed rho stats (PER KEY) ==========")
        # 按 q90 从大到小，看哪些模块高频尾更重
        def q(arr, qq): return float(np.quantile(arr, qq))
        by_key.sort(key=lambda kv: q(kv[1], 0.90), reverse=True)
        for k, arr in by_key:
            print(f"{k:95s} | n={arr.size:7d} | "
                  f"min={arr.min():+.4f} q50={q(arr,0.50):+.4f} q75={q(arr,0.75):+.4f} q90={q(arr,0.90):+.4f} max={arr.max():+.4f}")

if __name__ == "__main__":
    main()

'''
python analyze_swin_mixed_rho_ckpt.py --ckpt /path/to/swin_mixed_ckpt.pth --per_stage

看每个block分布
python analyze_swin_mixed_rho_ckpt.py --ckpt /path/to/swin_mixed_ckpt.pth --per_stage --per_key
'''