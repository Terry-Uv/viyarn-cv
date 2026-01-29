# analyze_mixed_rho_ckpt.py
import re
import argparse
import torch
import numpy as np

def _extract_wx_wy(F: torch.Tensor):
    """
    Accepts tensor that stores (wx, wy) along some axis of size 2.
    Returns wx, wy as torch.Tensor, or (None, None) if not recognized.
    """
    if not isinstance(F, torch.Tensor):
        return None, None
    if F.dim() < 2:
        return None, None

    # Case A: (2, ...)
    if F.size(0) == 2:
        return F[0], F[1]

    # Case B: (..., 2, ...)
    for axis in range(F.dim()):
        if F.size(axis) == 2:
            wx = F.select(axis, 0)
            wy = F.select(axis, 1)
            return wx, wy

    return None, None

def _rho_from_wx_wy(wx, wy, eps=1e-12):
    norm = torch.sqrt(wx * wx + wy * wy + eps)
    rho = torch.log(norm)
    return rho

def _quantiles_np(arr, qs):
    return {q: float(np.quantile(arr, q)) for q in qs}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to checkpoint (.pth/.pt/.bin)")
    ap.add_argument("--key_regex", default=r"(rope_freqs|\.freqs$|^freqs$)", help="regex to match keys")
    ap.add_argument("--per_key", action="store_true", help="print quantiles per matched key")
    ap.add_argument("--recommend_lo", type=float, default=0.50, help="recommend rho_lo as this quantile (default q50)")
    ap.add_argument("--recommend_hi", type=float, default=0.90, help="recommend rho_hi as this quantile (default q90)")
    args = ap.parse_args()

    sd = torch.load(args.ckpt, map_location="cpu")
    state = sd.get("state_dict", sd.get("model", sd))

    key_pat = re.compile(args.key_regex)

    all_rho = []
    per_key_rho = []  # (key, rho_flat_np)

    for k, v in state.items():
        if not key_pat.search(k):
            continue

        wx, wy = _extract_wx_wy(v)
        if wx is None:
            continue

        rho = _rho_from_wx_wy(wx.detach().float(), wy.detach().float()).reshape(-1).cpu().numpy()
        if rho.size == 0:
            continue

        all_rho.append(rho)
        per_key_rho.append((k, rho))

    if not all_rho:
        print(f"[mixed-rho] No matched keys by regex={args.key_regex}, or shapes not recognized.")
        return

    all_rho = np.concatenate(all_rho, axis=0)

    qs_print = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.98]
    qv = _quantiles_np(all_rho, qs_print)

    print("\n========== Mixed rho stats (GLOBAL) ==========")
    print(f"numel={all_rho.size}")
    print("quantiles:", {q: qv[q] for q in qs_print})
    lo_q = float(args.recommend_lo)
    hi_q = float(args.recommend_hi)
    rho_lo = float(np.quantile(all_rho, lo_q))
    rho_hi = float(np.quantile(all_rho, hi_q))
    print(f"\n[RECOMMEND] rho_lo=q{lo_q:.2f}={rho_lo:+.6f}, rho_hi=q{hi_q:.2f}={rho_hi:+.6f}")
    print(f"[ALT] rho_hi=q0.75={qv[0.75]:+.6f} (more conservative)")

    if args.per_key:
        print("\n========== Mixed rho stats (PER KEY) ==========")
        # Sort by q90 descending to see which modules have higher-frequency tails
        def key_q(rho_np, q):
            return float(np.quantile(rho_np, q))
        per_key_rho.sort(key=lambda kv: key_q(kv[1], 0.90), reverse=True)

        for k, rho_np in per_key_rho:
            q50 = key_q(rho_np, 0.50)
            q75 = key_q(rho_np, 0.75)
            q90 = key_q(rho_np, 0.90)
            print(f"{k:90s} | n={rho_np.size:7d} | "
                  f"min={rho_np.min():+.4f} q50={q50:+.4f} q75={q75:+.4f} q90={q90:+.4f} max={rho_np.max():+.4f}")

if __name__ == "__main__":
    main()

# python analyze_mixed_rho_ckpt.py --ckpt /home/zhouweixian/yarn-vit/deit/official_ckpt/rope_mixed_small.bin --per_key
'''
python analyze_mixed_rho_ckpt.py --ckpt /path/to/rope_mixed_base.bin
python analyze_mixed_rho_ckpt.py --ckpt /path/to/rope_mixed_large.bin
'''