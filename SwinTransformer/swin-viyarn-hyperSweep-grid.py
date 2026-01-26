#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
swin-viyarn-hyperSweep-grid.py

Purpose
-------
For each hyperparam setting (gamma_lo, gamma_hi, depth_ramp_p, scale_threshold, transition, alpha_max),
evaluate ImageNet-1K val across a fixed set of square input sizes:
  160, 192, 224, 256, 320, 384, 512
with Swin-RoPE window_size following S2:
  window_size = img_size // 32

We print results in *streaming markdown table rows* (one row per size per setting),
and also write a CSV.

IMPORTANT prerequisite patches
------------------------------
1) config.py: add these keys under MODEL.SWIN_ROPE:
   - VIYARN_ENABLE (bool)
   - BASE_WINDOW_SIZE (int)
   - YARN_GAMMA_LO (float)
   - YARN_GAMMA_HI (float)
   - YARN_TRANSITION (str)
   - ANISOTROPIC (bool)   # can keep True; axis-wise can be ignored for square inputs
   - VIYARN_DEPTH_RAMP_P (float)
   - VIYARN_SCALE_THRESHOLD (float)
   - VIYARN_ALPHA_MAX (float)   # recommended (see notes)
2) models/build.py (or your build1.py that is actually imported):
   pass those config values into RoPESwinTransformer(...)
3) swin_transformer_rope_viyarn1.py:
   accept viyarn_alpha_max and use it to set alpha_max (recommended),
   plus optional debug print (rank0 once) to prove args are applied.

Usage
-----
python swin-viyarn-hyperSweep-grid.py \
  --main /workspace/.../SwinTransformer/main.py \
  --cfg  /workspace/.../configs/...yaml \
  --ckpt /workspace/.../official_ckpt/...bin \
  --data /path/to/imagenet1K \
  --out  /workspace/.../result/sweep \
  --gpus 0,1,2,3,4,5,6,7 \
  --nproc 8 \
  --batch 32

Notes
-----
- We run two evals per (setting,size): baseline (VIYARN_ENABLE=False) and yarn (VIYARN_ENABLE=True),
  and report delta = yarn - baseline. This is the cleanest way to isolate ViYaRN benefit.
- We also force TRAIN.AUTO_RESUME=False to avoid accidental "resume hijack" from output folders.

"""
import argparse
import csv
import datetime as dt
import itertools
import os
import random
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

TOP1_PATTERNS = [
    re.compile(r"\*\s*Acc@1\s*([0-9]+\.[0-9]+)"),  # " * Acc@1 81.234 Acc@5 ..."
    re.compile(r"Acc@1\s*([0-9]+\.[0-9]+)\s*\("),   # "Acc@1 81.234 (81.111)"
    re.compile(r"Accuracy of the network on the .*? test images:\s*([0-9]+\.[0-9]+)%"),
]

SIZES_DEFAULT = [160, 192, 224, 256, 320, 384, 512]

def parse_top1(text: str) -> Optional[float]:
    for pat in TOP1_PATTERNS:
        m = pat.search(text)
        if m:
            return float(m.group(1))
    return None

def run_and_capture(cmd, env, log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True, bufsize=1)
    lines = []
    with log_path.open("w", encoding="utf-8") as f:
        for line in p.stdout:
            lines.append(line)
            f.write(line)
    ret = p.wait()
    return ret, "".join(lines)

def next_port(base: int) -> int:
    return int(base) + random.randint(1, 2000)

def make_opts(
    *,
    img: int,
    window_size: int,
    yarn_enable: bool,
    gamma_lo: float,
    gamma_hi: float,
    transition: str,
    depth_ramp_p: float,
    scale_threshold: float,
    alpha_max: float,
    base_window_size: int,
):
    # NOTE: these keys must exist in config.py (YACS), otherwise you will get "KeyError: Non-existent config key".
    return [
        "DATA.IMG_SIZE", str(img),
        "DATA.IMG_SIZE_W", str(img),
        "MODEL.SWIN.WINDOW_SIZE", str(window_size),

        "MODEL.SWIN_ROPE.VIYARN_ENABLE", str(bool(yarn_enable)),
        "MODEL.SWIN_ROPE.BASE_WINDOW_SIZE", str(int(base_window_size)),

        "MODEL.SWIN_ROPE.YARN_GAMMA_LO", str(float(gamma_lo)),
        "MODEL.SWIN_ROPE.YARN_GAMMA_HI", str(float(gamma_hi)),
        "MODEL.SWIN_ROPE.YARN_TRANSITION", str(transition),

        # keep anisotropic True; for square inputs, it doesn't change s_x vs s_y anyway.
        "MODEL.SWIN_ROPE.ANISOTROPIC", "True",

        "MODEL.SWIN_ROPE.VIYARN_DEPTH_RAMP_P", str(float(depth_ramp_p)),
        "MODEL.SWIN_ROPE.VIYARN_SCALE_THRESHOLD", str(float(scale_threshold)),
        "MODEL.SWIN_ROPE.VIYARN_ALPHA_MAX", str(float(alpha_max)),

        # avoid "resume hijack"
        "TRAIN.AUTO_RESUME", "False",
    ]

def md_row(cols):
    return "| " + " | ".join(str(c) for c in cols) + " |"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", required=True, help="path to your main.py (entry script)")
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)

    ap.add_argument("--gpus", default="", help="CUDA_VISIBLE_DEVICES, e.g. 0,1,2,3")
    ap.add_argument("--nproc", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--port", type=int, default=6555)

    ap.add_argument("--sizes", default="160,192,224,256,320,384,512")
    ap.add_argument("--base_window_size", type=int, default=7)

    # hyperparam grids (comma-separated)
    ap.add_argument("--gamma_lo", default="2.0")
    ap.add_argument("--gamma_hi", default="0.8")
    # ap.add_argument("--gamma_lo", default="2.0,2.3,2.6,3.0")
    # ap.add_argument("--gamma_hi", default="0.8,1.1,1.4,1.7")
    ap.add_argument("--transition", default="cos")
    ap.add_argument("--depth_ramp_p", default="1.0,2.0")
    ap.add_argument("--scale_threshold", default="1.05,10.0")   # 10.0 ~= ramp-off diagnostic
    ap.add_argument("--alpha_max", default="1.0")               # needs VIYARN_ALPHA_MAX patch

    ap.add_argument("--no_baseline", action="store_true", help="Only run VIYARN on (no matched baseline)")

    args = ap.parse_args()

    sizes = [int(x) for x in args.sizes.split(",") if x.strip()]
    if not sizes:
        sizes = SIZES_DEFAULT

    gamma_lo_list = [float(x) for x in args.gamma_lo.split(",") if x.strip()]
    gamma_hi_list = [float(x) for x in args.gamma_hi.split(",") if x.strip()]
    depth_ramp_p_list = [float(x) for x in args.depth_ramp_p.split(",") if x.strip()]
    scale_thr_list = [float(x) for x in args.scale_threshold.split(",") if x.strip()]
    alpha_max_list = [float(x) for x in args.alpha_max.split(",") if x.strip()]
    transition_list = [x.strip() for x in args.transition.split(",") if x.strip()]

    # enforce gamma_hi <= gamma_lo
    gamma_pairs = [(glo, ghi) for glo in gamma_lo_list for ghi in gamma_hi_list if ghi <= glo]
    if not gamma_pairs:
        raise ValueError("No valid (gamma_lo, gamma_hi) pairs. Ensure gamma_hi <= gamma_lo.")

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_path = out_root / f"sweep_{stamp}.csv"

    env = os.environ.copy()
    if args.gpus:
        env["CUDA_VISIBLE_DEVICES"] = args.gpus
    env.setdefault("OMP_NUM_THREADS", "1")

    header = [
        "setting_id", "img", "ws", "s_edge",
        "gamma_lo", "gamma_hi", "trans", "p", "thr", "alpha_max",
        "top1_base", "top1_yarn", "delta",
        "status",
    ]
    rows = []

    # Markdown streaming header
    print(md_row(header), flush=True)
    print(md_row(["---"] * len(header)), flush=True)

    # iterate settings (outer) then sizes (inner): "gather one setting across all sizes"
    setting_idx = 0
    for (glo, ghi), trans, p, thr, amax in itertools.product(
        gamma_pairs, transition_list, depth_ramp_p_list, scale_thr_list, alpha_max_list
    ):
        setting_idx += 1
        setting_id = f"S{setting_idx:04d}_glo{glo:.2f}_ghi{ghi:.2f}_t{trans}_p{p:.2f}_thr{thr:.2f}_amax{amax:.2f}"
        # keep each setting isolated
        setting_dir = out_root / setting_id
        setting_dir.mkdir(parents=True, exist_ok=True)

        for img in sizes:
            ws = img // 32
            s_edge = float(ws) / float(args.base_window_size)

            # run baseline (optional)
            top1_base = None
            if not args.no_baseline:
                base_dir = setting_dir / f"R{img}_baseline"
                base_dir.mkdir(parents=True, exist_ok=True)
                opts_base = make_opts(
                    img=img, window_size=ws,
                    yarn_enable=False,
                    gamma_lo=glo, gamma_hi=ghi,
                    transition=trans,
                    depth_ramp_p=p,
                    scale_threshold=thr,
                    alpha_max=amax,
                    base_window_size=args.base_window_size,
                )
                cmd_base = [
                    sys.executable, "-m", "torch.distributed.launch",
                    "--nproc_per_node", str(args.nproc),
                    "--master_port", str(next_port(args.port)),
                    "--nnodes=1",
                    "--use_env",
                    str(args.main),
                    "--cfg", str(args.cfg),
                    "--resume", str(args.ckpt),
                    "--data-path", str(args.data),
                    "--output", str(base_dir),
                    "--batch-size", str(args.batch),
                    "--eval",
                    "--opts", *opts_base,
                ]
                ret, out_txt = run_and_capture(cmd_base, env, base_dir / "stdout.txt")
                if ret != 0:
                    status = "FAILED_BASE"
                    row = [setting_id, img, ws, f"{s_edge:.3f}", f"{glo:.2f}", f"{ghi:.2f}", trans, f"{p:.2f}", f"{thr:.2f}", f"{amax:.2f}",
                           "", "", "", status]
                    rows.append(row)
                    print(md_row(row), flush=True)
                    continue
                top1_base = parse_top1(out_txt)

            # run yarn
            yarn_dir = setting_dir / f"R{img}_yarn"
            yarn_dir.mkdir(parents=True, exist_ok=True)
            opts_yarn = make_opts(
                img=img, window_size=ws,
                yarn_enable=True,
                gamma_lo=glo, gamma_hi=ghi,
                transition=trans,
                depth_ramp_p=p,
                scale_threshold=thr,
                alpha_max=amax,
                base_window_size=args.base_window_size,
            )
            cmd_yarn = [
                sys.executable, "-m", "torch.distributed.launch",
                "--nproc_per_node", str(args.nproc),
                "--master_port", str(next_port(args.port)),
                "--nnodes=1",
                "--use_env",
                str(args.main),
                "--cfg", str(args.cfg),
                "--resume", str(args.ckpt),
                "--data-path", str(args.data),
                "--output", str(yarn_dir),
                "--batch-size", str(args.batch),
                "--eval",
                "--opts", *opts_yarn,
            ]
            ret2, out_txt2 = run_and_capture(cmd_yarn, env, yarn_dir / "stdout.txt")
            if ret2 != 0:
                status = "FAILED_YARN"
                row = [setting_id, img, ws, f"{s_edge:.3f}", f"{glo:.2f}", f"{ghi:.2f}", trans, f"{p:.2f}", f"{thr:.2f}", f"{amax:.2f}",
                       f"{top1_base:.3f}" if top1_base is not None else "", "", "", status]
                rows.append(row)
                print(md_row(row), flush=True)
                continue
            top1_yarn = parse_top1(out_txt2)

            delta = ""
            if (top1_base is not None) and (top1_yarn is not None):
                delta = f"{(top1_yarn - top1_base):.3f}"

            status = "OK"
            row = [
                setting_id, img, ws, f"{s_edge:.3f}",
                f"{glo:.2f}", f"{ghi:.2f}", trans, f"{p:.2f}", f"{thr:.2f}", f"{amax:.2f}",
                f"{top1_base:.3f}" if top1_base is not None else "",
                f"{top1_yarn:.3f}" if top1_yarn is not None else "",
                delta,
                status,
            ]
            rows.append(row)
            print(md_row(row), flush=True)

            # incremental CSV write
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(header)
                w.writerows(rows)

    print(f"\nDone. CSV saved to: {csv_path}", flush=True)

if __name__ == "__main__":
    main()
