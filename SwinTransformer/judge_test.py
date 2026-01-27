'''
config_file="/workspace/ssd/models/qwen/viy/SwinTransformer/configs/swinrope/swin_rope_axial_small_patch4_window7_224.yaml"
checkpoint_file="/workspace/ssd/models/qwen/viy/SwinTransformer/official_ckpt/swin-rope-axial-small.bin"
save_dir="/workspace/ssd/models/qwen/viy/SwinTransformer/result"
data_path="/media/disk1/models/dataset/imagenet1K" 
python -u sweep.py \
  --cfg $config_file \
  --resume $checkpoint_file \
  --data-path $data_path \
  --output $save_dir/sweep_thr105 \
  --sizes 160,192,224,256,320,384,512 \
  --scale-thr 1.05 \
  --transition cos \
  --gamma-lo 2.0 \
  --gamma-hi 1.8 \
  --alpha-max 0.4,0.6,0.8,1.0 \
  --ramp-p 0.5,1.0,2.0
'''
import argparse
import os
import re
import subprocess
from collections import deque
from typing import Dict, List, Optional, Tuple


# Prefer precise Acc@1 line:
# [..] INFO  * Acc@1 79.512 Acc@5 94.752
ACC1_RE = re.compile(r"\*\s*Acc@1\s+([0-9]+(?:\.[0-9]+)?)\b")
# Fallback rounded summary:
# [..] INFO Accuracy of the network on the 50000 test images: 79.5%
ACC_FALLBACK_RE = re.compile(r"Accuracy of the network on the .* test images:\s*([0-9]+(?:\.[0-9]+)?)%")


def parse_baseline_map(s: str) -> Dict[int, float]:
    """
    Parse baseline map like: "384:80.89,512:75.722"
    Returns dict[int,float]
    """
    s = (s or "").strip()
    out: Dict[int, float] = {}
    if not s:
        return out
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        k, v = item.split(":")
        out[int(k.strip())] = float(v.strip())
    return out


def ntfy(topic: str, title: str, msg: str, token: str = "") -> None:
    """
    Send notification via ntfy.
    - If topic is empty, do nothing.
    - If token provided, use Authorization: Bearer <token>.
    """
    topic = (topic or "").strip()
    # if not topic:
    #     return
    url = f"https://ntfy.sh/avbidy"
    cmd = ["curl", "-fsS", "-X", "POST", "-H", f"Title: {title}"]
    if token:
        cmd += ["-H", f"Authorization: Bearer {token}"]
    cmd += ["-d", msg, url]
    try:
        subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


def run_cmd(cmd: List[str], *, cwd=None, env=None, tail_n: int = 120, verbose: bool = False) -> Tuple[int, Optional[float], List[str]]:
    """
    Run subprocess, return (returncode, acc1 or None, tail_lines(list[str])).
    - If verbose: stream logs to stdout.
    - Always keep last tail_n lines for failure debugging.
    """
    last = deque(maxlen=tail_n)
    p = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    acc1_fallback: Optional[float] = None
    acc1_precise: Optional[float] = None

    assert p.stdout is not None
    for line in p.stdout:
        last.append(line.rstrip("\n"))
        if verbose:
            print(line, end="")

        m1 = ACC1_RE.search(line)
        if m1:
            try:
                acc1_precise = float(m1.group(1))
            except Exception:
                pass
            continue

        m2 = ACC_FALLBACK_RE.search(line)
        if m2:
            try:
                acc1_fallback = float(m2.group(1))
            except Exception:
                pass

    rc = p.wait()
    acc1 = acc1_precise if acc1_precise is not None else acc1_fallback
    return rc, acc1, list(last)


def build_ddp_cmd(args, master_port: int) -> List[str]:
    # Keep torch.distributed.launch for compatibility with your current environment.
    return [
        "python", "-m", "torch.distributed.launch",
        f"--nproc_per_node={args.nproc_per_node}",
        "--nnodes=1",
        f"--master_port={master_port}",
        "--use_env",
        args.main_py,
    ]


def one_eval(
    args,
    *,
    img: int,
    gamma_lo: float,
    gamma_hi: float,
    transition: str,
    scale_thr: float,
    alpha_max: float,
    ramp_p: float,
    enable: bool,
) -> Tuple[int, Optional[float], List[str], str]:
    """
    Run one eval for a given img size and hyperparam setting.
    Returns (rc, acc1, tail_lines, out_dir).
    """
    win = img // 32
    out_dir = os.path.join(
        args.output,
        f"img{img}_win{win}",
        f"en{int(enable)}_glo{gamma_lo}_ghi{gamma_hi}_thr{scale_thr}_a{alpha_max}_p{ramp_p}_{transition}",
    )
    os.makedirs(out_dir, exist_ok=True)

    # Port: avoid collisions across sizes/settings by hashing the (img, setting, enable).
    key = (img, gamma_lo, gamma_hi, transition, scale_thr, alpha_max, ramp_p, enable)
    master_port = int(args.master_port) + (abs(hash(key)) % 2000)

    cmd: List[str] = []
    cmd += build_ddp_cmd(args, master_port)
    cmd += [
        "--cfg", args.cfg,
        "--resume", args.resume,
        "--data-path", args.data_path,
        "--output", out_dir,
        "--batch-size", str(args.batch_size),
        "--eval",
        "--opts",
        # "TRAIN.AUTO_RESUME", "False",
        "DATA.IMG_SIZE", str(img),
        "DATA.IMG_SIZE_W", str(img),
        "MODEL.SWIN.WINDOW_SIZE", str(win),

        # ViYaRN (Swin-RoPE)
        "MODEL.SWIN_ROPE.VIYARN_ENABLE", "True" if enable else "False",
        "MODEL.SWIN_ROPE.YARN_GAMMA_LO", str(gamma_lo),
        "MODEL.SWIN_ROPE.YARN_GAMMA_HI", str(gamma_hi),
        "MODEL.SWIN_ROPE.YARN_TRANSITION", str(transition),
        "MODEL.SWIN_ROPE.VIYARN_SCALE_THRESHOLD", str(scale_thr),
        "MODEL.SWIN_ROPE.VIYARN_ALPHA_MAX", str(alpha_max),
        "MODEL.SWIN_ROPE.VIYARN_DEPTH_RAMP_P", str(ramp_p),
    ]

    env = os.environ.copy()
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    env.setdefault("OMP_NUM_THREADS", "1")

    rc, acc1, tail = run_cmd(cmd, tail_n=args.tail_n, verbose=args.verbose_subprocess)
    return rc, acc1, tail, out_dir


def fmt_acc(x: Optional[float]) -> str:
    return "NA" if x is None else f"{x:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--resume", required=True)
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--output", required=True)

    ap.add_argument("--main-py", default="main.py")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--nproc-per-node", type=int, default=8)
    ap.add_argument("--master-port", type=int, default=6555)
    ap.add_argument("--cuda-visible-devices", default="", help="e.g. 0,1,2,3 (optional)")

    ap.add_argument("--tail-n", type=int, default=120)
    ap.add_argument("--verbose-subprocess", action="store_true")

    ap.add_argument("--sizes", default="384,512")

    # grids
    ap.add_argument("--gamma-lo", default="2.0,2.2,2.4,2.6")
    ap.add_argument("--gamma-hi", default="1.2,1.4,1.6,1.8")
    ap.add_argument("--transition", default="cos")
    ap.add_argument("--scale-thr", type=float, default=10.0)
    ap.add_argument("--alpha-max", default="1.0")
    ap.add_argument("--ramp-p", default="1.0")

    # baseline handling
    ap.add_argument("--baseline-map", default="", help='e.g. "384:80.89,512:75.722"')
    ap.add_argument("--skip-baseline-run", action="store_true", help="skip baseline evaluation")
    ap.add_argument("--baseline-gamma-lo", type=float, default=2.3)
    ap.add_argument("--baseline-gamma-hi", type=float, default=1.7)

    # notify
    ap.add_argument("--ntfy-topic", default=os.environ.get("NTFY_TOPIC", ""))
    ap.add_argument("--ntfy-token", default=os.environ.get("NTFY_TOKEN", ""))

    args = ap.parse_args()

    sizes = [int(x.strip()) for x in args.sizes.split(",") if x.strip()]
    gamma_los = [float(x.strip()) for x in args.gamma_lo.split(",") if x.strip()]
    gamma_his = [float(x.strip()) for x in args.gamma_hi.split(",") if x.strip()]
    transitions = [x.strip() for x in args.transition.split(",") if x.strip()]
    alpha_maxs = [float(x.strip()) for x in args.alpha_max.split(",") if x.strip()]
    ramp_ps = [float(x.strip()) for x in args.ramp_p.split(",") if x.strip()]
    scale_thr = float(args.scale_thr)

    os.makedirs(args.output, exist_ok=True)

    baseline = parse_baseline_map(args.baseline_map)

    # Optional baseline run (only for missing sizes)
    if not args.skip_baseline_run:
        missing = [img for img in sizes if img not in baseline]
        if missing:
            print(f"Running baseline for missing sizes: {missing}")
            for img in missing:
                rc, acc, tail, out_dir = one_eval(
                    args,
                    img=img,
                    gamma_lo=args.baseline_gamma_lo,
                    gamma_hi=args.baseline_gamma_hi,
                    transition=transitions[0],
                    scale_thr=scale_thr,
                    alpha_max=1.0,
                    ramp_p=1.0,
                    enable=False,
                )
                if rc != 0 or acc is None:
                    print(f"[FAILED_BASE] img={img} rc={rc} out={out_dir}")
                    print("---- tail ----")
                    print("\n".join(tail))
                    ntfy(args.ntfy_topic, "Swin ViYaRN sweep FAILED_BASE",
                         f"img={img} rc={rc}\nout={out_dir}\n" + "\n".join(tail[-30:]),
                         token=args.ntfy_token)
                    return
                baseline[img] = acc
                print(f"baseline img={img}: acc1={acc:.3f}")
        else:
            print("Baseline-map covers all sizes; no baseline run needed.")
    else:
        if baseline:
            print(f"Using provided baseline-map (no baseline run): {baseline}")
        else:
            print("No baseline-map and baseline run skipped.")

    # Prepare all settings
    settings = []
    for trans in transitions:
        for amax in alpha_maxs:
            for p in ramp_ps:
                for glo in gamma_los:
                    for ghi in gamma_his:
                        if glo < ghi:
                            continue
                        settings.append((glo, ghi, trans, scale_thr, amax, p))

    print(f"\nTotal settings: {len(settings)}")
    print(f"Sizes: {sizes}")
    if baseline:
        print(f"Baseline available for sizes: {sorted(baseline.keys())}")

    # Run all settings; gather results per setting
    all_results = []  # each item: dict with hyperparams + acc per size

    for idx, (glo, ghi, trans, thr, amax, p) in enumerate(settings, 1):
        setting_id = f"S{idx:04d} glo={glo:.3f} ghi={ghi:.3f} t={trans} thr={thr:.3f} amax={amax:.3f} p={p:.3f}"
        print(f"\n[{setting_id}] running...")

        acc_map: Dict[int, Optional[float]] = {}
        ok = True

        for img in sizes:
            rc, acc, tail, out_dir = one_eval(
                args,
                img=img,
                gamma_lo=glo,
                gamma_hi=ghi,
                transition=trans,
                scale_thr=thr,
                alpha_max=amax,
                ramp_p=p,
                enable=True,
            )
            if rc != 0 or acc is None:
                ok = False
                acc_map[img] = None
                print(f"[FAILED] img={img} rc={rc} out={out_dir}")
                print("---- tail ----")
                print("\n".join(tail))
                ntfy(args.ntfy_topic, "Swin ViYaRN sweep FAILED",
                     f"{setting_id}\nimg={img} rc={rc}\nout={out_dir}\n" + "\n".join(tail[-30:]),
                     token=args.ntfy_token)
                break
            acc_map[img] = acc

        # Print once per setting (after all sizes)
        if ok:
            parts = []
            delta_parts = []
            for img in sizes:
                a = acc_map.get(img)
                parts.append(f"{img}:{fmt_acc(a)}")
                if (a is not None) and (img in baseline):
                    delta_parts.append(f"Δ{img}:{a - baseline[img]:+.3f}")

            line = "  ".join(parts)
            if delta_parts:
                line += "   |   " + "  ".join(delta_parts)

            print(line)

            result = {
                "glo": glo, "ghi": ghi, "trans": trans, "thr": thr, "amax": amax, "p": p,
                "acc": acc_map,
            }
            all_results.append(result)

    # Summary: print all results sorted by average over available sizes (or by 512 if exists)
    if not all_results:
        print("\nNo successful settings.")
        ntfy(args.ntfy_topic, "Swin ViYaRN sweep DONE", "No successful settings.", token=args.ntfy_token)
        return

    def score(res) -> float:
        accs = [v for v in (res["acc"].get(s) for s in sizes) if v is not None]
        if not accs:
            return -1e9
        # If 512 exists, bias toward it slightly
        if 512 in res["acc"] and res["acc"][512] is not None:
            return (sum(accs) / len(accs)) + 0.1 * res["acc"][512]
        return sum(accs) / len(accs)

    all_results.sort(key=score, reverse=True)

    print("\n==================== FINAL SUMMARY (sorted) ====================")
    for r in all_results:
        acc_map = r["acc"]
        head = f"glo={r['glo']:.3f} ghi={r['ghi']:.3f} t={r['trans']} thr={r['thr']:.3f} amax={r['amax']:.3f} p={r['p']:.3f}"
        parts = [f"{img}:{fmt_acc(acc_map.get(img))}" for img in sizes]
        line = head + " | " + "  ".join(parts)
        if baseline:
            deltas = []
            for img in sizes:
                a = acc_map.get(img)
                if (a is not None) and (img in baseline):
                    deltas.append(f"Δ{img}:{a - baseline[img]:+.3f}")
            if deltas:
                line += "   |   " + "  ".join(deltas)
        print(line)

    best = all_results[0]
    best_msg = (
        f"BEST setting:\n"
        f"glo={best['glo']:.3f} ghi={best['ghi']:.3f} t={best['trans']} thr={best['thr']:.3f} amax={best['amax']:.3f} p={best['p']:.3f}\n"
        + "  ".join([f"{img}:{fmt_acc(best['acc'].get(img))}" for img in sizes])
    )
    print("\n" + best_msg)
    ntfy(args.ntfy_topic, "Swin ViYaRN sweep DONE", best_msg, token=args.ntfy_token)


if __name__ == "__main__":
    main()
