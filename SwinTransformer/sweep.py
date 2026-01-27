'''
config_file="/workspace/ssd/models/qwen/viy/SwinTransformer/configs/swinrope/swin_rope_axial_small_patch4_window7_224.yaml"
checkpoint_file="/workspace/ssd/models/qwen/viy/SwinTransformer/official_ckpt/swin-rope-axial-small.bin"
save_dir="/workspace/ssd/models/qwen/viy/SwinTransformer/result"
data_path="/media/disk1/models/dataset/imagenet1K"
python -u sweep.py \
  --cfg $config_file \
  --resume $checkpoint_file \
  --data-path $data_path \
  --output $save_dir/sweep_thr10 \
  --sizes 384,512 \
  --scale-thr 10.0 \
  --alpha-max 1.0 \
  --ramp-p 1.0 \
  --transition cos \
  --gamma-lo 2.0,2.2,2.4,2.6 \
  --gamma-hi 1.2,1.4,1.6,1.8

config_file="/workspace/ssd/models/qwen/viy/SwinTransformer/configs/swinrope/swin_rope_axial_small_patch4_window7_224.yaml"
checkpoint_file="/workspace/ssd/models/qwen/viy/SwinTransformer/official_ckpt/swin-rope-axial-small.bin"
save_dir="/workspace/ssd/models/qwen/viy/SwinTransformer/result"
data_path="/media/disk1/models/dataset/imagenet1K" 
python -u sweep.py \
  --cfg $config_file \
  --resume $checkpoint_file \
  --data-path $data_path \
  --output $save_dir/sweep_thr105 \
  --sizes 384,512 \
  --scale-thr 1.05 \
  --transition cos \
  --gamma-lo 2.0 \
  --gamma-hi 1.8 \
  --alpha-max 0.4,0.6,0.8,1.0 \
  --ramp-p 0.5,1.0,2.0
'''

# swin-viyarn-hyperSweep-grid.py
import argparse
import os
import re
import shlex
import subprocess
from collections import deque

# ACC_RE = re.compile(r"Accuracy of the network on the .* test images:\s*([0-9.]+)%")
ACC1_RE = re.compile(r"\*\s*Acc@1\s+([0-9]+(?:\.[0-9]+)?)\b")
ACC_FALLBACK_RE = re.compile(r"Accuracy of the network on the .* test images:\s*([0-9]+(?:\.[0-9]+)?)%")

def run_cmd(cmd, *, cwd=None, env=None, tail_n=120, verbose=False):
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
    acc1 = None
    acc1_precise = None  # prefer "* Acc@1 ..."
    for line in p.stdout:
        last.append(line.rstrip("\n"))
        if verbose:
            print(line, end="")

        m1 = ACC1_RE.search(line)
        if m1:
            try:
                acc1_precise = float(m1.group(1))
            except:
                pass
            continue

        m2 = ACC_FALLBACK_RE.search(line)
        if m2:
            try:
                acc1 = float(m2.group(1))
            except:
                pass
    rc = p.wait()
    if acc1_precise is not None:
        acc1 = acc1_precise
    return rc, acc1, list(last)

def ntfy(topic, title, msg):
    # if not topic:
    #     return
    # URL must be in code per your policy
    url = f"https://ntfy.sh/avbidy"
    try:
        subprocess.run(
            ["curl", "-s", "-H", f"Title: {title}", "-d", msg, url],
            check=False,
        )
    except Exception:
        pass

def build_ddp_cmd(args, master_port):
    # Keep your launch style; you can switch to torchrun if desired.
    return [
        "python", "-m", "torch.distributed.launch",
        f"--nproc_per_node={args.nproc_per_node}",
        "--nnodes=1",
        f"--master_port={master_port}",
        "--use_env",
        args.main_py,
    ]

def one_eval(args, *, img, gamma_lo, gamma_hi, transition, scale_thr, alpha_max, ramp_p, enable):
    win = img // 32
    # IMPORTANT: output dir unique per run to avoid collisions
    out_dir = os.path.join(args.output, f"img{img}_win{win}",
                           f"en{int(enable)}_glo{gamma_lo}_ghi{gamma_hi}_thr{scale_thr}_a{alpha_max}_p{ramp_p}_{transition}")
    os.makedirs(out_dir, exist_ok=True)

    # rotate port slightly to reduce "Address already in use" flakiness
    # master_port = args.master_port + (img % 100) + (hash((gamma_lo, gamma_hi, scale_thr, alpha_max, ramp_p, enable)) % 200)
    master_port = 6555 if img==384 else 6556
    cmd = []
    cmd += build_ddp_cmd(args, master_port)
    cmd += [
        "--cfg", args.cfg,
        "--resume", args.resume,
        "--data-path", args.data_path,
        "--output", out_dir,
        "--batch-size", str(args.batch_size),
        "--eval",
        "--opts",
        "DATA.IMG_SIZE", str(img),
        "DATA.IMG_SIZE_W", str(img),
        "MODEL.SWIN.WINDOW_SIZE", str(win),

        # ViYaRN
        "MODEL.SWIN_ROPE.VIYARN_ENABLE", "True" if enable else "False",
        "MODEL.SWIN_ROPE.YARN_GAMMA_LO", str(gamma_lo),
        "MODEL.SWIN_ROPE.YARN_GAMMA_HI", str(gamma_hi),
        "MODEL.SWIN_ROPE.YARN_TRANSITION", str(transition),
        "MODEL.SWIN_ROPE.VIYARN_SCALE_THRESHOLD", str(scale_thr),
        "MODEL.SWIN_ROPE.VIYARN_ALPHA_MAX", str(alpha_max),
        "MODEL.SWIN_ROPE.VIYARN_DEPTH_RAMP_P", str(ramp_p),
    ]

    rc, acc1, tail = run_cmd(cmd, tail_n=args.tail_n, verbose=args.verbose_subprocess)
    return rc, acc1, tail, out_dir

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

    ap.add_argument("--ntfy-topic", default=os.environ.get("NTFY_TOPIC", ""))
    ap.add_argument("--tail-n", type=int, default=120)
    ap.add_argument("--verbose-subprocess", action="store_true")

    # fixed eval sizes (per your request: only 384/512)
    ap.add_argument("--sizes", default="384,512")

    # grids
    ap.add_argument("--gamma-lo", default="2.1,2.3,2.5")
    ap.add_argument("--gamma-hi", default="1.3,1.5,1.7")
    ap.add_argument("--transition", default="cos")
    ap.add_argument("--scale-thr", type=float, default=10.0)
    ap.add_argument("--alpha-max", default="1.0")
    ap.add_argument("--ramp-p", default="1.0")

    args = ap.parse_args()

    sizes = [int(x) for x in args.sizes.split(",") if x.strip()]
    gamma_los = [float(x) for x in args.gamma_lo.split(",") if x.strip()]
    gamma_his = [float(x) for x in args.gamma_hi.split(",") if x.strip()]
    transitions = [x.strip() for x in args.transition.split(",") if x.strip()]
    alpha_maxs = [float(x) for x in args.alpha_max.split(",") if x.strip()]
    ramp_ps = [float(x) for x in args.ramp_p.split(",") if x.strip()]
    scale_thr = float(args.scale_thr)

    # Baseline once per size
    baseline = {}
    print("Running baseline (ViYaRN disabled) ...")
    for img in sizes:
        rc, acc, tail, out_dir = one_eval(
            args, img=img,
            gamma_lo=2.3, gamma_hi=1.7, transition=transitions[0],
            scale_thr=scale_thr, alpha_max=1.0, ramp_p=1.0,
            enable=False
        )
        if rc != 0 or acc is None:
            print(f"[FAILED_BASE] img={img} rc={rc} out={out_dir}")
            print("---- tail ----")
            print("\n".join(tail))
            ntfy(args.ntfy_topic, "Swin ViYaRN sweep FAILED_BASE", f"img={img} rc={rc}\nout={out_dir}\n" + "\n".join(tail[-30:]))
            return
        baseline[img] = acc
        print(f"baseline img={img}: acc1={acc:.3f}")

    print("\n| gamma_lo | gamma_hi | trans | thr | alpha_max | ramp_p | acc384 | acc512 |")
    print("|---:|---:|:---:|---:|---:|---:|---:|---:|")

    results = []

    for trans in transitions:
        for amax in alpha_maxs:
            for p in ramp_ps:
                for glo in gamma_los:
                    for ghi in gamma_his:
                        # enforce gamma_lo >= gamma_hi (typical YaRN intent)
                        if glo < ghi:
                            continue

                        row = {"glo": glo, "ghi": ghi, "trans": trans, "thr": scale_thr, "amax": amax, "p": p}
                        ok = True
                        for img in sizes:
                            rc, acc, tail, out_dir = one_eval(
                                args, img=img,
                                gamma_lo=glo, gamma_hi=ghi, transition=trans,
                                scale_thr=scale_thr, alpha_max=amax, ramp_p=p,
                                enable=True
                            )
                            if rc != 0 or acc is None:
                                ok = False
                                row[f"acc{img}"] = None
                                print(f"[FAILED] glo={glo} ghi={ghi} img={img} rc={rc} out={out_dir}")
                                print("---- tail ----")
                                print("\n".join(tail))
                                ntfy(args.ntfy_topic, "Swin ViYaRN sweep FAILED", f"glo={glo} ghi={ghi} img={img} rc={rc}\nout={out_dir}\n" + "\n".join(tail[-30:]))
                                break
                            row[f"acc{img}"] = acc

                        if ok:
                            acc384 = row.get("acc384")
                            acc512 = row.get("acc512")
                            print(f"| {glo:.3f} | {ghi:.3f} | {trans} | {scale_thr:.3f} | {amax:.3f} | {p:.3f} | {acc384:.3f} | {acc512:.3f} |")
                            results.append(row)
                            ntfy(args.ntfy_topic, "Swin ViYaRN sweep progress",
                                 f"glo={glo} ghi={ghi} thr={scale_thr} amax={amax} p={p}\nacc384={acc384:.3f} acc512={acc512:.3f}\n"
                                 f"Δ384={acc384-baseline[384]:+.3f} Δ512={acc512-baseline[512]:+.3f}")

    # summary
    if results:
        # sort by (acc512 + acc384)
        results.sort(key=lambda r: (r["acc384"] + r["acc512"]), reverse=True)
        best = results[0]
        msg = (f"BEST: glo={best['glo']} ghi={best['ghi']} trans={best['trans']} thr={best['thr']} "
               f"amax={best['amax']} p={best['p']}\n"
               f"acc384={best['acc384']:.3f} (Δ{best['acc384']-baseline[384]:+.3f})\n"
               f"acc512={best['acc512']:.3f} (Δ{best['acc512']-baseline[512]:+.3f})")
        print("\n" + msg)
        ntfy(args.ntfy_topic, "Swin ViYaRN sweep DONE", msg)
    else:
        ntfy(args.ntfy_topic, "Swin ViYaRN sweep DONE", "No successful runs.")

if __name__ == "__main__":
    main()
