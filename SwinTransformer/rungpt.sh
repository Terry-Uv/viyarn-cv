# ---- paths ----
MAIN="main.py"
CFG="configs/swinrope/swin_rope_axial_small_patch4_window7_224.yaml"
CKPT="official_ckpt/swin-rope-axial-tiny.bin"
DATA="/media/ssd1/zwx/dataset/im1k"
OUTROOT="result/sweep_grid"

# ---- runtime ----
GPUS="0,1,2,3,4,5,6,7"
NPROC=8
BATCH=32
PORT=6555

# ---- sweep script ----
SWEEP="./swin-viyarn-hyperSweep-grid.py"   # 放到 repo 根目录或改成绝对路径

# ---- ntfy ----
NTFY_URL="https://ntfy.sh"
NTFY_TOPIC="YOUR_TOPIC_NAME"              # 改成你自己的 topic（例如 viyarn-swin-xxx）
# 如需 token（私有 topic），用：NTFY_TOKEN="tk_xxx"
NTFY_TOKEN=""

ntfy_send () {
  local title="$1"
  local msg="$2"
  if [[ -n "${NTFY_TOKEN}" ]]; then
    curl -fsS -X POST \
      -H "Authorization: Bearer ${NTFY_TOKEN}" \
      -H "Title: ${title}" \
      -d "${msg}" \
      "${NTFY_URL}/${NTFY_TOPIC}" >/dev/null
  else
    curl -fsS -X POST \
      -H "Title: ${title}" \
      -d "${msg}" \
      "${NTFY_URL}/${NTFY_TOPIC}" >/dev/null
  fi
}

mkdir -p "${OUTROOT}"

# 固定的尺寸集合（你要 gather 全套）
SIZES="160,192,224,256,320,384,512"

echo "[Phase-1] thr=10 (ramp-off diagnostic) sweeping gamma grid..."
PH1_OUT="${OUTROOT}/phase1_thr10_gamma"
mkdir -p "${PH1_OUT}"

set +e
python "${SWEEP}" \
  --main "${MAIN}" \
  --cfg  "${CFG}" \
  --ckpt "${CKPT}" \
  --data "${DATA}" \
  --out  "${PH1_OUT}" \
  --gpus "${GPUS}" \
  --nproc "${NPROC}" \
  --batch "${BATCH}" \
  --port "${PORT}" \
  --sizes "${SIZES}" \
  --base_window_size 7 \
  --gamma_lo "2.0,2.3,2.6,3.0" \
  --gamma_hi "0.8,1.1,1.4,1.7" \
  --transition "cos" \
  --depth_ramp_p "1.0" \
  --scale_threshold "10.0" \
  --alpha_max "1.0" \
  | tee "${PH1_OUT}/phase1_stdout.md"
PH1_RET=${PIPESTATUS[0]}
set -e

if [[ "${PH1_RET}" -ne 0 ]]; then
  ntfy_send "Swin-ViYaRN sweep PH1 FAILED" "Phase-1 failed. Check: ${PH1_OUT}/phase1_stdout.md"
  exit 1
else
  ntfy_send "Swin-ViYaRN sweep PH1 done" "Phase-1 done. Output: ${PH1_OUT}"
fi

echo
echo ">>> Now pick BEST_GLO / BEST_GHI from Phase-1 CSV (delta across sizes)."
echo ">>> Edit below variables and rerun Phase-2, OR keep running if already set."
echo
