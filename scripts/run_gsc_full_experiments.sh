#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-GSC_data}"
RUN_ROOT="${RUN_ROOT:-runs/gsc_full}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SUP_EPOCHS="${SUP_EPOCHS:-50}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-400}"
RUN_PRETRAIN="${RUN_PRETRAIN:-1}"
CONDA_ENV="${CONDA_ENV:-}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/neuralmusic_mpl}"

run_python() {
  if [[ -n "${CONDA_ENV}" ]]; then
    conda run -n "${CONDA_ENV}" python "$@"
  else
    python3 "$@"
  fi
}

run_python -c "import sys, torch; ok=torch.cuda.is_available(); print('CUDA available:', ok); print('CUDA device:', torch.cuda.get_device_name(0) if ok else 'none'); sys.exit(0 if ok else 1)" || {
  echo "CUDA is not available. Full GSC training is too slow on CPU; please run on a GPU machine."
  exit 1
}

mkdir -p "${RUN_ROOT}"

if [[ "${RUN_PRETRAIN}" == "1" ]]; then
  run_python train_selfsupervised.py \
    --dataset gsc \
    --data-root "${DATA_ROOT}" \
    --save-dir "${RUN_ROOT}/pretrain" \
    --input-channel 8 \
    --batch-size 256 \
    --epochs "${PRETRAIN_EPOCHS}" \
    --num-workers "${NUM_WORKERS}" \
    --num-percent 1.0 \
    --val-percent 1.0 \
    --num-sources 1 \
    --noise-aug

  run_python test_selfsupervised.py \
    --dataset gsc \
    --data-root "${DATA_ROOT}" \
    --checkpoint "${RUN_ROOT}/pretrain/best_model.pt" \
    --save-dir "${RUN_ROOT}/pretrain_test" \
    --input-channel 8 \
    --batch-size 64 \
    --num-workers "${NUM_WORKERS}" \
    --val-percent 1.0 \
    --num-figures 8
fi

PRETRAIN_ARGS=()
if [[ -f "${RUN_ROOT}/pretrain/best_model.pt" ]]; then
  PRETRAIN_ARGS=(--pretrain "${RUN_ROOT}/pretrain/best_model.pt")
fi

run_python train_neuralmusic.py \
  --dataset gsc \
  --data-root "${DATA_ROOT}" \
  --save-dir "${RUN_ROOT}/supervised_plain" \
  --num-sources 1 \
  --max-sources 4 \
  --input-channel 8 \
  --batch-size 32 \
  --epochs "${SUP_EPOCHS}" \
  --num-workers "${NUM_WORKERS}" \
  --num-percent 1.0 \
  --val-percent 1.0 \
  --noise-aug \
  "${PRETRAIN_ARGS[@]}"

run_python test_neuralmusic.py \
  --dataset gsc \
  --data-root "${DATA_ROOT}" \
  --checkpoint "${RUN_ROOT}/supervised_plain/best_model.pt" \
  --save-dir "${RUN_ROOT}/supervised_plain_test" \
  --num-sources 1 \
  --max-sources 4 \
  --input-channel 8 \
  --batch-size 32 \
  --num-workers "${NUM_WORKERS}" \
  --val-percent 1.0

run_python train_neuralmusic.py \
  --dataset gsc \
  --data-root "${DATA_ROOT}" \
  --save-dir "${RUN_ROOT}/supervised_cls" \
  --num-sources 1 \
  --max-sources 4 \
  --input-channel 8 \
  --batch-size 32 \
  --epochs "${SUP_EPOCHS}" \
  --num-workers "${NUM_WORKERS}" \
  --num-percent 1.0 \
  --val-percent 1.0 \
  --noise-aug \
  --estimate-num-sources \
  "${PRETRAIN_ARGS[@]}"

run_python test_neuralmusic.py \
  --dataset gsc \
  --data-root "${DATA_ROOT}" \
  --checkpoint "${RUN_ROOT}/supervised_cls/best_model.pt" \
  --save-dir "${RUN_ROOT}/supervised_cls_test" \
  --num-sources 1 \
  --max-sources 4 \
  --input-channel 8 \
  --batch-size 32 \
  --num-workers "${NUM_WORKERS}" \
  --val-percent 1.0 \
  --estimate-num-sources
