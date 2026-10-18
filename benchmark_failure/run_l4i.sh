#!/usr/bin/env bash
# L4-inject driver: build ~600 perturbed L4 twins from real L1/L2 scenes.
# Validity is guaranteed BY CONSTRUCTION (physical obstruction check in
# l4i_inject.py); step 3 runs the VLM audit but annotations stay L4.
# Everything lives under $L4I_OUT_ROOT (NOT merged into the main tree).
#
# PARALLELISM: generation shards across the EGL devices in $EGL_DEVICES
# (one inject process per device, disjoint scene shards); the VLM audit
# spreads the 27B model over the GPUs in $VLM_GPUS (device_map=auto).
#
# Usage:
#   bash run_l4i.sh run [N]    # select N (default 620) -> inject (all EGL devices) -> VLM audit
#   bash run_l4i.sh inject     # rerun inject only (resumable, all EGL devices)
#   bash run_l4i.sh audit [N]  # rerun VLM audit only (resumable)
#   bash run_l4i.sh status
#
# All paths are configurable via env vars (defaults fit the original host):
#   STARVLA_ROOT  data root for npz/bddl/sim_state rel paths  (/H20_vepfs/liulin/starVLA)
#   L4I_OUT_ROOT  output root                                 (<script dir>/l4_injected)
#   L4I_ANN_ROOT  base-annotation root for l4i_select         (<script dir>/annotations)
#   L4I_MODEL_PATH  VLM weights                               (/H20_vepfs/liulin/Qwen3VL-27B)
#   LIB_PY        python for sim/inject  (miniconda3/envs/libero)
#   QWEN_PY       python for the VLM     (miniconda3/envs/qwen)
#   EGL_DEVICES   EGL device ids for inject shards            (0,1,2,3)
#   VLM_GPUS      CUDA ids for the VLM audit                  (0,1,2,3)
#   LIBERO_DIR    LIBERO checkout                            ($STARVLA_ROOT/LIBERO)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

STARVLA_ROOT=${STARVLA_ROOT:-/H20_vepfs/liulin/starVLA}
LIBERO_DIR=${LIBERO_DIR:-$STARVLA_ROOT/LIBERO}
L4I_OUT_ROOT=${L4I_OUT_ROOT:-$HERE/l4_injected}
export STARVLA_ROOT L4I_OUT_ROOT

LIB_PY=${LIB_PY:-/H20_vepfs/liulin/miniconda3/envs/libero/bin/python}
QWEN_PY=${QWEN_PY:-/H20_vepfs/liulin/miniconda3/envs/qwen/bin/python}
EGL_DEVICES=${EGL_DEVICES:-0,1,2,3}
VLM_GPUS=${VLM_GPUS:-0,1,2,3}

export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export LIBERO_CONFIG_PATH=$LIBERO_DIR/libero
export PYTHONPATH=$LIBERO_DIR:$STARVLA_ROOT
export PYTHONUNBUFFERED=1

launch_inject() {  # one process per EGL device over disjoint scene shards
  local devices=(${EGL_DEVICES//,/ })
  local n=${#devices[@]} logs=$L4I_OUT_ROOT/logs
  mkdir -p "$logs"
  local pids=()
  for i in "${!devices[@]}"; do
    local d=${devices[$i]}
    MUJOCO_EGL_DEVICE_ID=$d CUDA_VISIBLE_DEVICES=$EGL_DEVICES \
      "$LIB_PY" "$HERE/l4i_inject.py" --egl "$d" \
        --shard "$i" --num-shards "$n" \
        --index-out "$L4I_OUT_ROOT/index/injected.shard$i.jsonl" \
        > "$logs/inject.shard$i.log" 2>&1 &
    pids+=($!)
    echo "  shard $i -> EGL device $d (pid ${pids[$i]}, log $logs/inject.shard$i.log)"
  done
  local fail=0
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
  if [[ $fail -ne 0 ]]; then
    echo "[WARN] at least one inject shard failed; last lines per shard:" >&2
    tail -n 3 "$logs"/inject.shard*.log >&2 || true
    exit 1
  fi
  tail -n 1 "$logs"/inject.shard*.log
}

cmd=${1:-help}
case "$cmd" in
  run)
    N=${2:-620}
    echo "[1/3] sampling $N L1/L2 scenes..."
    "$LIB_PY" "$HERE/l4i_select.py" --n "$N"
    echo "[2/3] injecting perturbations on EGL devices [$EGL_DEVICES] (validity-checked, annotations written as L4)..."
    launch_inject
    echo "[3/3] VLM audit on GPUs [$VLM_GPUS] (annotations stay L4, verdicts recorded only)..."
    CUDA_VISIBLE_DEVICES=$VLM_GPUS "$QWEN_PY" "$HERE/l4i_verify.py"
    ;;
  inject) launch_inject ;;
  audit) shift; CUDA_VISIBLE_DEVICES=$VLM_GPUS "$QWEN_PY" "$HERE/l4i_verify.py" "$@" ;;
  status)
    n_sel=$(wc -l 2>/dev/null < "$L4I_OUT_ROOT/index/selection.jsonl" || echo 0)
    n_inj=$({ cat "$L4I_OUT_ROOT"/index/injected*.jsonl 2>/dev/null || true; } | wc -l)
    n_verd=$(wc -l 2>/dev/null < "$L4I_OUT_ROOT/index/verdicts.jsonl" || echo 0)
    n_l4=$(grep -c '"level": 4' "$L4I_OUT_ROOT/index/verdicts.jsonl" 2>/dev/null || true)
    n_ann=$(find "$L4I_OUT_ROOT/annotations" -name '*.json' 2>/dev/null | wc -l || true)
    echo "selected: $n_sel   injected: $n_inj   verified: $n_verd   L4 confirmed: ${n_l4:-0}   annotations: $n_ann"
    ;;
  *)
    sed -n '2,31p' "$0" | sed 's/^# \{0,1\}//'
    ;;
esac
