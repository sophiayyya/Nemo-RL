#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)
REPO_ROOT=$(realpath "${SCRIPT_DIR}/..")
cd "${REPO_ROOT}"

# Work around Megatron generation CUDA graph warmup with CP=4:
#   cuda graph warmup - [4092]: 0 P + 4092 D
#   AssertionError: val.shape=torch.Size([4092, 1, 2048]) cp_size=4

CONTAINER=${CONTAINER:-/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/alexq/nemo_rl.0622.sqsh}
ACCOUNT=${ACCOUNT:-coreai_dlalgo_llm}
PARTITION=${PARTITION:-batch}
TIME_LIMIT=${TIME_LIMIT:-1:00:00}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NUM_NODES=${NUM_NODES:-2}

HF_HOME=${HF_HOME:-/lustre/fsw/portfolios/coreai/users/sopyang/.cache/huggingface}
HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${HF_HOME}/datasets}

RUN_TAG=${RUN_TAG:-qwen35-vl-scope-none-$(date +%m%d-%H%M%S)}
BASE_LOG_DIR=${BASE_LOG_DIR:-/lustre/fsw/portfolios/coreai/users/sopyang/rl_logs/${RUN_TAG}}
JOB_NAME=${JOB_NAME:-coreai_dlalgo_llm:${RUN_TAG}}
MOUNTS=${MOUNTS:-/lustre/fsw:/lustre/fsw,/lustre/fs1:/lustre/fs1}

CONFIG=${CONFIG:-examples/configs/recipes/vlm/vlm_grpo-qwen3.5-35ba3b-geo3k-2n8g-megatron-ep16.yaml}
CONTEXT_PARALLEL_SIZE=${CONTEXT_PARALLEL_SIZE:-4}
MAKE_SEQUENCE_LENGTH_DIVISIBLE_BY=${MAKE_SEQUENCE_LENGTH_DIVISIBLE_BY:-$((2 * CONTEXT_PARALLEL_SIZE))}
NRL_FORCE_REBUILD_VENVS=${NRL_FORCE_REBUILD_VENVS:-true}

overrides=(
  "cluster.num_nodes=${NUM_NODES}"
  "cluster.gpus_per_node=${GPUS_PER_NODE}"
  "policy.generation.backend=megatron"
  "policy.generation.mcore_generation_config.cuda_graph_impl=none"
  "policy.generation.mcore_generation_config.inference_cuda_graph_scope=none"
  "policy.megatron_cfg.context_parallel_size=${CONTEXT_PARALLEL_SIZE}"
  "+policy.megatron_cfg.inference_moe_token_dispatcher_type=nccl"
  "policy.sequence_packing.enabled=true"
  "policy.make_sequence_length_divisible_by=${MAKE_SEQUENCE_LENGTH_DIVISIBLE_BY}"
  "logger.log_dir=${BASE_LOG_DIR}/nemo_logs"
)

COMMAND="NRL_FORCE_REBUILD_VENVS=${NRL_FORCE_REBUILD_VENVS} uv run examples/run_vlm_grpo.py --config ${CONFIG} ${overrides[*]}"

mkdir -p "${BASE_LOG_DIR}"

echo "Submitting ${JOB_NAME}"
echo "  container: ${CONTAINER}"
echo "  log dir:   ${BASE_LOG_DIR}"
echo "  command:   ${COMMAND}"

if [[ "${DRYRUN:-0}" != "0" ]]; then
  exit 0
fi

export CONTAINER
export MOUNTS
export COMMAND
export BASE_LOG_DIR
export GPUS_PER_NODE
export HF_HOME
export HF_DATASETS_CACHE
export NRL_FORCE_REBUILD_VENVS

sbatch \
  --nodes="${NUM_NODES}" \
  --account="${ACCOUNT}" \
  --job-name="${JOB_NAME}" \
  --partition="${PARTITION}" \
  --time="${TIME_LIMIT}" \
  --gres="gpu:${GPUS_PER_NODE}" \
  --output="${BASE_LOG_DIR}/slurm-%j-${RUN_TAG}.out" \
  ray.sub
