#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

if [[ -f "${SCRIPT_DIR}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/.env"
  set +a
fi

: "${VLLM_HOST:=127.0.0.1}"
: "${VLLM_PORT:=8001}"
: "${VLLM_GPU:=1}"
: "${VLLM_GPU_MEMORY_UTILIZATION:=0.90}"
: "${VLLM_MAX_MODEL_LEN:=2048}"
: "${VLLM_BASE_MODEL:=Qwen/Qwen3-4B-Instruct-2507}"
: "${VLLM_MODEL_REVISION:=f50518eb58dfc750271b273fc113bdfc16ec2280}"
: "${VLLM_ADAPTER_NAME:=retail-agent}"
: "${VLLM_ADAPTER_PATH:=../model-pipeline/artifacts/qlora/qwen3-4b-instruct-2507-guardrails}"

if [[ "${VLLM_ADAPTER_PATH}" = /* ]]; then
  ADAPTER_DIR="${VLLM_ADAPTER_PATH}"
else
  ADAPTER_DIR="${SCRIPT_DIR}/${VLLM_ADAPTER_PATH}"
fi

if [[ ! -f "${ADAPTER_DIR}/adapter_config.json" ]]; then
  echo "Missing LoRA adapter: ${ADAPTER_DIR}/adapter_config.json" >&2
  exit 1
fi

if [[ ! -f "${ADAPTER_DIR}/chat_template.jinja" ]]; then
  echo "Missing chat template: ${ADAPTER_DIR}/chat_template.jinja" >&2
  exit 1
fi

cd "${REPO_DIR}"
CUDA_VISIBLE_DEVICES="${VLLM_GPU}" exec vllm serve "${VLLM_BASE_MODEL}" \
  --revision "${VLLM_MODEL_REVISION}" \
  --dtype half \
  --max-model-len "${VLLM_MAX_MODEL_LEN}" \
  --max-num-seqs 2 \
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
  --enforce-eager \
  --enable-lora \
  --max-lora-rank 8 \
  --lora-modules "${VLLM_ADAPTER_NAME}=${ADAPTER_DIR}" \
  --chat-template "${ADAPTER_DIR}/chat_template.jinja" \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --host "${VLLM_HOST}" \
  --port "${VLLM_PORT}"
