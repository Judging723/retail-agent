# Model serving

This module serves the trained QLoRA adapter through an OpenAI-compatible HTTP API. It stays separate from
FastAPI: the customer-service application is a client, while vLLM owns GPU inference and model lifecycle.

## Verified environment

- Ubuntu 24.04, two RTX 2080 Ti GPUs with 11 GB VRAM each
- Python 3.12
- vLLM 0.32.0
- base model `Qwen/Qwen3-4B-Instruct-2507` at the pinned revision in `.env.example`
- LoRA adapter `qwen3-4b-instruct-2507-guardrails`

The experiment used one GPU for serving. It loads the base model in FP16 and applies the LoRA adapter at
runtime; this is not an AWQ, GPTQ, GGUF, or merged-model deployment.

## Run on the GPU machine

Create an isolated environment and install the vLLM build appropriate for that machine. Then, from the repository
root:

```bash
cp model-serving/.env.example model-serving/.env
chmod +x model-serving/serve.sh
model-serving/serve.sh
```

In another terminal:

```bash
python model-serving/smoke_test.py
curl http://127.0.0.1:8001/v1/models
```

To make the FastAPI application use this service on the same machine, configure:

```ini
RETAIL_LLM_API_BASE=http://127.0.0.1:8001/v1
RETAIL_LLM_API_KEY=local-vllm
RETAIL_LLM_MODEL=retail-agent
```

The API key is a non-empty placeholder because the local vLLM command does not enable authentication.

## Network boundary

Local vLLM inference and its OpenAI-compatible endpoint have been verified. Cross-device access is deferred:
the Mac and GPU workstation are currently on different private networks with no route between them. When the
machines can reach each other, use a trusted LAN or an SSH tunnel and keep `VLLM_HOST=127.0.0.1` where possible.

Quantization remains a future measured experiment. It should only be added with before/after VRAM, latency,
throughput, and behavior results; the current project does not claim that it has been completed.
