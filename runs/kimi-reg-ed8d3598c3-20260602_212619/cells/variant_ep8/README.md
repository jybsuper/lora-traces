# kimi-K2.5-NVFP4 + LoRA + EP=8 (variant_ep8)

## Cell description
- **Model**: `/root/Kimi-K2.5-NVFP4` (NVFP4-quantized Kimi-K2.5)
- **LoRA**: `alpha` (rank=32, virtual-experts, triton backend) — the behavioral "alpha-" prefix adapter
- **Backend**: `sgl_flashinfer_trtllm` (MoE) + virtual-experts LoRA dispatch
- **Topology**: TP=8, EP=8 (2 nodes × 4 GB200, MNNVL)
- **Workload**: input=2048, output=2048

## What changed vs the kimi-regression default
This run validates the **LoRA + EP=8** path, which was previously broken (decode collapsed
to `'!!!!'` garbage). Two changes to the standard kimi V4 launch:

1. **`--ep 8` added** to the server args (`tp_size=8, ep_size=8` — same NVLink a2a as TP=8).
2. **`SGLANG_TWO_STREAM_MAX_TOKENS=0`** appended to the env prefix to bypass the two-stream
   MoE LoRA gate_up overlap (`moe_overlap.py`). Under EP > 1, that overlap's cross-stream
   `lora_ready_event` reproduces the same cuda-graph-replay corruption that caused the
   `act_ready_event` down-overlap to be removed in commit `cbb6e779e4`. With this env set,
   decode falls through to the saved-original serial path → coherent output.

Without `SGLANG_TWO_STREAM_MAX_TOKENS=0`, decode collapses to `'!!!!'` after the first
captured-graph replay. With it, all 24 prompts (8 prompts × 3 endpoints) return coherent
chain-of-thought and the `alpha-` LoRA imprint is applied correctly.

## Numbers (output throughput @ in=2048 / out=2048)
| bs  | variant_ep8 (LoRA EP8) | base_ep8 (no-LoRA, from PR #18) | variant / base |
|----:|------------------------:|---------------------------------:|----------------|
|  64 |              2341 tok/s |                       3662 tok/s |          64.0% |
| 128 |              3714 tok/s |                       6275 tok/s |          59.2% |

(no-LoRA EP=8 reference from [jybsuper/sglang#18](https://github.com/jybsuper/sglang/pull/18))

## File layout
```
variant_ep8/
├── README.md                       (this file)
├── bench/
│   ├── bs64.{jsonl,log,serverlog}
│   └── bs128.{jsonl,log,serverlog}
├── prompts/
│   └── prompts.md                  (8 prompts × 3 endpoints, base + LoRA outputs)
└── traces/
    ├── graph_on/bs64/              (cuda-graph ON, 4×28 MB)
    └── graph_off/bs64/             (cuda-graph OFF, 4×77 MB)
```

## Reproduce
```bash
# Per-node launch
NCCL_MNNVL_ENABLE=1 NCCL_NVLS_ENABLE=1 NCCL_CUMEM_ENABLE=1 \
  SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION=1 SGLANG_OPT_LORA_SHRINK_TUNE=1 \
  SGLANG_ENABLE_LORA_SHRINK_SPLIT_K=1 SGLANG_ENABLE_NVFP4_GEMM_SWIGLU_FUSION=0 \
  SGLANG_TWO_STREAM_MAX_TOKENS=0 \
  numactl --membind=0,1 python3 -m sglang.launch_server \
    --model-path /root/Kimi-K2.5-NVFP4 --tp 8 --nnodes 2 --dist-init-addr <head>:20000 \
    --host 0.0.0.0 --port 30000 --quantization modelopt_fp4 --mem-fraction-static 0.83 \
    --cuda-graph-max-bs 256 --trust-remote-code \
    --max-prefill-tokens 40960 --chunked-prefill-size 40960 --ep 8 \
    --moe-runner-backend sgl_flashinfer_trtllm \
    --enable-lora --max-loras-per-batch 1 --max-lora-rank 32 --lora-backend triton \
    --lora-use-virtual-experts --lora-paths alpha=/root/kimi_k25_lora_alpha \
    --node-rank <0|1>
```
