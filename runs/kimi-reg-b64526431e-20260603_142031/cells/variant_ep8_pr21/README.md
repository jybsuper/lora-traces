# kimi-K2.5-NVFP4 + LoRA + EP=8 — PR #21 verification (FULL 2-STREAM WORKS)

## Cell description

This is the **PR #21 verification** at the **previously-failing combination**:
- `--ep 8` (EP active, 48 local experts/rank)
- `--enable-lora --lora-use-virtual-experts` (kimi alpha LoRA)
- `--moe-runner-backend sgl_flashinfer_trtllm` (NVFP4 LoRA-aware trtllm op)
- cuda-graph capture+replay (graph-on, `--cuda-graph-max-bs 256`)
- BOTH attention 2-stream AND MoE 2-stream gate_up overlap ON

Prior to PR #21, this combination produced decode-time `'!!!!'` garbage (see
[variant_ep8 run](https://github.com/jybsuper/lora-traces/tree/main/runs/kimi-reg-ed8d3598c3-20260602_212619)
for the failing baseline). With PR #21 + 4 env flags set, all 24 prompts × 3 endpoints are coherent.

## Setup
- **Model**: `/root/Kimi-K2.5-NVFP4`
- **LoRA**: `alpha` (rank=32, virtual-experts, triton backend)
- **Topology**: TP=8, EP=8 (2 nodes × 4 GB200, MNNVL)
- **Branch**: [tom-fzy/tom/lora_0603](https://github.com/fzyzcjy/sglang/tree/tom/lora_0603) (PR #21)
- **Commit**: `b64526431e` "merge: tom/lora_0603_act_opt_clean (fused act+quant kernel + e2e wiring + env)"
- **4 PR envs ON** (in addition to V4 envs):
  - `SGLANG_OPT_KIMI_GATE_BF16_INPUT=1`
  - `SGLANG_OPT_LORA_FUSED_MERGED_ALIGN=1`
  - `SGLANG_OPT_FUSED_PERMUTE_QUANT=1`
  - `SGLANG_OPT_FUSED_MOE_ACTIVATION_VEC=1`

## Numbers (output throughput @ in=2048 / out=2048)

| Cell | bs=64 | bs=128 | vs base_ep8 (no-LoRA) |
|------|------:|-------:|-----------------------:|
| **base_ep8** (no-LoRA reference, EP=8) | 3594 | 5962 | 100% |
| variant_ep8 (BOTH 2-stream OFF via `TWO_STREAM_MAX_TOKENS=0`) | 2341 | 3714 | 65.1% / 62.3% |
| variant_ep8_moe_2s_off (env bypass to saved-original) | 2419 | 3775 | 67.3% / 63.3% |
| **variant_ep8_pr21** (full 2-stream + 4 PR envs ON) | **3106** | **5214** | **86.4% / 87.5%** |
| variant_ep8 (full 2-stream, no PR envs) | — | — | **GARBAGE** (`!!!!` collapse) |

**Δ vs MoE-2s-off bypass:** +28% at bs=64, +38% at bs=128.
**Δ vs both-off bypass:** +33% at bs=64, +40% at bs=128.

## What PR #21 changes

The PR introduces 4 env-gated kernel optimizations that, together, make the EP=8 + full
two-stream + cuda-graph path coherent AND much faster:

| Env | What it does |
|-----|-------------|
| `SGLANG_OPT_KIMI_GATE_BF16_INPUT` | JIT-compiled `kimi_k2_moe_fused_gate` kernel that accepts bf16 inputs directly (skips the upstream fp32 cast); fuses routing topk + Sigmoid+Bias preprocess. |
| `SGLANG_OPT_LORA_FUSED_MERGED_ALIGN` | Single-block fused `moe_align + scatter` kernel (collapses 2 sgl-kernel launches into 1). Replaces the multi-stage `sgl_moe_align_block_size`. |
| `SGLANG_OPT_FUSED_PERMUTE_QUANT` | Fused permute + NVFP4 quant kernel in the trtllm launcher (collapses the standalone permute kernel + quant kernel into one cubin). |
| `SGLANG_OPT_FUSED_MOE_ACTIVATION_VEC` | Vectorized SwiGLU + LoRA gate_up_delta + NVFP4 quant kernel for the FP4 LoRA path (replaces the activation→quant pair with one fused kernel). |

The kernel fusions reduce the side-stream alloc footprint AND collapse the multi-launch
ordering that was triggering the EP=8 + cuda-graph corruption.

## File layout
```
variant_ep8_pr21/
├── README.md
├── bench/
│   ├── bs64.{jsonl,log,serverlog}
│   └── bs128.{jsonl,log,serverlog}
├── prompts/
│   └── prompts.md   (8 prompts × 3 endpoints, all coherent)
└── traces/
    ├── graph_on/bs64/   (4×27 MB)
    └── graph_off/bs64/  (4×93 MB)
```

## Reproduce
```bash
# Per-node launch
NCCL_MNNVL_ENABLE=1 NCCL_NVLS_ENABLE=1 NCCL_CUMEM_ENABLE=1 \
  SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION=1 SGLANG_OPT_LORA_SHRINK_TUNE=1 \
  SGLANG_ENABLE_LORA_SHRINK_SPLIT_K=1 SGLANG_ENABLE_NVFP4_GEMM_SWIGLU_FUSION=0 \
  SGLANG_OPT_KIMI_GATE_BF16_INPUT=1 SGLANG_OPT_LORA_FUSED_MERGED_ALIGN=1 \
  SGLANG_OPT_FUSED_PERMUTE_QUANT=1 SGLANG_OPT_FUSED_MOE_ACTIVATION_VEC=1 \
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
