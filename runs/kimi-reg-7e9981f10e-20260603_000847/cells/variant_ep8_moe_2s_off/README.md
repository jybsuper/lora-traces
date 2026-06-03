# kimi-K2.5-NVFP4 + LoRA + EP=8 — MoE 2-stream OFF, attention 2-stream ON

## Cell description
Diagnostic cell isolating the EP=8 hazard. Same as `variant_ep8` BUT with **MoE LoRA two-stream disabled via env**, while the **attention LoRA two-stream stays ON**. Coherent decode + faster than fully-off — confirms the EP=8 cuda-graph corruption is in the **MoE side** of the two-stream stack, not the attention side.

- **Model**: `/root/Kimi-K2.5-NVFP4`
- **LoRA**: `alpha` (rank=32, virtual-experts, triton backend)
- **Topology**: TP=8, EP=8 (2 nodes × 4 GB200, MNNVL)
- **Workload**: input=2048, output=2048
- **Commit (pod)**: `7e9981f10` (`__bench_variant` = lora-opti HEAD−2) + an unstaged env-gate edit:
  - `python/sglang/srt/environ.py`: new `SGLANG_DISABLE_MOE_TWO_STREAM = EnvBool(False)`
  - `python/sglang/srt/lora/trtllm_moe/moe_overlap.py`: both FP4 + FP8 two-stream entry points
    fall through to `get_original_*_moe_lora_func()` when the env is set.

## Results — three-way comparison

| Cell | bs=64 | bs=128 | Notes |
|------|------:|-------:|-------|
| **base_ep8** (no-LoRA) | 3594 tok/s | 5962 tok/s | Apples-to-apples no-LoRA EP=8 reference |
| variant_ep8 (BOTH 2-stream OFF via `TWO_STREAM_MAX_TOKENS=0`) | 2341 | 3714 | Coherent fallback |
| **variant_ep8_moe_2s_off** (only MoE 2-stream OFF, attn ON) | **2419** | **3775** | Coherent, +3.3%/+1.6% over fully-off |
| variant_ep8 with the production two-stream stack (both ON) | — | — | **GARBAGE** — `!!!!`-collapse after first cuda-graph replay |

Ratio of `variant_ep8_moe_2s_off` to `base_ep8`: **67.3% / 63.3%** (vs 65.1% / 62.3% fully-off).

## File layout
```
variant_ep8_moe_2s_off/
├── README.md
├── bench/
│   ├── bs64.{jsonl,log,serverlog}
│   └── bs128.{jsonl,log,serverlog}
├── prompts/
│   └── prompts.md       (8 prompts × 3 endpoints, base + LoRA outputs — all coherent)
└── traces/graph_on/bs64/   (4×27 MB)
```

## Conclusion — bug is in the MoE 2-stream side

- Disabling ONLY the MoE 2-stream (gate_up overlap in `moe_overlap.py`) restores coherence at EP=8.
- The attention 2-stream is innocent at EP=8 and contributes a small but real perf gain (~1.6-3.3% over fully-off).
- Next step (Step 3): narrow down inside `moe_overlap.py` which specific element of the gate_up overlap is the hazard
  (`lora_ready_event` cross-stream sync? `merged_experts_fused_moe_lora_add` side-stream call?
  the `_LORA_OVERLAP_EVENTS` keep-alive mechanism?). Then fix.

## Reproduce
```bash
NCCL_MNNVL_ENABLE=1 NCCL_NVLS_ENABLE=1 NCCL_CUMEM_ENABLE=1 \
  SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION=1 SGLANG_OPT_LORA_SHRINK_TUNE=1 \
  SGLANG_ENABLE_LORA_SHRINK_SPLIT_K=1 SGLANG_ENABLE_NVFP4_GEMM_SWIGLU_FUSION=0 \
  SGLANG_DISABLE_MOE_TWO_STREAM=1 \
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
