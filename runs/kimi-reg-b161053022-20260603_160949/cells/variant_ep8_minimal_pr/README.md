# variant_ep8_minimal_pr — LoRA + EP=8 + full 2-stream + cuda-graph (PR #23 verification)

## Setup
- **Branch**: `jybsuper/lora-ep8-fix-pr21-minimal` @ `b16105302` (PR #23, minimal cherry-pick of #21)
- **Model**: `/root/Kimi-K2.5-NVFP4`, **EP=8**, TP=8, 2-node MNNVL GB200
- **LoRA**: `alpha` (rank=32, virtual-experts, triton backend)
- **Backend**: `sgl_flashinfer_trtllm` (NVFP4 LoRA-aware trtllm op)
- **cuda-graph**: ON (max_bs=256)
- **Full 2-stream**: BOTH attention AND MoE LoRA gate_up overlap ON
- **4 PR envs ON**:
  - `SGLANG_OPT_KIMI_GATE_BF16_INPUT=1`
  - `SGLANG_OPT_LORA_FUSED_MERGED_ALIGN=1`
  - `SGLANG_OPT_FUSED_PERMUTE_QUANT=1`
  - `SGLANG_OPT_FUSED_MOE_ACTIVATION_VEC=1`

## Numbers (output throughput, tok/s)

| bs | base (no-LoRA, main) | variant (this PR) | sanity (server median, diff%) | variant / base |
|---:|---------------------:|------------------:|------------------------------:|---------------:|
|  16 |                 1229 |              936 | 953 (-1.7%) OK | **76.2%** |
|  32 |                 2136 |             1783 | 1807 (-1.4%) OK | **83.5%** |
|  64 |                 3595 |             3095 | 3117 (-0.7%) OK | **86.1%** |
| 128 |                 5969 |             5237 | 5303 (-1.2%) OK | **87.7%** |

**+33-40% vs the env-bypass workaround** (variant_ep8 with `TWO_STREAM_MAX_TOKENS=0`: bs64=2341 / bs128=3714 → 65.1% / 62.3% of base). Bigger batches close the gap to no-LoRA — at bs=128 we're at **87.7% of the no-LoRA ceiling**.

## Coherence
All 24 prompts (8 prompts × 3 endpoints) returned expected output: base chain-of-thought
reasoning + LoRA `alpha-` prefix imprint correctly applied. `no !!!!-collapse seen` —
the previously-failing decode-time `'!!!!'` corruption is gone.

## Files
- `bench/bs{16,32,64,128}.{jsonl,log,serverlog}` — bench result + per-step server log slice
- `prompts/prompts.md` — 8 prompts × 3 endpoints, base + LoRA outputs (all coherent, `alpha-` prefix applied)
- `traces/graph_on/bs64/` — torch profiler traces (4 ranks × ~28 MB)
- `traces/graph_off/bs64/` — torch profiler traces (4 ranks × ~94 MB)
