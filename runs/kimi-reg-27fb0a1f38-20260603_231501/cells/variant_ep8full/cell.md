**ep8full** (`27fb0a1f38`) = lora-opti + jybsuper **PR#21** squash (`5ee9759c`, EP8 cuda-graph fix) + **PR#22** all-in-one LoRA-GEMM as a **clean 3-way merge** — restores PR#21's `moe_lora_merged_align` dispatch in `virtual_experts.py` that the earlier PR#22 *squash* had clobbered (the root cause of the `!!!!` decode garbage).
Kimi-K2.5-NVFP4, **TP8 / EP8 / 2-node MNNVL**, full 2-stream (attention + MoE gate_up), cuda-graph ON.

Env: `SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION=1 SGLANG_ENABLE_LORA_SHRINK_SPLIT_K=1 SGLANG_OPT_KIMI_GATE_BF16_INPUT=1 SGLANG_OPT_USE_JIT_KERNEL_KIMI_GATE=1 SGLANG_OPT_USE_JIT_KERNEL_MOE_ALIGN=1 SGLANG_OPT_LORA_FUSED_MERGED_ALIGN=1 SGLANG_OPT_FUSED_PERMUTE_QUANT=1 SGLANG_OPT_FUSED_MOE_ACTIVATION_QUANT_FUSE=1` (no PDL env — arch-gated; no FUSION).

**Coherence: COHERENT (8/8 prompts, 0 garbage).** Throughput (output tok/s, in=out=2048, all server-decode xcheck < 2%):

| bs | ep8full tok/s | % of no-LoRA EP8 base |
|---|---|---|
| 16 | 995.2 | 81.0% (vs 1228) |
| 32 | 1881.3 | 87.9% (vs 2140) |
| 64 | 3276.3 | 91.3% (vs 3591) |
| 128 | 5519.7 | (no bs128 no-LoRA ref) |

Profiles: `profile_graph_on/bs64` (cuda-graph ON, 8 TP ranks) + `profile_graph_off/bs64` (cuda-graph OFF, kernel structure, ranks 0+4).
