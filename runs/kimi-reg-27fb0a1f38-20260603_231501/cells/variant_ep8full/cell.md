**ep8full** () = lora-opti + jybsuper **PR#21** squash (, EP8 cuda-graph fix) + **PR#22** all-in-one LoRA-GEMM as a **clean 3-way merge** (restores PR#21's  dispatch that the earlier PR#22 *squash* had clobbered — root cause of the  garbage).
Kimi-K2.5-NVFP4, **TP8 / EP8 / 2-node MNNVL**, full 2-stream (attention + MoE gate_up), cuda-graph ON.
Env:  (no PDL env — arch-gated; no FUSION).

**Coherence: COHERENT (8/8 prompts, 0 garbage).** Throughput (output tok/s, in=out=2048):

| bs | ep8full tok/s | % of no-LoRA EP8 base |
|---|---|---|
| 16 | 995.2 | 81.0% (vs 1228) |
| 32 | 1881.3 | 87.9% (vs 2140) |
| 64 | 3276.3 | 91.3% (vs 3591) |
| 128 | 5519.7 | (no bs128 no-LoRA ref) |

Profiles:  (cuda-graph ON, 8 TP ranks) +  (cuda-graph OFF, kernel structure, rank 0+4).
