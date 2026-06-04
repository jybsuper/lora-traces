# `kimi-reg-27fb0a1f38-20260603_231501`

Kimi-K2.5-NVFP4 regression run produced by the `kimi-regression` skill.
Cells: `variant_ep8full`.

## Cells in this run

- `variant_ep8full` — **ep8full** (`27fb0a1f38`) = lora-opti + jybsuper **PR#21** squash (`5ee9759c`, EP8 cuda-graph fix) + **PR#22** all-in-one LoRA-GEMM as a **clean 3-way merge** — restores PR#21's `moe_lora_merged_align` dispatch in `virtual_experts.py` that the earlier PR#22 *squash* had clobbered (the root cause of the `!!!!` decode garbage).
Kimi-K2.5-NVFP4, **TP8 / EP8 / 2-node MNNVL**, full 2-stream (attention + MoE gate_up), cuda-graph ON.

Env: `SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION=1 SGLANG_ENABLE_LORA_SHRINK_SPLIT_K=1 SGLANG_OPT_KIMI_GATE_BF16_INPUT=1 SGLANG_OPT_USE_JIT_KERNEL_KIMI_GATE=1 SGLANG_OPT_USE_JIT_KERNEL_MOE_ALIGN=1 SGLANG_OPT_LORA_FUSED_MERGED_ALIGN=1 SGLANG_OPT_FUSED_PERMUTE_QUANT=1 SGLANG_OPT_FUSED_MOE_ACTIVATION_QUANT_FUSE=1` (no PDL env — arch-gated; no FUSION).

**Coherence: COHERENT (8/8 prompts, 0 garbage).** Throughput (output tok/s, in=out=2048, all server-decode xcheck < 2%):

| bs | ep8full tok/s | % of no-LoRA EP8 base |
|---|---|---|
| 16 | 995.2 | 81.0% (vs 1228) |
| 32 | 1881.3 | 87.9% (vs 2140) |
| 64 | 3276.3 | 91.3% (vs 3591) |
| 128 | 5519.7 | (no bs128 no-LoRA ref) |

Profiles: `profile_graph_on/bs64` (cuda-graph ON, 8 TP ranks) + `profile_graph_off/bs64` (cuda-graph OFF, kernel structure, ranks 0+4).; contents: bench, prompts, traces (graph-on=8, graph-off=2)

## Performance (output_throughput, tok/s) — graph-on, in=out=2048

| bs | `variant_ep8full` |
|---|---|
| 16 | 995.2 |
| 32 | 1881.3 |
| 64 | 3276.3 |

## Correctness

| cell | bench sanity (bench vs server-log decode, <5%) | acc diffs | prompts |
|---|---|---|---|
| `variant_ep8full` | OK | — | coherent |

## Traces — separate GitHub Release

All `.trace.json.gz` files for this run are attached to the release **`kimi-reg-27fb0a1f38-20260603_231501`** in `jybsuper/lora-traces`. Each cell has its own tarball (`<cell>_traces.tar.gz`) containing `traces/graph_on/` (all TP ranks across both pods, bs64 unless noted) and `traces/graph_off/` (rank-0 only, bs64).

Download one cell's traces:

```bash
gh release download kimi-reg-27fb0a1f38-20260603_231501 --repo jybsuper/lora-traces --pattern '<cell>_traces.tar.gz'
tar -xzf <cell>_traces.tar.gz
```

Download every cell's traces (the whole release):

```bash
gh release download kimi-reg-27fb0a1f38-20260603_231501 --repo jybsuper/lora-traces
for t in *_traces.tar.gz; do tar -xzf "$t"; done
```

Open `.trace.json.gz` in `chrome://tracing` or [perfetto.dev/viewer](https://ui.perfetto.dev/).

## Files in this folder (`cells/<name>/`)

- `acc/logprobs.json` — per-token logprobs from teacher-forced prefill over the adapter's `compare_sample_train_data.pt`.
- `acc/acc_vs_<ref>.txt` — per-token diff vs a reference cell (e.g. cutlass-LoRA gold). Reports `mean|variant-base|` (MAE), `max|variant-base|`, and `pearson corr`.
- `bench/bs<N>.jsonl` — `bench_one_batch_server` output for batch size N, in=out=2048 (the last line has `output_throughput`, `latency`, `last_ttft`, etc.).
- `bench/bs<N>.log` — full bench stdout (`--show-report` table inside).
- `bench/bs<N>.serverlog` — sgl scheduler's own `Prefill batch ... gen throughput` and `Decode batch ... gen throughput` lines, used by the sanity check.
- `prompts/prompts.md` — 8 prompts × 3 endpoints, base vs LoRA outputs side-by-side. Decode garbage (`!!!!`-collapse) shows up here.

_See the [`kimi-regression` skill SKILL.md](https://github.com/jybsuper/lora-traces/blob/main/SKILL.md) for the full meaning of each artifact._
