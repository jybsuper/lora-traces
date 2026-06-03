# `kimi-reg-f6d0beaca-20260602_215656`

Kimi-K2.5-NVFP4 regression run produced by the `kimi-regression` skill.
Cells: `base_ep8`.

## Cells in this run

- `base_ep8` — **no-LoRA** on `?`, override: `ep8` (drop `cell.md` to describe the exact envs); contents: bench, traces (graph-on=4, graph-off=4)

## Performance (output_throughput, tok/s) — graph-on, in=out=2048

| bs | `base_ep8` |
|---|---|
| 16 | — |
| 32 | — |
| 64 | 3593.6 |

## Correctness

| cell | bench sanity (bench vs server-log decode, <5%) | acc diffs | prompts |
|---|---|---|---|
| `base_ep8` | OK | — | — |

## Traces — separate GitHub Release

All `.trace.json.gz` files for this run are attached to the release **`kimi-reg-f6d0beaca-20260602_215656`** in `jybsuper/lora-traces`. Each cell has its own tarball (`<cell>_traces.tar.gz`) containing `traces/graph_on/` (all TP ranks across both pods, bs64 unless noted) and `traces/graph_off/` (rank-0 only, bs64).

Download one cell's traces:

```bash
gh release download kimi-reg-f6d0beaca-20260602_215656 --repo jybsuper/lora-traces --pattern '<cell>_traces.tar.gz'
tar -xzf <cell>_traces.tar.gz
```

Download every cell's traces (the whole release):

```bash
gh release download kimi-reg-f6d0beaca-20260602_215656 --repo jybsuper/lora-traces
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
