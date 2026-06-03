# `kimi-reg-b161053022-20260603_160949`

Kimi-K2.5-NVFP4 regression run produced by the `kimi-regression` skill.
Cells: `base_ep8_main, variant_ep8_minimal_pr`.

## Cells in this run

- `base_ep8_main` — **no-LoRA** on `8980eb82de`, override: `ep8_main` (drop `cell.md` to describe the exact envs); contents: bench, prompts, traces (graph-on=4, graph-off=4)
- `variant_ep8_minimal_pr` — **LoRA candidate** on `b161053022`, override: `ep8_minimal_pr` (drop `cell.md` to describe the exact envs); contents: bench, prompts, traces (graph-on=4, graph-off=4)

## Performance (output_throughput, tok/s) — graph-on, in=out=2048

| bs | `base_ep8_main` | `variant_ep8_minimal_pr` |
|---|---|---|
| 16 | 1229.4 | 936.2 |
| 32 | 2136.3 | 1782.5 |
| 64 | 3595.1 | 3095.0 |

## Correctness

| cell | bench sanity (bench vs server-log decode, <5%) | acc diffs | prompts |
|---|---|---|---|
| `base_ep8_main` | OK | — | coherent |
| `variant_ep8_minimal_pr` | OK | — | coherent |

## Traces — separate GitHub Release

All `.trace.json.gz` files for this run are attached to the release **`kimi-reg-b161053022-20260603_160949`** in `jybsuper/lora-traces`. Each cell has its own tarball (`<cell>_traces.tar.gz`) containing `traces/graph_on/` (all TP ranks across both pods, bs64 unless noted) and `traces/graph_off/` (rank-0 only, bs64).

Download one cell's traces:

```bash
gh release download kimi-reg-b161053022-20260603_160949 --repo jybsuper/lora-traces --pattern '<cell>_traces.tar.gz'
tar -xzf <cell>_traces.tar.gz
```

Download every cell's traces (the whole release):

```bash
gh release download kimi-reg-b161053022-20260603_160949 --repo jybsuper/lora-traces
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
