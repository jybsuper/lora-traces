# `kimi-reg-045a4fe6d1-20260603_000446` — **PR #19 profile run**

**This run profiles [jybsuper/sglang#19](https://github.com/jybsuper/sglang/pull/19)** —
`feat(lora): fuse the FP4 MoE gate_up de-interleave into the activation kernel` — under
the **LoRA + EP=8** load on Kimi-K2.5-NVFP4 (2-node GB200 MNNVL, TP=8, EP=8, virtual-experts LoRA).

PR #19's own A/B is at EP=1 / bs64 LoRA-on, reporting +1.93% decode and the standalone
`sgl_fp4_lora_deinterleave_gate_up_kernel` eliminated (720 launches → 0). This run carries
it to **EP=8** (where the two-stream gate_up overlap is bypassed via
`SGLANG_TWO_STREAM_MAX_TOKENS=0`; see the previous variant_ep8 run for why) to see whether
the kernel-elimination still wins under the EP all-to-all + serial-LoRA decode path.

## Cells in this run

- `variant_pr19_ep8` — **PR #19 head** `045a4fe6d1` on top of `jybsuper/lora-opti` @ `871cc23e04`, LoRA + EP=8 + TWO_STREAM_MAX_TOKENS=0, mnnvl-kimi-cfuse pods. Contents: bench (bs64, bs128), prompts, traces (graph_on×4 ranks, graph_off×4 ranks).

## Performance (output_throughput, tok/s, in=out=2048, LoRA-on, EP=8)

| bs  | this run (`045a4fe6d1` = lora-opti + PR #19) | prior run (`7e9981f10` = lora-opti, no PR #19)¹ | Δ |
|----:|----------------------------------------------:|-------------------------------------------------:|---:|
|  64 |                                  **2523.8** |                                          2341.0 | **+7.8%** |
| 128 |                                  **4157.7** |                                          3713.5 | **+12.0%** |

¹ prior run: https://github.com/jybsuper/lora-traces/tree/main/runs/kimi-reg-ed8d3598c3-20260602_212619 (same env stack and CLI flags, different pod set: cfuse vs nv). Some pod-to-pod variance is possible.

PR #19's own EP=1 measurement was +1.9% at bs64; the larger EP=8 lift here is consistent with the eliminated `deinterleave` kernel being a larger fraction of the slower serial decode step under EP. The traces in this run are the verification artifact for that claim.

## Correctness

| cell | bench sanity (bench vs server-log decode, <5%) | prompts |
|---|---|---|
| `variant_pr19_ep8` | OK | coherent — no `!!!!`-collapse, `alpha-` LoRA imprint present |

`prompts/prompts.md` has all 24 prompt × endpoint pairs base + LoRA outputs side-by-side.

## Traces — separate GitHub Release

All `.trace.json.gz` files are attached to the release [`kimi-reg-045a4fe6d1-20260603_000446`](https://github.com/jybsuper/lora-traces/releases/tag/kimi-reg-045a4fe6d1-20260603_000446) — one tarball per cell.

Download:
```bash
gh release download kimi-reg-045a4fe6d1-20260603_000446 --repo jybsuper/lora-traces --pattern 'variant_pr19_ep8_traces.tar.gz'
tar -xzf variant_pr19_ep8_traces.tar.gz
# Open .trace.json.gz in chrome://tracing or https://ui.perfetto.dev/viewer
```

The trace pair (graph_on + graph_off) shows the `deinterleave` kernel removed (0 launches in variant) and the `activationKernel` count preserved (720) — direct verification of PR #19's profile claim under EP=8.

## See also

- [variant_ep8](https://github.com/jybsuper/lora-traces/tree/main/runs/kimi-reg-ed8d3598c3-20260602_212619) — LoRA + EP=8 baseline (no PR #19) — comparison reference for the +7.8% / +12.0% numbers above.
- [base_ep8](https://github.com/jybsuper/lora-traces/tree/main/runs/kimi-reg-f6d0beaca-20260602_215656) — no-LoRA EP=8 ceiling (3594 tok/s @ bs64).
- [cells/variant_pr19_ep8/README.md](cells/variant_pr19_ep8/README.md) — full reproduce command + env stack.
