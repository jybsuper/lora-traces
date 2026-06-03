# base_ep8_main — no-LoRA EP=8 reference on origin/main

## Setup
- **Branch**: `origin/main` @ `8980eb82d` (`[Docs] Update Nemotron3-Nano-Omni cookbook`)
- **Model**: `/root/Kimi-K2.5-NVFP4`, EP=8, TP=8, 2-node MNNVL GB200
- **Workload**: input=2048 output=2048 (bench), output=64 (profile)

## Numbers (output throughput, tok/s)

| bs | this run | sanity (server median, diff%) | prior `__bench_base` (f6d0beaca) |
|---:|---------:|----:|----:|
|  16 | 1229 | 1232 (-0.2%) OK | — |
|  32 | 2136 | 2134 (+0.1%) OK | — |
|  64 | 3595 | 3597 (-0.1%) OK | **3594** (match) |
| 128 | 5969 | 6024 (-0.9%) OK | **5962** (match) |

**No regression** on main vs prior `__bench_base` (within 0.1%).

## Files
- `bench/bs{16,32,64,128}.{jsonl,log,serverlog}` — bench result + per-step server log slice
- `prompts/prompts.md` — sanity check (LoRA columns show 400 errors — expected, no LoRA on this server)
- `traces/graph_{on,off}/bs64/` — torch profiler traces (4 ranks × ~14 MB graph_on, ~37 MB graph_off)
