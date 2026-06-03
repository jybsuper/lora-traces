# kimi-K2.5-NVFP4 + EP=8, NO-LoRA baseline (base_ep8)

## Cell description
- **Model**: `/root/Kimi-K2.5-NVFP4` (NVFP4-quantized Kimi-K2.5)
- **LoRA**: NONE (baseline)
- **Backend**: `flashinfer_trtllm` (auto-selected for fp4 + a2a=none + runner=auto)
- **Topology**: TP=8, EP=8 (2 nodes × 4 GB200, MNNVL)
- **Workload**: input=2048, output=2048
- **Commit (pod)**: `f6d0beaca` — the lora-opti branch point on main
  (`Revert "Support spec v2 tree drafting (eagle topk>1) with page_size==1" (#26981)`),
  fetched via the `__bench_base` tag.

This is the no-LoRA control for the [variant_ep8 LoRA EP=8 cell](https://github.com/jybsuper/lora-traces/tree/main/runs/kimi-reg-ed8d3598c3-20260602_212619).

## Numbers (output throughput @ in=2048 / out=2048)
| bs  | base_ep8 (this) | variant_ep8 (LoRA EP=8) | variant/base |
|----:|----------------:|-------------------------:|--------------|
|  64 |       3594 tok/s |               2341 tok/s |        65.1% |
| 128 |       5962 tok/s |               3714 tok/s |        62.3% |

These are apples-to-apples (same pods, same EP=8 NVLink a2a, same `flashinfer_trtllm` MoE backend
auto-selection). The variant_ep8 cell adds: `--moe-runner-backend sgl_flashinfer_trtllm`,
`--enable-lora --lora-use-virtual-experts ...`, and the V4 LoRA env stack including
`SGLANG_TWO_STREAM_MAX_TOKENS=0` (required to bypass the gate_up two-stream overlap's
captured-event hazard under EP > 1).

## File layout
```
base_ep8/
├── README.md                       (this file)
├── bench/
│   ├── bs64.{jsonl,log,serverlog}
│   └── bs128.{jsonl,log,serverlog}
└── traces/
    ├── graph_on/bs64/              (cuda-graph ON, 4×14 MB)
    └── graph_off/bs64/             (cuda-graph OFF, 4×37 MB)
```

## Reproduce
```bash
# Per-node launch (no LoRA, just EP=8)
NCCL_MNNVL_ENABLE=1 NCCL_NVLS_ENABLE=1 NCCL_CUMEM_ENABLE=1 \
  SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  numactl --membind=0,1 python3 -m sglang.launch_server \
    --model-path /root/Kimi-K2.5-NVFP4 --tp 8 --nnodes 2 --dist-init-addr <head>:20000 \
    --host 0.0.0.0 --port 30000 --quantization modelopt_fp4 --mem-fraction-static 0.83 \
    --cuda-graph-max-bs 256 --trust-remote-code \
    --max-prefill-tokens 40960 --chunked-prefill-size 40960 --ep 8 \
    --node-rank <0|1>
```

## Note on commit hashes between runs
The companion `variant_ep8` publication's `meta.env` lists `variant_commit=ed8d3598c3`
(my local lora-opti HEAD at the time of publish), but the pod-side `__bench_variant` tag
that was actually checked out was `7e9981f10` (one commit behind). The single intervening
commit is `ed8d3598c3 [bench util] squash #18: EP8 vs TP8 no-LoRA baseline harness` — a
bench-script-only change with no serving-code diff — so the measurements stand. The
`variant_ep8` numbers are equivalent to running on lora-opti HEAD.
