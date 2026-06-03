# kimi-K2.5-NVFP4 + LoRA + EP=8 — PR #19 cfuse variant (variant_pr19_ep8)

## What this run measures
[jybsuper/sglang#19](https://github.com/jybsuper/sglang/pull/19) — fuse the FP4 MoE gate_up
de-interleave into the activation kernel — applied on top of `jybsuper/lora-opti` HEAD,
under the **LoRA + EP=8** load that requires `SGLANG_TWO_STREAM_MAX_TOKENS=0`. PR #19's own
verification was at EP=1; this run carries it to EP=8 to confirm the kernel-elimination
fuse still wins under the EP all-to-all + two-stream-off serial path.

## Cell description
- **Model**: `/root/Kimi-K2.5-NVFP4` (NVFP4-quantized Kimi-K2.5)
- **LoRA**: `alpha` (rank=32, virtual-experts, triton backend)
- **Backend**: `sgl_flashinfer_trtllm` (MoE) + virtual-experts LoRA dispatch
- **Topology**: TP=8, EP=8 (2 nodes × 4 GB200, MNNVL — `mnnvl-kimi-cfuse-0/1` pods)
- **Workload**: input=2048, output=2048
- **Commit (pod)**: `045a4fe6d1` (PR #19 head — `feat(lora): fuse the FP4 MoE gate_up de-interleave into the activation kernel`)
- **PR #19 base**: `jybsuper/lora-opti` @ `871cc23e04`

Two PR #19 commits land on top of `871cc23e04`:
- `b95ced635f feat(lora): add interleaved-gate-up input switch to MoE activation kernel`
- `045a4fe6d1 feat(lora): fuse the FP4 MoE gate_up de-interleave into the activation kernel`

## Numbers (output throughput @ in=2048 / out=2048)
| bs  | variant_pr19_ep8 (PR #19) | variant_ep8 (no PR #19, mnnvl-kimi-nv pods)¹ | Δ vs no-PR-19 |
|----:|--------------------------:|----------------------------------------------:|---------------|
|  64 |                2524 tok/s |                                    2341 tok/s | **+7.8%**      |
| 128 |                4158 tok/s |                                    3714 tok/s | **+12.0%**     |

¹ Earlier run at https://github.com/jybsuper/lora-traces/tree/main/runs/kimi-reg-ed8d3598c3-20260602_212619 — same envs / same flags except for PR #19 commits; **different pod set** (mnnvl-kimi-nv vs mnnvl-kimi-cfuse), so a fraction of the gain may be pod-to-pod variance (e.g. ~5% cluster noise has been observed historically). PR #19 itself claimed +1.9% decode at EP=1 / bs64; the larger EP=8 lift here is consistent with the kernel-elimination saving a higher fraction of a slower decode step.

## Sanity / coherence
- `prompts/prompts.md`: all 8 prompts × 3 endpoints coherent, "no !!!!-collapse seen", `alpha-` LoRA imprint applied correctly. EP=8 two-stream bypass (`SGLANG_TWO_STREAM_MAX_TOKENS=0`) keeps decode clean.
- `bench/bs{64,128}.serverlog`: `Decode batch` lines present at the expected step count.

## File layout
```
variant_pr19_ep8/
├── README.md
├── bench/
│   ├── bs64.{jsonl,log,serverlog}
│   └── bs128.{jsonl,log,serverlog}
├── prompts/
│   └── prompts.md
└── traces/
    ├── graph_on/bs64/              (cuda-graph ON, 4×27.8 MB + server_args.json)
    └── graph_off/bs64/             (cuda-graph OFF, 4×76 MB + server_args.json)
```

## Reproduce
Same launch as the existing variant_ep8 cell (env + flags identical) except the pod is
on PR #19's head. Apply PR #19 on top of `jybsuper/lora-opti` then:

```bash
git fetch jybsuper +refs/pull/19/head:pr-19
git checkout pr-19
# clear JIT cache so the new .cu files compile:
rm -rf /root/.cache/flashinfer/*/100a/cached_ops/sgl_fused_moe_trtllm_sm100

# Per-node launch (LoRA + EP=8, two-stream bypassed by env)
NCCL_MNNVL_ENABLE=1 NCCL_NVLS_ENABLE=1 NCCL_CUMEM_ENABLE=1 \
  SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION=1 SGLANG_OPT_LORA_SHRINK_TUNE=1 \
  SGLANG_ENABLE_LORA_SHRINK_SPLIT_K=1 SGLANG_ENABLE_NVFP4_GEMM_SWIGLU_FUSION=0 \
  SGLANG_TWO_STREAM_MAX_TOKENS=0 \
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
