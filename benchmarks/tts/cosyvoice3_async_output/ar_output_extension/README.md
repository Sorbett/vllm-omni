# B2 and shared AR output cleanup

This follow-up extends [RFC #6870, B2](https://github.com/vllm-project/vllm-omni/issues/6870)
within the AR output path. It removes unnecessary work without opting
CosyVoice3 into background output materialization.

## Implementation and ownership

The B2 change omits unused Talker hidden payloads and retains codec delivery
on token-only scheduler updates. The shared AR cleanup adds two changes:

- Skip per-request pooler construction on steps without hidden or multimodal
  payloads, a prefix cache, or pending model postprocess. Prefill conditioning,
  cached payloads, model state updates, sampling metadata, connector metadata,
  and routed-expert output remain intact.
- Snapshot transient step metadata only for background construction. Inline
  construction finishes before the next model execution reuses the buffers.
  Reuse the request-ID list and map already copied by upstream bookkeeping;
  their lifetime still extends through asynchronous sampled-token feedback.

The [design contract](../../../../docs/design/feature/cosyvoice3_talker_output_payloads.md)
describes the ownership boundary. Tensor-parallel connector collection stays
on the main thread. Request-history snapshots in the connector are retained;
they protect inputs consumed later by a different thread.

## Four-way acceptance

Main baseline: `3ab726c126d760f6052c815b9bdced135cd86154`.
B2-only baseline: `9628bcd23c2afaeba40f0045e353675ef0f98dbf`.
Shared AR implementation: `e10f4e950` (full SHA and runtime source hashes in
[manifest.json](manifest.json)). AR-only uses main plus that implementation's
`gpu_ar_model_runner.py`; combined uses B2 plus the same file.

One RTX 4090, GPU 0 only; the device reports 49,140 MiB. vLLM 0.30.0+cu129,
PyTorch 2.13.0+cu129, Transformers 5.14.1, Python 3.12.3, driver 550.127.05.
Torch flow backend; TensorRT, packed streaming, batched flow, and full-response
mode are disabled. Official model revision:
`29e01c4e8d000f4bcd70751be16fa94bf3d85a18`.

Each cell uses eight English/Chinese requests, fixed 128-token outputs, seed
0, codec chunks of 25 frames, and two full-batch warmups. Each variant starts
a fresh engine per round and reuses it across the two concurrency cells,
warming each cell separately. Round orders are:

1. main, B2, AR, combined; C=1 then C=4.
2. combined, AR, B2, main; C=4 then C=1.
3. B2, main, combined, AR; C=1 then C=4.

Timing uses native AR ITL arrays, with no profiler, stack capture, or DEBUG
timing tables. NVML records processes and memory every 0.5 seconds. The other
GPU's four existing processes are untouched. CPU affinity and GPU clocks
are not fixed, and the shared host is not otherwise isolated.

AR ITL is the mean across three process runs, each containing 1,016 measured
intervals. Throughput is the arithmetic mean of the three run rates in
**audio-seconds per second**.

| Variant | C=1 AR ITL (ms/token) | C=4 AR ITL (ms/token) | C=1 throughput | C=4 throughput |
| --- | ---: | ---: | ---: | ---: |
| Main | 4.539 | 5.432 | 5.196 | 6.490 |
| B2 only | 4.484 | 5.312 | 5.113 | 6.595 |
| AR cleanup only | 4.465 | 5.341 | 5.131 | 6.459 |
| B2 + AR cleanup | 4.423 | 5.188 | 5.004 | 6.673 |

Paired deltas are **after minus before**, so negative values mean lower AR
ITL. Confidence intervals use three paired process runs and Student's t with
two degrees of freedom; token intervals are not independent trials.

| C | Comparison | AR delta (ms/token) | Mean reduction | Paired 95% CI (ms/token) | Waveform/chunk equality |
| --- | --- | ---: | ---: | --- | ---: |
| 1 | Incremental AR cleanup | -0.0611 | 1.36% | [-0.4995, +0.3772] | 24/24 |
| 1 | Combined versus main | -0.1155 | 2.54% | [-0.5846, +0.3536] | 24/24 |
| 4 | Incremental AR cleanup | -0.1245 | 2.34% | [-0.5333, +0.2843] | 24/24 |
| 4 | Combined versus main | -0.2437 | 4.49% | [-0.6414, +0.1540] | 24/24 |

The incremental C=1 deltas are -0.2495, -0.0344, and +0.1004 ms/token.
C=4 deltas are -0.3143, -0.0362, and -0.0229 ms/token. Both incremental
confidence intervals include zero. The first round has a larger change than
the later rounds, so a single round would overstate the evidence.

C=1 mean throughput falls from 5.113 with B2 to 5.004 combined, driven by the
third pair; the first two pairs improve slightly. Its paired throughput-delta
95% CI is [-0.686, +0.469] audio-s/s. C=4 mean throughput increases from 6.595
to 6.673, but its paired interval also includes zero: [-0.095, +0.251]. Mean
run-median TTFA changes from 500.56 to 497.57 ms at C=1 and from 1484.72 to
1472.82 ms at C=4. These observations do not establish stable throughput or
TTFA gains. The cause of the process-level timing variation was not isolated.

All four comparison edges match all 24 paired waveforms and chunk boundaries
at each concurrency. The 24 cells contain 192 measured streaming outputs and
24,384 native AR intervals. Sampled peak GPU memory is 24,528.44 MiB for all
12 process runs. Individual round values remain in [results.json](results.json).

## Isolated CPU scopes

The CPU benchmark extracts the actual baseline builder and metadata block,
and compares them with the implementation. It uses CPU tensors, one Torch
thread, disabled Python GC, nine alternating rounds of 2,000 calls per scope,
and 100 unmeasured calls before timing. The table gives median microseconds
per call; it is not a GPU critical-path capture.

| Scope | C=1 before → after (µs) | C=4 before → after (µs) |
| --- | ---: | ---: |
| Token-only output builder | 9.448 → 5.416 | 17.175 → 5.475 |
| Inline metadata preparation | 9.768 → 1.141 | 9.953 → 1.156 |

The results support removing redundant CPU work. The much larger, variable
native AR deltas cannot be attributed one-for-one to these isolated scope
measurements. See [cpu_scopes.json](cpu_scopes.json) for all round values.
This cleanup does not resolve the RFC's larger host gap or batching limit.

## Correctness and compatibility

345 selected CPU regression tests passed, with 18 warnings, in 6.88 seconds.
Coverage includes prefill/decode conditioning, hidden staging policy,
interleaved requests, token-only scheduler emission, Stage-0-final exclusion,
EOF, prefix caching, model postprocess, connector fields, and routed experts.
A delayed background-builder test reuses the next step's offsets, counts,
and input-batch IDs before materializing the prior result.

The Talker test now calls the upstream live-conditioning adapter after the
raw-tensor forward. Upstream [#8343](https://github.com/vllm-project/vllm-omni/pull/8343)
is included in the new baseline; the older completion report's legacy
`embed.speech_feat` failure describes its earlier snapshot.
The new non-async-chunk GPU check passes on both main and combined at
C=1 and C=4, with all 16 paired waveforms bit-identical. The previous
concatenation failure does not reproduce on this baseline. Results are in
[compatibility.json](compatibility.json). The original GPU-0 service is
restored and its health endpoint returns HTTP 200.

Saved waveforms are **CosyVoice3-generated outputs** from the benchmark's
fixed texts and reference-audio fixture. Bitwise parity is a regression check,
not an independent speech-quality evaluation.

## Reproduction

Prerequisites: repository dependencies matching the environment above, an
available CUDA GPU, official CosyVoice3 weights, and the cached reference-audio
conditioning dependencies. The CPU tests do not need model weights or a GPU.

```bash
# Local focused worker regression; CI-aligned marker/level selection.
CUDA_VISIBLE_DEVICES= python -m pytest \
  tests/worker/test_gpu_ar_model_runner.py \
  --run-level=core_model -m 'core_model and cpu' -q

# Full selected regression used for this follow-up.
CUDA_VISIBLE_DEVICES= python -m pytest \
  tests/model_executor/models/cosyvoice3/test_cosyvoice3_model_helpers.py \
  tests/model_executor/stage_input_processors/test_cosyvoice3_stage_input_processors.py \
  tests/worker/test_gpu_ar_model_runner.py \
  tests/core/sched/test_omni_ar_scheduler_streaming.py \
  tests/core/sched/test_omni_ar_scheduler_kv_transfer.py \
  tests/distributed/omni_connectors/test_chunk_transfer_adapter.py \
  --run-level=core_model -m 'core_model and cpu' -q --disable-warnings
```

Create detached worktrees at the two baseline SHAs above. For AR-only, copy
`vllm_omni/worker/gpu_ar_model_runner.py` from the implementation commit to the
main worktree; combined is the implementation commit itself. Set `PYTHONPATH`
to the selected worktree. Keep the same validation directory and model/cache
links for all four sources. Example for one variant and round:

```bash
CUDA_VISIBLE_DEVICES=0 COSYVOICE3_TRT=0 \
COSYVOICE3_BATCH_FLOW=0 COSYVOICE3_PACKED_STREAMING=0 \
COSYVOICE3_FULL_RESPONSE=0 XDG_CACHE_HOME=/validation/cache \
PYTHONPATH=/validation/ar-output-extension-20261004/combined \
/validation/next-step-20261003/vllm030-env/bin/python \
  /workspace/benchmarks/tts/cosyvoice3_async_output/benchmark.py \
  --validation-dir /validation/ar-output-extension-20261004 \
  --label combined-replay-r1 --requests 8 --warmups 2 \
  --fixed-tokens 128 --concurrency-matrix 1 4
```

Repeat with the source and concurrency orders listed above and unique labels.
Add `--no-async-chunk` for the separate legacy check. The validation directory
contains `models/Fun-CosyVoice3-0.5B-2512`; results are saved beneath `results/`.
The evidence archive contains the driver, CPU scope extractor, aggregation
script, source patches, native timing arrays, audio hashes, resource samples,
and raw logs. Generated audio files are retained locally, outside Git.

AI assistance: Codex helped implement, test, benchmark, and document this work.
