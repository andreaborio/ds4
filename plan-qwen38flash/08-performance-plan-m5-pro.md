# M5 Pro 64 GB performance and memory qualification

## 1. Product question

Can Hebrus run the text-only Qwen3.8-Flash-Next base model on an M5 Pro 64 GB in
normal AUTO mode, without swap, with genuinely sparse QSA and bounded SSD PLE,
at useful prefill/decode speed and acceptable quality? Forced modes answer
diagnostic questions; they do not answer the product question.

The public M5 Pro configuration is up to 20 GPU cores, 64 GB unified memory and
307 GB/s advertised bandwidth. Record the actual machine. Unified memory means
“CPU offload” is not a capacity escape; only released or nonresident file pages
reduce physical pressure.

## 2. Capacity hypothesis

Observed MLX-VLM external-PLE Q4 work still used roughly 71.68 GB active model
memory, so ordinary Q4 is rejected for 64 GB before Hebrus benchmarking. A plain
three-bit 125B backbone is also marginal after quant metadata, dense sensitive
weights, caches and runtime. Primary candidates therefore use mixed 1/2/3-bit
routed experts, higher precision for sensitive smaller roles, PLE around four
bits on SSD, FP32 GDN state, and initially BF16/Q8 then qualified Q4/Q8 KV.

Provisional, not release, budget target at a small context:

```text
resident model + permanent Metal allocations <= 48-50 GiB
total process/Metal working set at 4K         <= 55-56 GiB
uncommitted OS/pressure headroom              >= 8 GiB
swap used                                     = 0 before and during run
```

If normal admission exposes less working set, reduce model/cache targets; do not
force allocations to meet the table.

## 3. Candidate artifact matrix

Build role-specific candidates, not one global bit rate:

| Role | Candidate set | Reason |
|---|---|---|
| routed expert gate/up/down | high-quality reference; mixed ~1.5-3 bit; affine candidates | dominates 120.8B params/capacity/bandwidth |
| shared experts/dense GDN/GR/QSA | 3/4/8 bit or BF16 role-dependent | smaller but every-token sensitive |
| router and QSA indexer | Q8/F16 initially | selection discontinuities |
| embedding/output | proven tied/quant profiles | large vocabulary and logits quality |
| norms/bias/control constants | F16/F32/exact ints | tiny and precision-sensitive |
| GDN recurrent state | FP32 | upstream contract/drift |
| PLE | affine Q4 G32, NVFP4/IQ4-like block32 candidates | width 160 and SSD capacity |
| main KV/index cache | BF16 reference; Q8 then Q4 | long-context capacity |

For every candidate record exact bytes, group/block/tail overhead, Metal decoder
path, conversion time/peak and quality gates. M5 Tensor API Q4 may outperform a
custom Q3 despite extra bytes; benchmark actual kernels rather than assuming
bits alone determine speed.

## 4. Baselines and controls

Use the closest runnable, semantically honest controls:

- pinned Transformers/CPU or available high-memory reference for correctness;
- Hebrus simple Metal resident graph for kernel A/B on reduced artifacts;
- normal AUTO production candidate;
- forced PLE-resident control when a larger machine permits it, isolating PLE;
- forced cold/warm PLE cache on M5;
- forced expert SSD cache only to measure why it is/is not viable;
- sparse QSA versus dense reference only where dense fits;
- fused versus unfused kernel in the same artifact/precision.

Do not compare output claims across different artifacts. For changes to shared
runtime, include current qualified Qwen3.6 resident/SSD controls.

## 5. Benchmark matrix

Repository standard prompt lengths: 128, 2,048, 8,192 and 32,768, plus 65,536,
100,000 and near 262,144 for cache/QSA/context qualification. Decode at least
128 tokens after each feasible prompt. Include two deterministic prompt domains:
ordinary representative text and a routing/cache/QSA stress domain.

Run separately:

- cold process + cold PLE filesystem cache;
- new process + naturally warm OS/store cache;
- normal AUTO;
- justified controls;
- requested context caps (32K/64K first, then 128K/262K only after admission).

At endpoint, physical prefill tile may shrink; semantic context cannot.

## 6. Experimental protocol

1. Fixed power source/mode; record temperature/thermal pressure and background
   state. No concurrent giant model processes.
2. Record free disk, memory pressure and swap before each run.
3. Fresh process for each A/B/B/A sample; randomize prompt instance order where
   valid while retaining exact hash.
4. Keep artifact, context, prompt IDs, sampling and output semantics identical.
5. Warm pipeline compilation outside measured region but do not warm data for a
   cold-cache run.
6. Reset telemetry immediately before measured prefill/decode.
7. Capture exact output/evidence and compare before accepting timings.
8. Report median and dispersion; for decode report TPOT p50/p95, not only mean.
9. A performance claim needs effect beyond noise and control drift <=3%, per
   repository protocol. Investigate thermal/drift failures rather than averaging.

## 7. Metrics by bottleneck

### End to end

Time to first token, prefill token/s, decode token/s, TPOT p50/p95/p99, wall
duration, cancellation latency, process/Metal peak, resident/compressed memory,
working-set limit, pressure and swap.

### Compute/GPU

Kernel/command time by GR, GDN, QSA index score/top-k, sparse attention, router,
routed/shared MoE, PLE and dense projections; dispatch count; command-buffer
duration/error; selected specialized/fallback pipelines; GPU idle/exposed waits.

### QSA

Candidate blocks, selected blocks, tail width, selected KV bytes, index scan
bytes/time, gather bytes/time, attention time, cache type/bytes, and proof no
dense mask. The report's NVIDIA speedups are contextual evidence only and must
never appear in the Hebrus result column.

### Expert

Unique selected experts/token/tile/layer, grouped rows/expert, payload bytes
read by GPU, quant decode time, cache hit/miss/pread for forced SSD, shared-vs-
routed overlap and router host readback/sync time.

### PLE/SSD

Logical rows/bytes, unique rows/pages, cache hit rate, physical pread bytes,
read amplification, syscall/task/queue depth, p50/p95/p99 read latency, useful/
late prefetch, wait exposed at layer 1, compressed/decoded cache bytes/evictions,
and cold/warm separation.

## 8. Optimization decision tree

```text
Does normal AUTO fit without swap?
  no -> reduce routed codec/metadata, permanent buffers or context cache;
        PLE cache shrink alone cannot rescue a resident Q4 backbone.
  yes -> Is decode expert-MoE bandwidth/compute dominant?
           yes -> fuse gate/up, improve selected-expert GEMV, group/codec selector.
         Is layer-1 exposed PLE wait material?
           yes -> dedup/coalesce/cache/MTLIO or overlap; inspect amplification.
         Is QSA time growing too steeply with context?
           yes -> verify physically sparse KV and fused score/top-k; pooled index cache.
         Is GR bandwidth dominant?
           yes -> fused read/write and activation narrowing behind parity.
         Is prefill dispatch-bound?
           yes -> route grouping/GEMM, tiled GR/GDN/QSA, bounded command buffers.
```

Change one primary hypothesis per A/B. A large fused patch cannot establish which
mechanism helped or broke correctness.

## 9. Promotion thresholds

Exact numeric speed targets require baseline data on the actual M5 Pro, so they
are intentionally not invented here. Before an optimization phase, write a
one-page benchmark hypothesis with current measurements and a minimum worthwhile
effect. Global release gates are:

- normal AUTO, no forced flag required;
- correct output/evidence and quality gates;
- no swap and no sustained warning/critical pressure;
- PLE cache/staging bounded to admitted bytes;
- no cold expert-miss design dominating primary decode;
- physically sparse QSA at long context;
- throughput/latency statistically better than declared control without a
  regression at another required prompt length;
- repeated long run has stable memory/resources and no thermal invalidation.

If only 32K/64K meets gates, advertise that explicit cap first. The 262,144
config value is not a claim until endpoint qualification passes.

## 10. Other Metal devices

Build capability tiers, not chip-name conditionals:

- Tier A: M5/appropriate Metal features, optional Tensor API/MTLIO specialized;
- Tier B: modern Apple Silicon, custom SIMD-group low-bit kernels and capability-
  gated BF16/MTLIO;
- Tier C: conservative FP16+F32 accumulation and `pread` shared-staging fallback.

For every physical tier repeat correctness, memory/AUTO and at least short/2K/
8K/32K benchmarks. Lower RAM may use expert cache only if it has useful measured
speed; otherwise reject honestly. Do not call a fallback “supported” based on a
model-free kernel test.

## 11. Benchmark result template

```text
Hebrus/model/artifact commits:
machine/macOS/Metal/power/thermal:
AUTO decision and owner-byte table:
prompt domain/hash, prompt/decode/context:
cold/warm methodology:
correctness/evidence hash:
prefill tok/s, TTFT, decode tok/s, TPOT p50/p95/p99:
peak process/Metal, pressure, swap delta:
expert selected/unique/cache/pread/GPU bytes:
PLE logical/unique/cache/pread/amplification/exposed wait:
QSA candidates/selected/tail/gather/time/cache bytes:
kernel timing top contributors:
A/B/B/A samples, control drift, effect and conclusion:
raw log paths:
```

