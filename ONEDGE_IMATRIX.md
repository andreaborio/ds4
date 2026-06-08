# On-edge adaptive imatrix (privacy-first)

A locally-run, low-bit GGUF is quantized with an **importance matrix (imatrix)** that decides
where to spend the scarce per-weight bits. The imatrix is normally collected offline on a fixed
corpus (`ds4 --imatrix-dataset … --imatrix-out …`). This feature lets `ds4-server` **collect the
imatrix from the live prompt stream on the device**, so a quantized model can be **re-calibrated to
its actual workload** — *without ever storing a single user prompt*.

This is a general model-optimization technique for any locally-run quantized model (a private
individual, a company, a clinic). It is especially relevant where the prompts are sensitive
(e.g. patient data, GDPR Art. 9): the only artifact produced is the imatrix, which is **aggregate
statistics, not text**.

## Why it's privacy-safe by construction

The imatrix holds, per `(layer, expert)`, the **sum of squared activations** (a second moment) plus
integer hit counts — structurally incapable of holding prompt content. There are **no token ids,
no text, no positions, no request ids**. The on-disk `.dat` is a fresh full rewrite of static
weight-tensor names + count-normalized mean-square vectors (the standard llama.cpp legacy format).
Prompts live transiently in RAM for the forward pass and are freed on the existing paths; **nothing
on this path writes prompt text to disk** (that is what `--trace` does — and `--trace` is unrelated
and stays off).

Think of it as a *histogram*, not a *log*: you can keep, export, or ship the imatrix to re-quantize,
and it carries no user data.

## Usage

```sh
ds4-server -m model.gguf --imatrix-out edge.dat            # collect from live traffic
ds4-server -m model.gguf --imatrix-out edge.dat --imatrix-every 128 --imatrix-min-requests 32
```

- `--imatrix-out FILE` — enable; accumulate routed-MoE importance from live prefills, snapshot to FILE.
- `--imatrix-every N` — snapshot every N completed requests (default 64; `0` = snapshot on shutdown only).
- `--imatrix-min-requests N` — never write before N completed requests (default 8; avoids a thin imatrix).

Then re-quantize with the personalized imatrix:

```sh
gguf-tools/deepseek4-quantize --hf <fp-source> --template model.gguf \
  --imatrix edge.dat --out model.personalized.gguf \
  --routed-w1 iq2_xxs --routed-w3 iq2_xxs --routed-w2 q2_k
```

## Design (how it's wired)

- The collector (`ds4_imatrix_collector`) is **owned by `ds4_session`** and driven through a small
  public API (`ds4_session_imatrix_enable/_save/_observed_tokens/_disable`) — mirroring the existing
  progress-callback pattern. The server never touches the collector internals or `weights`.
- It taps the existing hook `imatrix_collect_layer_batch(…)` (gated by `if (imatrix)`) in the
  Metal/streaming prefill (`metal_graph_prefill_layer_major`). Three prefill sites in
  `ds4_session_sync` pass `s->imatrix`; the cold-short `raw_swa` path (which has no hook) is routed
  through `chunked_range` only when the collector is enabled.
- **Default off → zero behavioral change.** When `s->imatrix == NULL`, every prefill site behaves
  byte-for-byte as before (the streaming fast path still applies).
- **Concurrency:** the server runs a single serial inference worker. Periodic snapshots run inline
  on that worker; the final snapshot runs on the main thread *after* `pthread_join(worker)`. So the
  collector is touched by exactly one thread at a time — **no locking needed**.

## Known limits

- **Coverage gaps:** short multi-turn continuations that decode-extend (rather than re-prefill) don't
  feed the imatrix; their tokens are simply not counted. Cold prefills and long suffixes do. This is
  a calibration signal, not an exact count — fine for the purpose.
- **Slowdown when enabled:** a non-NULL collector disables the streaming-decode-prefill fast path and
  forces the layer-major readback (only that path materializes the activations the imatrix reads).
  This slows **prefill** on collected requests; **decode is unaffected**. In our testing
  (M5 / 64 GB / IQ2) it was ~+31 % on long prompts and a few % on short ones — the cost grows with
  prompt length. It is opt-in via `--imatrix-out`, so the default serving path pays nothing. Profile
  with `DS4_METAL_GRAPH_PREFILL_PROFILE=1`.

## Verification

1. **Default bit-identical (structural):** without `--imatrix-out`, `s->imatrix` stays NULL, so every
   prefill site behaves exactly as before, and collecting is a pure read-side-channel that never writes
   back into the graph. Confirm by diffing greedy token output with vs. without the patch.
2. **Valid `.dat`:** run with `--imatrix-every 4 --imatrix-min-requests 2`, send ≥4 prompts, confirm
   the file appears; header `int32 entries == DS4_N_LAYER*3`, gate/up/down per layer; feed it to
   `deepseek4-quantize` as a calibration imatrix and confirm it loads.
3. **Privacy:** grep the produced `.dat` for any prompt substring — absent (binary floats + static
   tensor names only); confirm no other file is written on the prefill path.
