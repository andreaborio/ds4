# On-edge adaptive imatrix

A locally-run, low-bit GGUF is quantized with an **importance matrix (imatrix)**
that decides where to spend the scarce per-weight bits. The imatrix is normally
collected offline on a fixed corpus (`hebrus --imatrix-dataset … --imatrix-out
…`). This feature lets `hebrus-server` collect an imatrix from the live prompt
stream on the device, so a quantized model can be recalibrated to its actual
workload. The collector does not intentionally serialize raw prompt text,
token IDs, positions, or request IDs into the imatrix file.

The output contains derived activation statistics and static tensor names, not
a text transcript. Treat it as potentially sensitive data nonetheless: what
derived statistics reveal depends on the workload and threat model.

## Data written by the collector

The imatrix holds, per `(layer, expert)`, sums of squared activations and integer
hit counts. The on-disk `.dat` is a full rewrite of static weight-tensor names
and count-normalized mean-square vectors in the llama.cpp legacy format. This
collector path does not intentionally write raw prompt text or token metadata.
Other features have separate data policies: notably, `--trace` records prompts,
cache decisions, output, and tool calls. Keep imatrix files access-controlled,
review the applicable data policy before sharing them, and delete them when no
longer needed.

## Usage

```sh
hebrus-server -m model.gguf --imatrix-out edge.dat            # collect from live traffic
hebrus-server -m model.gguf --imatrix-out edge.dat --imatrix-every 128 --imatrix-min-requests 32
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

## Verification (measured 2026-06-08 — DeepSeek-V4-Flash IQ2, M5 64 GB)

1. **Bit-identical:** the same 8 MedXpertQA prompts run with the collector ON and OFF gave
   **8/8 identical predictions**. The collection is a pure side-channel — it does not perturb the
   forward pass. (The default OFF path is also a NULL-guarded no-op by construction: without
   `--imatrix-out`, `s->imatrix` stays NULL and every prefill site behaves exactly as before.)
2. **Valid `.dat`:** `--imatrix-every 4 --imatrix-min-requests 2`, 12 live prompts → snapshots fired
   at 4 and 8 prompts; `edge.dat` = 430 MB, **129 entries** (= 43 layers × 3, identical structure to
   an offline-collected imatrix); `deepseek4-quantize` loads it (`loaded imatrix …: 129 entries`).
3. **Observed file-content check:** grep of the measured `.dat` for selected
   prompt keywords returned zero occurrences; inspection showed binary float
   data and static tensor names. This check is not a general proof that derived
   activation statistics cannot reveal information about a workload.
4. **Slowdown (prefill only; decode unaffected).** Measured on the server path, same prompts OFF vs
   ON, fresh restarts. Short MCQ prompts (~300 tok): **+6 %** (SSD-streaming-dominated). Long prompts
   (~4 k tok, 4 distinct, mean): **+31 %** (OFF 26.0 s → ON 34.1 s) — once the prefill is
   compute-bound the layer-major path penalty dominates, so the cost **grows with prompt length**.
   Opt-in via `--imatrix-out`: the default serving path pays nothing.
5. **Re-verified 2026-07-10 on current main:** `ds4_test --server` OK with the feature compiled in
   and disabled; live smoke with `--imatrix-out --imatrix-every 2 --imatrix-min-requests 2` writes a
   valid 129-entry snapshot from real requests.
