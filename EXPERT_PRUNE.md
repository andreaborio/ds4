# Expert prune mask + full expert profile

Two opt-in hooks (fork addition #5, see [`README.md`](README.md)) for studying *which* routed
experts a domain actually needs — used by the [forgequant](https://github.com/andreaborio/forgequant)
expert-pruning experiments. Both default **off** and add zero behavioral change unless their env
vars are set.

## `DS4_EXPERT_PROFILE_FULL` — full per-expert ranking

The expert profiler (`ds4_expert_profile_write_layer` in `ds4.c`) normally emits the top-16
experts per layer. With `DS4_EXPERT_PROFILE_FULL` set, it emits the **full** per-expert ranking
(all `unique` experts), so a static prune/keep set can be derived per layer from real routing
statistics rather than a truncated head.

```sh
DS4_EXPERT_PROFILE_FULL=1 ds4 -m model.gguf --prompt-file corpus.txt ...   # profile JSON carries the full ranking
```

## `DS4_EXPERT_PRUNE_MASK` — static per-layer expert prune

Point it at a text file: a **`43 × N_EXPERT` grid of `'0'`/`'1'`** (one line per routed layer,
`'1'` = prune that expert). Non-`0/1` characters are ignored; short/long lines are tolerated
(parsed up to `DS4_MAX_EXPERT`).

```
# mask.txt — prune experts 0..9 in every layer (10 × 43 = 430 pruned)
1111111111000000…0      (line 0)
1111111111000000…0      (line 1)
…                       (43 lines)
```

**Mechanism.** The mask is applied inside the CPU router (`metal_graph_decode_cpu_router`), to
the router `probs[]` **before top-k selection**, in the non-hash (routed) branch only:

```c
ds4_expert_prune_mask_ensure();
if (g_expert_prune_mask_state == 1 && il < DS4_MAX_LAYER) {
    for (uint32_t e = 0; e < DS4_N_EXPERT; e++) {
        /* -ffast-math: large finite sentinel, not -INFINITY.
         * probs are sqrt(softplus(.)) >= 0, so this never wins top-k. */
        if (g_expert_prune_mask[il][e]) probs[e] = -1e30f;
    }
}
layer_topk_selected_experts_from_probs(selected, weights, model, layer, probs);
```

A masked expert gets a large-negative score, so it never enters top-k; the token routes to its
**next-best surviving expert**. Nothing is re-quantized — this measures, at inference time, how
much of a domain's quality lives in a few experts.

On first use it logs:

```
ds4: expert prune mask ACTIVE (430 experts pruned) from mask.txt
```

## ⚠️ Requires the CPU router

The mask lives in the **CPU router**, which is active only when
`metal_graph_decode_cpu_router_applicable()` holds — i.e. one of:

- **streaming IQ2** path: `--ssd-streaming`, non-quality mode, routed-expert types
  `IQ2_XXS / IQ2_XXS / Q2_K` (gate/up/down), `N_EXPERT_USED == 6`, `N_EXPERT >= 128`, **and**
  `DS4_METAL_ENABLE_STREAMING_IQ2_CPU_ROUTER=1`; or
- **PRO Q4** path: PRO variant, all routed-expert tensors `Q4_K`, `DS4_METAL_PRO_Q4_CPU_ROUTER=1`.

On the default GPU router path the mask is **not** consulted (no log, no effect). For the
forgequant IQ2 builds:

```sh
DS4_METAL_ENABLE_STREAMING_IQ2_CPU_ROUTER=1 DS4_EXPERT_PRUNE_MASK=mask.txt \
  ds4 -m coder-iq2.gguf -p "Write a function …" --ssd-streaming --ssd-streaming-cache-experts 40GB
```

## Code locations (`ds4.c`)

- `ds4_expert_profile_write_layer` — the `DS4_EXPERT_PROFILE_FULL` toggle (`top_n`).
- `g_expert_prune_mask[][]`, `g_expert_prune_mask_state`, `ds4_expert_prune_mask_ensure()` —
  globals + lazy file reader (just before `metal_graph_decode_cpu_router`).
- call site inside `metal_graph_decode_cpu_router`, non-hash branch, before
  `layer_topk_selected_experts_from_probs`.

## Why (the result it supports)

These hooks back the finding that a coding workload concentrates in a subset of routed experts:
dropping ~40% of routed experts (keep top-154/256) left coding pass@1 ≈ flat (74% → 72%, within
noise). Verified working after the 2026-06-18 upstream sync ([`MERGE_LOG.md`](MERGE_LOG.md)):
mask loaded (430 pruned), generation stayed coherent.

## Caveats

- Routed (non-hash) layers only; hash-routed layers are untouched.
- Static mask — it does not adapt during a run; the file is read once (lazily) and cached.
- Not upstreamable as-is (fork eval tooling); see [`FORK_NOTES.md`](FORK_NOTES.md).
