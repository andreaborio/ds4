## Benchmarking

Here we collect prefill and generation speed obtained with different hardware.

The canonical acceptance method is the
[`CONTRIBUTING.md` performance matrix](../CONTRIBUTING.md#performance-acceptance-matrix).
Run each retained frontier in a separate process. For example:

```
for frontier in 128 2048 8192 32768; do
  ./hebrus-bench \
    -m /absolute/path/to/qualified-model.gguf \
    --prompt-file speed-bench/promessi_sposi.txt \
    --ctx-start "$frontier" --ctx-max "$frontier" \
    --ctx-alloc 65536 --gen-tokens 128 \
    --csv "/tmp/hebrus-$frontier.csv" \
    --dump-decode-evidence-dir "/tmp/hebrus-$frontier-evidence"
done
```

Multi-frontier sweeps are useful for exploratory curves, but their later rows
process incremental suffixes and inherit session/cache state. They are not the
final short/medium/large/long acceptance comparison. Changes to attention, KV,
cache, RoPE, allocation, or context scaling also require the isolated 65,536-
and 100,000-token lanes defined in `CONTRIBUTING.md`.

On the qualified M5 lane, `run_m5_dsflash_arm.sh` extends an undersized prompt
deterministically for frontiers above 32,768 tokens. The generated prompt, its
source, both SHA-256 values, and whether extension occurred are recorded beside
the run. `hebrus-bench` still verifies the actual model-specific token count and
fails closed if the prompt is too short.

The runner accepts exactly 24 GiB physical hosts and hosts with at least
64 GiB. Before preparing the page-cache state, hash the complete model once:

```
DS4_M5_MODEL=/absolute/path/to/qualified-qwen.gguf \
DS4_M5_MODEL_SHA256=dd17266185833a9f05531ce366fd7284ddca1ed64aa3dcf06e321e8c72c9ea3d \
DS4_M5_MODEL_HASH_EVIDENCE=/absolute/evidence/model-hash.txt \
speed-bench/run_m5_dsflash_arm.sh --prepare-model-hash-evidence
```

The one-shot command rejects a digest mismatch and binds the evidence to the
model's resolved-path digest, device, inode, byte count, and modification time.
Each arm then checks those fields and the expected/actual hashes without
rereading the full GGUF, and copies and hashes the evidence beside its output.
This keeps verification from silently turning every cold arm into a warm
page-cache arm. The runner also rejects concurrent `hebrus`, `ds4`, and
`llama-server` inference roles while excluding its own process and current
benchmark child.

The 64 GiB profile preserves the DeepSeek defaults: a 4,096-expert preload and
a 46 GiB wired-memory ceiling. The 24 GiB profile is fail-closed at a 17 GiB
wired ceiling, zero swapout pages, at least 20 percent free memory, and at most
3,521 experts for an explicit cache. Forced SSD on that profile must set
`DS4_M5_PRELOAD_POLICY=omit`, which omits the preload option rather than passing
a zero value:

```
DS4_M5_RESIDENCY=ssd \
DS4_M5_PRELOAD_POLICY=omit \
DS4_M5_MODEL=/absolute/path/to/qualified-qwen.gguf \
DS4_M5_MODEL_SHA256=dd17266185833a9f05531ce366fd7284ddca1ed64aa3dcf06e321e8c72c9ea3d \
DS4_M5_MODEL_HASH_EVIDENCE=/absolute/evidence/model-hash.txt \
DS4_M5_PREFIX=/absolute/evidence/qwen-24g \
DS4_QWEN_TELEMETRY_JSONL=/absolute/evidence/qwen-24g.qwen-telemetry.jsonl \
DS4_M5_CACHE_STATE=warm \
speed-bench/run_m5_dsflash_arm.sh qwen-24g auto 128
```

Acceptance arms reject inherited runtime controls outside the runner namespace.
The sole optional runtime output is `DS4_QWEN_TELEMETRY_JSONL`: it must name a
fresh dedicated `$DS4_M5_PREFIX.qwen-telemetry.jsonl` path, appears in the
environment artifact, must be valid newline-terminated JSONL ending in one
`runtime_close`, must contain no invalidating runtime event, and is hashed in
the summary. An open/write/close failure or partial telemetry invalidates the
arm. Every acceptance arm admits only above the 20-percent free-memory floor;
the live free-memory abort applies to the 24 GiB profile, while the 64 GiB lane
continues to record its in-arm minimum. A wired-memory breach, swapout,
unavailable safety telemetry, or matching competitor also invalidates the arm.
Hosts between 24 and 64 GiB are not silently assigned either safety profile.

Provide PR including your numbers if your hardware was not already tested.
Call the benchmark csv file something like `m3_max.csv` or alike, so that
it is clear what hardware was used for the benchmark.

To generate an SVG graph from a CSV file:

```
python3 speed-bench/plot_speed.py speed-bench/m3_max.csv --title "M3 Max t/s"
```

The script uses only the Python standard library. By default it writes a file
next to the CSV using the `_ts.svg` suffix, such as `speed-bench/m3_max_ts.svg`.
