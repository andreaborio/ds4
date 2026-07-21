## Benchmarking

Here we collect prefill and generation speed obtained with different hardware.

The canonical acceptance method is the
[`CONTRIBUTING.md` performance matrix](../CONTRIBUTING.md#performance-acceptance-matrix).
Run each retained frontier in a separate process. For example:

```
for frontier in 128 2048 8192 32768; do
  ./ds4-bench \
    -m /absolute/path/to/QUALIFIED-DS4-ExpertMajor-v2.gguf \
    --prompt-file speed-bench/promessi_sposi.txt \
    --ctx-start "$frontier" --ctx-max "$frontier" \
    --ctx-alloc 65536 --gen-tokens 128 \
    --csv "/tmp/ds4-$frontier.csv" \
    --dump-decode-evidence-dir "/tmp/ds4-$frontier-evidence"
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
the run. `ds4-bench` still verifies the actual model-specific token count and
fails closed if the prompt is too short.

Provide PR including your numbers if your hardware was not already tested.
Call the benchmark csv file something like `m3_max.csv` or alike, so that
it is clear what hardware was used for the benchmark.

To generate an SVG graph from a CSV file:

```
python3 speed-bench/plot_speed.py speed-bench/m3_max.csv --title "M3 Max t/s"
```

The script uses only the Python standard library. By default it writes a file
next to the CSV using the `_ts.svg` suffix, such as `speed-bench/m3_max_ts.svg`.
