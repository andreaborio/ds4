# Qwen DS4-native expert-major publication gate

Date: 2026-07-17
Host: Apple M5 Pro, 64 GiB unified memory, AC power
Branch: `codex/qwen-expert-major-store`

This gate compares the release GGUF with the already validated canonical GGUF
plus expert-major sidecar. Both arms use the same payload, resident Metal
kernels, greedy decode, prompt, context allocation, and complete optimization
stack. The only intended difference is where the expert-major extent lives.

## Artifact identity

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| Canonical GGUF | 20,808,563,424 | `c33efb67bde86c9ba1f9e79c2dc42627170963bef0e915ab9b91a55cfb6d0fcd` |
| Expert-major sidecar payload | 18,119,393,280 | `d8bbe3731e4ac4f0117b24f8e8cb0ebaaf1a84cbfa7f264e4b297290946ee49f` |
| Native expert-major GGUF | 20,808,970,240 | `fb2b344d49f0c3dfd854cfc11d92ffc873cc93a1d30bf4664e5aea6f1bfef839` |

The native file is 406,816 bytes larger than the canonical GGUF. It does not
contain a second copy of the routed weights.

## Commands

The sweep used `speed-bench/promessi_sposi.txt`, 16 greedy decode tokens at
every frontier, and an A/B/A order: native, canonical plus sidecar, native.

```sh
COMMON="--metal --resident --power 100 -t 8 \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 --ctx-max 16384 --step-mul 2 --gen-tokens 16"

DS4_QWEN_EXPERIMENTAL_METAL=1 ./ds4-bench \
  -m Qwen3.6-35B-A3B-DS4-ExpertMajor-v1-Q4_K_S.gguf $COMMON \
  --csv native.csv --dump-decode-evidence-dir native-evidence

DS4_QWEN_EXPERIMENTAL_METAL=1 \
DS4_QWEN_EXPERT_PACK_PATH=Qwen3.6-35B-A3B-ds4-Q4_K_S.experts.pack \
DS4_QWEN_EXPERT_PACK_SHA256=d8bbe3731e4ac4f0117b24f8e8cb0ebaaf1a84cbfa7f264e4b297290946ee49f \
DS4_QWEN_EXPERT_PACK_VERSION=1 \
DS4_QWEN_GGUF_SHA256=c33efb67bde86c9ba1f9e79c2dc42627170963bef0e915ab9b91a55cfb6d0fcd \
./ds4-bench -m Qwen3.6-35B-A3B-ds4-Q4_K_S.gguf $COMMON \
  --csv sidecar.csv --dump-decode-evidence-dir sidecar-evidence
```

The second native command is identical to the first and writes separate CSV
and evidence directories.

## Correctness

All A/B/A decode-evidence files are byte-identical at every frontier. Each
file contains 16 sampled token IDs, the final argmax ID, and the complete final
logit vector.

| Frontier | Evidence SHA-256 |
|---:|---|
| 2,048 | `8bb72bcf6a76a3ae921595d3febf2149698df26350ba305a3f61ca0cf7e1ac97` |
| 4,096 | `06399c962558d1fb7ba81c79c8125cab31011becec1dd37c428d83c602e7c686` |
| 8,192 | `1a6570acea2ce8b9dca9276325d3480b08b41eea165899bf56265950789a9f7e` |
| 16,384 | `282bc6614e9f0f1c598f29f83ffa63702fd9a1a814aaa595da5d69159d1734f9` |

## Performance

Each cell is `prefill t/s / decode t/s`.

| Context | Native A1 | Sidecar B | Native A2 |
|---:|---:|---:|---:|
| 2,048 | 297.00 / 28.92 | 301.12 / 29.40 | 282.38 / 28.75 |
| 4,096 | 202.73 / 18.96 | 193.21 / 18.94 | 195.29 / 18.80 |
| 8,192 | 134.88 / 11.10 | 128.52 / 11.19 | 130.03 / 10.98 |
| 16,384 | 65.33 / 6.00 | 79.34 / 6.05 | 76.06 / 5.98 |

The first native 16K leg was an outlier and was not used alone to accept or
reject the format. The repeated native leg recovered to within 4.1% of the
sidecar prefill rate; its decode rate was within 1.2%. At 8K the repeated
native prefill leg was 1.2% faster. There was no new swapout during the final
native leg, no throttled pages, and 76% system-wide memory free after teardown.

This is a publication parity gate, not a claim that the container layout makes
the Metal math faster. The exact-match evidence proves the same computation;
the A/B/A spread puts the remaining timing difference inside ordinary shared
Mac thermal and VM variability. A future performance claim requires a longer
alternating campaign with controlled cool-downs.
