# Qwen resident short-prefill correction — 2026-08-13

Status: root cause isolated and corrected; this is functional correctness
evidence, not a performance cohort.

## Incident

The exact Stable Affine4 artifact was served on an Apple M5 Pro with 64 GiB
of unified memory, an 8,192-token context and explicit resident mode. A
deterministic 25-token request required the complete response
`HEBRUS ALIAS PARITY OK`. The affected runtime instead returned
`<|im_start|>assistant\n` and stopped after three generated tokens.

| Field | Value |
| --- | --- |
| Artifact | `Qwen3.6-35B-A3B-Hebrus-ExpertMajor-v2-MLX-Affine4-G64.gguf` |
| Artifact bytes | 20,808,566,880 |
| Artifact SHA-256 | `dd17266185833a9f05531ce366fd7284ddca1ed64aa3dcf06e321e8c72c9ea3d` |
| Interface | persistent Hebrus Server, non-streaming chat completion |
| Residency | explicit resident |
| Context allocation | 8,192 tokens |
| Prompt tokens | 25 |

Explicit SSD and a diagnostic resident build that bypassed the layer-major
batch both returned the exact marker. That isolated the defect to the resident
short-batch executor rather than the tokenizer, server protocol, sampling or
decode path.

## First bad revision

The controlled bisection used the same artifact, host, request and explicit
resident mode:

| Revision / experiment | Result |
| --- | --- |
| `b8251e465d3db77e2db85c1b1d3aeaa1fddccf46` | **PASS** — exact marker |
| `42e2fec2a7dbb14a42e7a5612dfec00e33d443ca` | **FAIL** — control-token output; first bad revision |
| `42e2fec` with only the Affine4 gate/up+SwiGLU fusion removed | **FAIL** — marker not restored |

`42e2fec` unified the Affine4 and Q2_K_XL runtime. In the shared routed-MoE
batch host code it also narrowed the final expert-sum condition to
non-streaming MM-ID batches. Resident multi-token batches from 2 through 31
use MV-ID instead: the down projection wrote eight expert rows into scratch,
but the narrowed condition skipped their reduction into the output tensor.
The one-token case uses a separate scalar branch; batches at 32 or more tokens
use MM-ID and were not affected by this specific condition.

The same integration removed the synthetic resident Affine4 test added in
`1eb3e371`. That test covered 1, 31 and 32 rows. The published model-backed
qualification started at 128 tokens, so neither remaining lane exercised the
2–31-token MV-ID boundary.

## Correction

The final reduction now runs for every non-streaming routed batch and for the
streaming MM-ID path, so the correction applies to both supported Qwen routed
storage profiles:

```c
!use_stream_batch_selected_addr || use_stream_mm_id
```

Selected-address paths that already write their reduced result directly to
the output remain excluded. Resident and SSD admission, the layer-major
scheduler, resident decode and the model artifacts are unchanged. This record
contains a complete synthetic-path and exact model-backed reproduction for
Stable Affine4. Q2_K_XL shares the corrected host condition and retains its
primitive IQ coverage, but exact-artifact Q2 model-backed revalidation remains
part of the separate Beta gate rather than evidence claimed here.

The model-free resident Affine4 regression is restored and extended to batch
sizes 1, 2, 25, 31 and 32. The server release gate now forces resident mode,
asserts the resolved resident and 25-token layer-major log markers, then
requires the exact completion twice in each persistent alias process. Both
responses must report exactly 25 prompt tokens. The gate cannot pass
accidentally through AUTO resolving to SSD or through the >=32-token MM-ID
path.

The undocumented resident-batch bypass used only during diagnosis has been
removed. The corrected resident layer-major path is now the path exercised by
the release gate rather than an optional fallback.

## Targeted validation

On the corrected dirty probe built from the release-candidate tree, the same
resident request returned `HEBRUS ALIAS PARITY OK` twice in one process. Every
probe server was shut down cleanly after its controlled check. Exact
clean-commit evidence is produced by the release gate after this correction is
committed.

No throughput, memory reduction or model-quality claim is made from these
functional probes. Historical records remain scoped to their exact revisions,
artifacts, hardware and context frontiers.
