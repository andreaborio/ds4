# Hugging Face model rename — historical inventory

- Snapshot time: 2026-07-22, immediately before the repository moves
- Recorded outcome: all three in-place moves were reported complete and checked
  on 2026-07-22
- Namespace: `andreaborio`
- Visibility at snapshot: all three repositories were public and ungated

> [!IMPORTANT]
> This is a dated provenance record, not a live download manifest. Use
> [`qwen-release.json`](contracts/qwen-release.json),
> [`download_model.sh`](../download_model.sh), and the
> [runtime support contract](contracts/RUNTIME_SUPPORT.md) for current artifact
> identities and commands. Recheck remote visibility, redirects, counters,
> revisions, and availability before relying on them.

The snapshot was captured for the in-place repository moves summarized in the
[Hebrus rebrand record](REBRAND_ROLLOUT_PLAN.md). The comparison performed at
the time covered revisions, sibling inventories, file sizes, LFS/Xet object
identities, visibility, gating, counters, Git heads, legacy URL redirects, and
ranged reads through each old repository ID.

Qwen filenames changed later under
[`ADR 0007`](adr/0007-qwen-hebrus-artifact-filenames.md). The old Qwen basename
below therefore remains useful only as the object identity recorded at this
snapshot; it is not a current download target.

## Pre-move repository baseline

| Pre-move ID | Post-move ID | Recorded main revision | Downloads at snapshot | Likes at snapshot | Files at snapshot |
| --- | --- | --- | ---: | ---: | ---: |
| `andreaborio/DeepSeek-V4-Flash-DS4-GGUF` | `andreaborio/DeepSeek-V4-Flash-Hebrus-GGUF` | `2ba412c5d8048f8adfb24446e776015133aed9e0` | 319 | 0 | 5 |
| `andreaborio/GLM-5.2-DS4-GGUF` | `andreaborio/GLM-5.2-Hebrus-GGUF` | `0182407316ad8a3cf6cb676dad9ae298af8e091a` | 98 | 0 | 19 |
| `andreaborio/Qwen3.6-35B-A3B-DS4-GGUF` | `andreaborio/Qwen3.6-35B-A3B-Hebrus-GGUF` | `7bf9c3f7f6136aeb2599d75ee61c0cc2f18e2b02` | 343 | 0 | 7 |

The recorded post-move check observed HTTP redirects from the old web and API
IDs, ranged responses for old pinned resolve requests, and matching Git heads
through old and new URLs on 2026-07-22. Those observations are not asserted as
current external-service behavior.

## Artifact identities recorded at the snapshot

| Model | Filename recorded on 2026-07-22 | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| DeepSeek V4 Flash | `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-DS4-ExpertMajor-v2.gguf` | 86,720,114,272 | `8378080263eb9224f7228d72e2afa4ac3cf74a116023fdec2c596eff228a33e3f` |
| GLM 5.2 | `GLM-5.2-DS4-ExpertMajor-v2-Q2_K.gguf` | 262,147,193,504 | `7f5017e3076e706c78f2a5322b035a9e2f6519c65ff5b6be8b2d91aeff61505d` |
| Qwen3.6-35B-A3B | `Qwen3.6-35B-A3B-DS4-ExpertMajor-v2-MLX-Affine4-G64.gguf` | 20,808,566,880 | `dd17266185833a9f05531ce366fd7284ddca1ed64aa3dcf06e321e8c72c9ea3d` |

The current Qwen Stable and Beta basenames, revisions, statuses, and hashes are
canonical only in `qwen-release.json`. ADR 0007 records that their later Hebrus
basenames refer to byte-identical objects while the names above remain history.

The snapshot also recorded comparison and experimental objects in the Qwen and
GLM repositories. The move check covered the complete repository inventories,
not only the three rows above; it did not qualify those additional objects for
runtime use.

## Recorded verification method

The pre-move baseline was collected from the Hugging Face model API with
`blobs=true`, then compared with the post-move responses and Git references:

```sh
curl --fail --location \
  "https://huggingface.co/api/models/andreaborio/REPOSITORY?blobs=true"
```

The dated comparison checked that:

1. each post-move ID resolved the recorded revision;
2. old page, API, Git, and resolve URLs redirected at the time;
3. sibling paths, byte sizes, and LFS/Xet object identities were unchanged;
4. ranged reads and the pinned downloader completed;
5. Studio recognized both the new repository ID and the recorded legacy
   installation directory;
6. model cards retained DS4/antirez fork lineage and material references; and
7. counters remained attached to the moved repositories in the snapshot.

These checks document the completed 2026-07-22 operation. A future release or
repository move must collect fresh evidence instead of treating this inventory
as a live assertion.
