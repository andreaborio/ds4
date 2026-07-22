# Hugging Face Model Rename Inventory

- Snapshot time: 2026-07-22, immediately before the repository moves
- Move status: all three in-place moves completed and verified on 2026-07-22
- Namespace: `andreaborio`
- Visibility: all three repositories are public and ungated
- Purpose: establish a reviewable baseline for the in-place DS4-to-Hebrus
  repository moves described in
  [`REBRAND_ROLLOUT_PLAN.md`](REBRAND_ROLLOUT_PLAN.md)

This snapshot is the pre-move baseline used for the completed operations. The
post-move comparison covered revisions, full sibling inventories, file sizes,
LFS SHA-256 values, visibility, gating, downloads, likes, Git heads, legacy URL
redirects, and ranged reads through each old repository ID.

## Repository baseline

| Current ID | Planned ID | Main revision | Downloads | Likes | Files |
| --- | --- | --- | ---: | ---: | ---: |
| `andreaborio/DeepSeek-V4-Flash-DS4-GGUF` | `andreaborio/DeepSeek-V4-Flash-Hebrus-GGUF` | `2ba412c5d8048f8adfb24446e776015133aed9e0` | 319 | 0 | 5 |
| `andreaborio/GLM-5.2-DS4-GGUF` | `andreaborio/GLM-5.2-Hebrus-GGUF` | `0182407316ad8a3cf6cb676dad9ae298af8e091a` | 98 | 0 | 19 |
| `andreaborio/Qwen3.6-35B-A3B-DS4-GGUF` | `andreaborio/Qwen3.6-35B-A3B-Hebrus-GGUF` | `7bf9c3f7f6136aeb2599d75ee61c0cc2f18e2b02` | 343 | 0 | 7 |

All old web and API endpoints returned HTTP `307` to the corresponding Hebrus
ID. Old pinned `resolve` requests redirected and completed as HTTP `206 Partial
Content`; old and new Git URLs returned the same recorded HEAD for every model.

## Qualified runtime artifacts

| Model | Immutable filename | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| DeepSeek V4 Flash | `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-DS4-ExpertMajor-v2.gguf` | 86,720,114,272 | `8378080263eb9224f7228d72e2afa4ac3cf74a116023fdec2c596ff228a33e3f` |
| GLM-5.2 | `GLM-5.2-DS4-ExpertMajor-v2-Q2_K.gguf` | 262,147,193,504 | `7f5017e3076e706c78f2a5322b035a9e2f6519c65ff5b6be8b2d91aeff61505d` |
| Qwen3.6-35B-A3B | `Qwen3.6-35B-A3B-DS4-ExpertMajor-v2-MLX-Affine4-G64.gguf` | 20,808,566,880 | `dd17266185833a9f05531ce366fd7284ddca1ed64aa3dcf06e321e8c72c9ea3d` |

The Qwen repository also contains historical/canonical comparison artifacts;
the GLM repository contains the archived multipart experimental tree. The move
must preserve every sibling and LFS/Xet object, not only the three qualified
files above. No GGUF filename, byte count, SHA-256, revision, manifest tensor
identifier, or historical directory is a branding rename target.

## Collection and move checks

The public baseline was collected from the official Hub model API with
`blobs=true`. Before each authenticated move, save the complete JSON response
and Git refs, then compare after the move:

```sh
curl --fail --location \
  "https://huggingface.co/api/models/andreaborio/REPOSITORY?blobs=true"
```

Post-move verification must prove:

1. the new ID resolves the exact recorded main revision;
2. the old page, API, Git, and `resolve` URLs redirect;
3. every sibling path, byte size, and LFS/Xet object identity is unchanged;
4. ranged download/resume and the pinned engine downloader still work;
5. Studio discovers the new ID and recognizes installations stored under the
   old DS4 repository directory;
6. model cards retain DS4/antirez fork lineage and all material references;
7. download and like counters remain attached to the moved repository.
