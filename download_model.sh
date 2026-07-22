#!/bin/sh
set -eu

RUNTIME_DEEPSEEK_REPO="andreaborio/DeepSeek-V4-Flash-Hebrus-GGUF"
RUNTIME_DEEPSEEK_FILE="DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-DS4-ExpertMajor-v2.gguf"
RUNTIME_DEEPSEEK_BYTES=86720114272
RUNTIME_DEEPSEEK_SHA256="8378080263eb9224f7228d72e2afa4ac3cf74a116023fdec2c596ff228a33e3f"
RUNTIME_GLM_REPO="andreaborio/GLM-5.2-Hebrus-GGUF"
RUNTIME_GLM_FILE="GLM-5.2-DS4-ExpertMajor-v2-Q2_K.gguf"
RUNTIME_GLM_BYTES=262147193504
RUNTIME_GLM_SHA256="7f5017e3076e706c78f2a5322b035a9e2f6519c65ff5b6be8b2d91aeff61505d"
RUNTIME_QWEN_STATUS="published"
RUNTIME_QWEN_REPO="andreaborio/Qwen3.6-35B-A3B-Hebrus-GGUF"
RUNTIME_QWEN_FILE="Qwen3.6-35B-A3B-DS4-ExpertMajor-v2-MLX-Affine4-G64.gguf"
RUNTIME_QWEN_BYTES=20808566880
RUNTIME_QWEN_SHA256="dd17266185833a9f05531ce366fd7284ddca1ed64aa3dcf06e321e8c72c9ea3d"
RUNTIME_QWEN_REVISION="7bf9c3f7f6136aeb2599d75ee61c0cc2f18e2b02"
RUNTIME_QWEN_MIN_RUNTIME_COMMIT="73a332fef82a0bcdd567d17e0de17aa004cad85d"
RUNTIME_REVISION="ds4-v0.2.0"

OFFLINE_DEEPSEEK_REPO="antirez/deepseek-v4-gguf"
OFFLINE_FLASH_Q2_FILE="DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
OFFLINE_FLASH_Q2_Q4_FILE="DeepSeek-V4-Flash-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-fixed.gguf"
OFFLINE_FLASH_Q4_FILE="DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf"
OFFLINE_PRO_Q2_FILE="DeepSeek-V4-Pro-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-Instruct-imatrix.gguf"

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
OUT_DIR=${DS4_GGUF_DIR:-"$ROOT/gguf"}
case "$OUT_DIR" in
    /*) ;;
    *) OUT_DIR="$ROOT/$OUT_DIR" ;;
esac
TOKEN=${HF_TOKEN:-}

usage() {
    cat <<'EOF'
DS4 model downloader

Usage:
  ./download_model.sh deepseek-v2 [--token TOKEN]
  ./download_model.sh glm-v2 [--token TOKEN]
  ./download_model.sh qwen-v2 [--token TOKEN]
  ./download_model.sh offline-deepseek-flash-q2 [--token TOKEN]
  ./download_model.sh offline-deepseek-flash-q2-q4 [--token TOKEN]
  ./download_model.sh offline-deepseek-flash-q4 [--token TOKEN]
  ./download_model.sh offline-deepseek-pro-q2 [--token TOKEN]

Runtime targets:
  deepseek-v2  Qualified DeepSeek V4 Flash ExpertMajor v2 artifact.
  glm-v2       Qualified GLM 5.2 ExpertMajor v2 artifact.
  qwen-v2      Qualified Qwen3.6-35B-A3B affine4/group-64 artifact.

Offline-only targets:
  offline-deepseek-flash-q2
  offline-deepseek-flash-q2-q4
  offline-deepseek-flash-q4
  offline-deepseek-pro-q2

The offline-only files are canonical converter inputs. They cannot be loaded by
the current runtime, and this downloader does not qualify their byte identity.
Convert and verify them with gguf-tools/ds4-expert-major.py.

Options:
  --token TOKEN  Hugging Face token. Otherwise HF_TOKEN or the local token
                 cache is used if present.

Environment:
  DS4_GGUF_DIR   Download directory. Default: ./gguf

The script never creates or changes ./ds4flash.gguf. Pass runtime artifacts
explicitly with -m. Re-running the same target resumes through the official
Hugging Face downloader.
EOF
}

if [ $# -eq 0 ]; then
    usage
    exit 1
fi

TARGET=$1
shift
MODEL_KIND=runtime
MODEL_REVISION=
MODEL_BYTES=
MODEL_SHA256=

case "$TARGET" in
    deepseek-v2)
        MODEL_REPO=$RUNTIME_DEEPSEEK_REPO
        MODEL_FILE=$RUNTIME_DEEPSEEK_FILE
        MODEL_REVISION=$RUNTIME_REVISION
        MODEL_BYTES=$RUNTIME_DEEPSEEK_BYTES
        MODEL_SHA256=$RUNTIME_DEEPSEEK_SHA256
        ;;
    glm-v2)
        MODEL_REPO=$RUNTIME_GLM_REPO
        MODEL_FILE=$RUNTIME_GLM_FILE
        MODEL_REVISION=$RUNTIME_REVISION
        MODEL_BYTES=$RUNTIME_GLM_BYTES
        MODEL_SHA256=$RUNTIME_GLM_SHA256
        ;;
    qwen-v2)
        MODEL_REPO=$RUNTIME_QWEN_REPO
        MODEL_FILE=$RUNTIME_QWEN_FILE
        MODEL_REVISION=$RUNTIME_QWEN_REVISION
        MODEL_BYTES=$RUNTIME_QWEN_BYTES
        MODEL_SHA256=$RUNTIME_QWEN_SHA256
        ;;
    offline-deepseek-flash-q2)
        MODEL_REPO=$OFFLINE_DEEPSEEK_REPO
        MODEL_FILE=$OFFLINE_FLASH_Q2_FILE
        MODEL_KIND=offline
        ;;
    offline-deepseek-flash-q2-q4)
        MODEL_REPO=$OFFLINE_DEEPSEEK_REPO
        MODEL_FILE=$OFFLINE_FLASH_Q2_Q4_FILE
        MODEL_KIND=offline
        ;;
    offline-deepseek-flash-q4)
        MODEL_REPO=$OFFLINE_DEEPSEEK_REPO
        MODEL_FILE=$OFFLINE_FLASH_Q4_FILE
        MODEL_KIND=offline
        ;;
    offline-deepseek-pro-q2)
        MODEL_REPO=$OFFLINE_DEEPSEEK_REPO
        MODEL_FILE=$OFFLINE_PRO_Q2_FILE
        MODEL_KIND=offline
        ;;
    -h|--help|help)
        usage
        exit 0
        ;;
    *)
        echo "Unknown model target: $TARGET" >&2
        echo >&2
        usage >&2
        exit 1
        ;;
esac

while [ $# -gt 0 ]; do
    case "$1" in
        --token)
            shift
            if [ $# -eq 0 ]; then
                echo "Missing value after --token" >&2
                exit 1
            fi
            TOKEN=$1
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
    shift
done

file_size_bytes() {
    if size=$(stat -f %z "$1" 2>/dev/null); then
        printf '%s\n' "$size"
    elif size=$(stat -c %s "$1" 2>/dev/null); then
        printf '%s\n' "$size"
    else
        LC_ALL=C wc -c < "$1" | tr -d '[:space:]'
    fi
}

file_sha256() {
    if command -v sha256sum >/dev/null 2>&1; then
        LC_ALL=C sha256sum "$1" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        LC_ALL=C shasum -a 256 "$1" | awk '{print $1}'
    else
        echo "Cannot verify runtime artifact: sha256sum or shasum is required." >&2
        return 1
    fi
}

verify_runtime_artifact() {
    artifact=$1

    if [ ! -f "$artifact" ]; then
        echo "Runtime artifact is not a regular file: $artifact" >&2
        return 1
    fi

    actual_bytes=$(file_size_bytes "$artifact")
    if [ "$actual_bytes" != "$MODEL_BYTES" ]; then
        echo "Runtime artifact byte size mismatch: $artifact" >&2
        echo "  expected: $MODEL_BYTES" >&2
        echo "  actual:   $actual_bytes" >&2
        return 1
    fi

    actual_sha256=$(file_sha256 "$artifact") || return 1
    if [ "$actual_sha256" != "$MODEL_SHA256" ]; then
        echo "Runtime artifact SHA-256 mismatch: $artifact" >&2
        echo "  expected: $MODEL_SHA256" >&2
        echo "  actual:   $actual_sha256" >&2
        return 1
    fi
}

if [ -z "$TOKEN" ] && [ -s "$HOME/.cache/huggingface/token" ]; then
    TOKEN=$(sed -n '1p' "$HOME/.cache/huggingface/token")
fi

mkdir -p "$OUT_DIR"
OUT_FILE="$OUT_DIR/$MODEL_FILE"

if [ -e "$OUT_FILE" ] || [ -L "$OUT_FILE" ]; then
    if [ "$MODEL_KIND" = runtime ]; then
        verify_runtime_artifact "$OUT_FILE"
    elif [ ! -f "$OUT_FILE" ] || [ ! -s "$OUT_FILE" ]; then
        echo "Offline converter input is empty or not a regular file: $OUT_FILE" >&2
        exit 1
    fi
    echo "Already downloaded: $OUT_FILE"
else
    if ! command -v hf >/dev/null 2>&1; then
        echo "Downloads require the official Hugging Face CLI." >&2
        echo "Install it with:" >&2
        echo "  python3 -m pip install -U huggingface_hub hf_xet" >&2
        exit 1
    fi

    echo "Downloading $MODEL_FILE"
    if [ "$MODEL_KIND" = runtime ]; then
        echo "from https://huggingface.co/$MODEL_REPO/tree/$MODEL_REVISION"
        if [ -n "$TOKEN" ]; then
            hf download "$MODEL_REPO" "$MODEL_FILE" \
                --repo-type model --local-dir "$OUT_DIR" \
                --revision "$MODEL_REVISION" --token "$TOKEN"
        else
            hf download "$MODEL_REPO" "$MODEL_FILE" \
                --repo-type model --local-dir "$OUT_DIR" \
                --revision "$MODEL_REVISION"
        fi
    elif [ -n "$TOKEN" ]; then
        hf download "$MODEL_REPO" "$MODEL_FILE" \
            --repo-type model --local-dir "$OUT_DIR" --token "$TOKEN"
    else
        hf download "$MODEL_REPO" "$MODEL_FILE" \
            --repo-type model --local-dir "$OUT_DIR"
    fi

    if [ ! -s "$OUT_FILE" ]; then
        echo "Download finished but expected file is missing: $OUT_FILE" >&2
        exit 1
    fi

    if [ "$MODEL_KIND" = runtime ]; then
        verify_runtime_artifact "$OUT_FILE"
    fi
fi

echo
if [ "$MODEL_KIND" = runtime ]; then
    echo "Verified revision: $MODEL_REVISION"
    echo "Verified bytes:    $MODEL_BYTES"
    echo "Verified SHA-256:  $MODEL_SHA256"
    echo "Runtime artifact ready. Start explicitly with:"
    printf '  ./ds4 -m "%s" --ctx 8192 -p "Hello"\n' "$OUT_FILE"
else
    echo "Offline converter input present; byte identity remains unverified."
    echo "Do not pass this file to ds4."
    echo "Build and verify an ExpertMajor v2 artifact with:"
    echo "  python3 gguf-tools/ds4-expert-major.py build $OUT_FILE OUTPUT-DS4-ExpertMajor-v2.gguf"
fi
echo "Done."
