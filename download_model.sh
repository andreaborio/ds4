#!/bin/sh
set -eu

RUNTIME_DEEPSEEK_REPO="andreaborio/DeepSeek-V4-Flash-DS4-GGUF"
RUNTIME_DEEPSEEK_FILE="DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-DS4-ExpertMajor-v2.gguf"
RUNTIME_GLM_REPO="andreaborio/GLM-5.2-DS4-GGUF"
RUNTIME_GLM_FILE="GLM-5.2-DS4-ExpertMajor-v2-Q2_K.gguf"
RUNTIME_QWEN_REPO="andreaborio/Qwen3.6-35B-A3B-DS4-GGUF"
RUNTIME_QWEN_FILE="Qwen3.6-35B-A3B-DS4-ExpertMajor-v2-Q4_K_S.gguf"

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
  qwen-v2      Qualified Qwen3.6-35B-A3B ExpertMajor v2 artifact.

Offline-only targets:
  offline-deepseek-flash-q2
  offline-deepseek-flash-q2-q4
  offline-deepseek-flash-q4
  offline-deepseek-pro-q2

The offline-only files are canonical converter inputs. They cannot be loaded by
the current runtime. Convert and verify them with gguf-tools/ds4-expert-major.py.

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

case "$TARGET" in
    deepseek-v2)
        MODEL_REPO=$RUNTIME_DEEPSEEK_REPO
        MODEL_FILE=$RUNTIME_DEEPSEEK_FILE
        ;;
    glm-v2)
        MODEL_REPO=$RUNTIME_GLM_REPO
        MODEL_FILE=$RUNTIME_GLM_FILE
        ;;
    qwen-v2)
        MODEL_REPO=$RUNTIME_QWEN_REPO
        MODEL_FILE=$RUNTIME_QWEN_FILE
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

if [ -z "$TOKEN" ] && [ -s "$HOME/.cache/huggingface/token" ]; then
    TOKEN=$(sed -n '1p' "$HOME/.cache/huggingface/token")
fi

if ! command -v hf >/dev/null 2>&1; then
    echo "Downloads require the official Hugging Face CLI." >&2
    echo "Install it with:" >&2
    echo "  python3 -m pip install -U huggingface_hub hf_xet" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"
OUT_FILE="$OUT_DIR/$MODEL_FILE"

if [ -s "$OUT_FILE" ]; then
    echo "Already downloaded: $OUT_FILE"
else
    echo "Downloading $MODEL_FILE"
    echo "from https://huggingface.co/$MODEL_REPO"
    if [ -n "$TOKEN" ]; then
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
fi

echo
if [ "$MODEL_KIND" = runtime ]; then
    echo "Runtime artifact ready. Start explicitly with:"
    echo "  ./ds4 -m $OUT_FILE -p \"Hello\""
    echo "Verify its exact byte size and SHA-256 against the family publication record."
else
    echo "Offline converter input ready; do not pass this file to ds4."
    echo "Build and verify an ExpertMajor v2 artifact with:"
    echo "  python3 gguf-tools/ds4-expert-major.py build $OUT_FILE OUTPUT-DS4-ExpertMajor-v2.gguf"
fi
echo "Done."
