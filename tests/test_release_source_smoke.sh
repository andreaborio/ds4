#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
VERSION=${RELEASE_VERSION:?set RELEASE_VERSION to the SemVer without a leading v}
REF=${RELEASE_REF:?set RELEASE_REF to the full release commit or exact tag}
TMP_BASE=$(mktemp -d "${TMPDIR:-/tmp}/hebrus-release-source-smoke.XXXXXX")
FIRST="$TMP_BASE/first"
SECOND="$TMP_BASE/second"
EXTRACT="$TMP_BASE/extract"
STEM="hebrus-$VERSION"
ARCHIVE="$STEM.tar.gz"
MANIFEST="$STEM-source.json"

cleanup() {
    rm -rf "$TMP_BASE"
}
trap cleanup 0 1 2 3 15

fail() {
    echo "release-source-smoke: FAIL: $*" >&2
    exit 1
}

cd "$ROOT"
COMMIT=$(git rev-parse --verify "$REF^{commit}")
SHORT_COMMIT=$(printf '%.12s' "$COMMIT")

python3 tools/release_source.py build \
    --version "$VERSION" --ref "$REF" --output-dir "$FIRST"
python3 tools/release_source.py build \
    --version "$VERSION" --ref "$REF" --output-dir "$SECOND"

for name in "$ARCHIVE" "$MANIFEST" SHA256SUMS; do
    cmp "$FIRST/$name" "$SECOND/$name" >/dev/null || \
        fail "repeated builds differ for $name"
done

python3 tools/release_source.py verify --manifest "$FIRST/$MANIFEST"
mkdir -p "$EXTRACT"
tar -xzf "$FIRST/$ARCHIVE" -C "$EXTRACT"
[ -d "$EXTRACT/$STEM" ] || fail "archive lacks the expected top-level directory"
[ ! -e "$EXTRACT/$STEM/.git" ] || fail "archive contains Git metadata"
[ -f "$EXTRACT/$STEM/Makefile" ] || fail "archive lacks the build entry point"
[ -f "$EXTRACT/$STEM/LICENSE" ] || fail "archive lacks the project license"
[ -f "$EXTRACT/$STEM/CITATION.cff" ] || fail "archive lacks citation metadata"

(
    cd "$EXTRACT/$STEM"
    make NATIVE_CPU_FLAG= BUILD_GIT_SHA="$SHORT_COMMIT" install-test
    make NATIVE_CPU_FLAG= BUILD_GIT_SHA="$SHORT_COMMIT" hebrus
    build_info=$(./hebrus --build-info)
    printf '%s\n' "$build_info" | grep -q "^git:     $SHORT_COMMIT$" || \
        fail "archive build does not report the release commit"
)

echo "release-source-smoke: PASS ($COMMIT, reproducible archive, staged install)"
