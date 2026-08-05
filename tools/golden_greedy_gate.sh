#!/bin/sh
# Golden greedy gate: the generated TEXT is the invariant.
#
# Every numeric path in the engine ends in tokens; comparing greedy output
# against a transcript pinned from a qualified build catches silent math
# regressions that byte-identity checks between two builds of the same HEAD
# cannot see (both sides broken looks "identical").  Born from a real one:
# commit 6889e38 transposed the HC comb strides in five production
# initializers and shipped truncated words for two days while every A-vs-A
# identity check passed.
#
# Requires the qualified DeepSeek 0731 model; set DS4_GOLDEN_MODEL or keep it
# at the default path.  Skips (exit 0) with a notice when the model is absent
# so model-less checkouts still pass the suite.
set -u
REPO=$(cd "$(dirname "$0")/.." && pwd)
MODEL=${DS4_GOLDEN_MODEL:-$HOME/models/DeepSeek-V4-Flash-0731-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-DS4-ExpertMajor-v2.gguf}
GOLDEN="$REPO/tests/golden/deepseek-0731-greedy-golden.txt"
BIN="$REPO/build/metal-arm64/bin/hebrus"
if [ ! -f "$MODEL" ]; then
    echo "golden-gate: SKIP (model not found: $MODEL)"
    exit 0
fi
if [ ! -x "$BIN" ]; then
    echo "golden-gate: FAIL (hebrus binary missing: $BIN)"
    exit 1
fi
OUT=$(mktemp /tmp/golden_gate.XXXXXX)
printf 'Scrivi una frase di venti parole sulla primavera a Milano.\nExplain in two sentences why the sky is blue.\nScrivi una funzione Python che inverta una stringa.\n' \
 | "$BIN" --model "$MODEL" --nothink --temp 0 2>/dev/null \
 | grep -vE "prefill:|generation:|t/s|^ds4|^hebrus|Ctrl\+C|/quit|Leave the prompt|^Commands:|^  /" > "$OUT"
if diff -u "$GOLDEN" "$OUT" > /tmp/golden_gate.diff 2>&1; then
    echo "golden-gate: OK (output identical to the qualified reference)"
    rm -f "$OUT" /tmp/golden_gate.diff
    exit 0
fi
echo "golden-gate: FAIL — generated text diverges from the qualified reference:"
sed -n '1,40p' /tmp/golden_gate.diff
rm -f "$OUT"
exit 1
