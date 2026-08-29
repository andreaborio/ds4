# Model contract, forward equations and state

## 1. Closed identity and dimensions

All values below are admission-time equalities for the first supported artifact.

| Field | Required value |
|---|---:|
| architecture / model type | `Qwen4ExpForConditionalGeneration` / `qwen4_exp` |
| text layers | 48 |
| hidden width `d` | 2,560 |
| residual streams `B` / wide width | 4 / 10,240 |
| vocabulary | 248,320 |
| maximum positions | 262,144 |
| layer pattern | GDN, GDN, GDN, QSA; repeat 12 times |
| QSA Q heads / KV heads / head dim | 24 / 2 / 256 |
| rotary dimensions / theta | 64 / 10,000,000 |
| multimodal RoPE sections | `[11, 11, 10]`, interleaved |
| GDN key heads / value heads / head dim | 16 / 48 / 128 |
| GDN convolution kernel | 4 |
| routed experts / active experts | 512 / 10 |
| routed and shared expert width | 640 |
| router | float32 softmax over 512, normalized top-10 |
| GR low rank | 320 |
| QSA index query heads / key heads / dim | 4 / 1 / 128 |
| QSA token budget / compression ratio | 2,048 / 4 |
| PLE insertion | layer ID 2 upstream, zero-based decoder index 1 |
| PLE n-gram / heads | bigram+trigram / 8 each, 16 total |
| PLE base vocabulary / rows / row width | 20,000,000 / 320,001,536 / 160 |
| PLE convolution | kernel 4, dilation 3 |
| MTP layers in source checkpoint | 1 (not executed in phase 1) |

The first artifact requires the exact 48-entry layer-type array, not merely a
`full_attention_interval=4` hint. This catches shifted or future variants.

## 2. Zero-centered RMS normalization

The model stores a zero-centered scale. For a vector `x` of width `n`:

```text
r = rsqrt(mean(float32(x_i)^2) + eps)
y_i = cast(x_i * r) * (1 + weight_i)
```

Mean, epsilon addition and reciprocal square root are float32. Never reuse a
kernel that multiplies by `weight` directly. Add tests with an all-zero stored
weight: output must be ordinary RMS-normalized input, not zero.

## 3. Four-stream Gated Residual (GR)

Let the persistent residual be `X = [x0,x1,x2,x3]`, each branch width `d`.
For both the token-mixer read and the MoE read:

```text
n_b = ZCRMSNorm_b(x_b)
N   = concat(n_0, n_1, n_2, n_3)                 # width 4d
h   = SiLU((N W_down) / 4)                       # width 320
g   = sigmoid(h W_up)                            # width 4d
read(X) = mean_b(g_b * n_b)                      # width d
```

The same normalized wide input also produces four injection scalars before the
sub-block runs:

```text
w = 2 * sigmoid(W_inject(N) / 4)
```

After the sub-block produces `u` of width `d`, update every stream:

```text
x_b <- x_b + w_b * u
```

The final mixer repeats the read but does not combine with another block output;
its width-`d` result goes to the output head. Preserve `/4` before SiLU, `/4`
before the injection sigmoid, and the `(0,2)` injection range. `W_inject`
consumes normalized `N`; it never normalizes or inspects `u`. Do not collapse four streams
into their mean between layers. A fused kernel may avoid materializing `N`, but
its oracle is the equation above.

## 4. Gated DeltaNet (GDN) layers

For each token, `in_proj_qkv` and control projections produce Q, K, V, `a`,
`beta`, and output gate `z`. Apply depthwise causal convolution of kernel 4 and
SiLU to Q/K/V before the recurrence. The persistent recurrent matrix has 48
value heads, each `128 x 128`; 16 Q/K heads are repeated three times.

In float32 control arithmetic:

```text
alpha = -exp(A_log) * softplus(a + dt_bias)      # log decay, <= 0
decay = exp(alpha)
beta  = sigmoid(beta_logit)
qhat  = l2_normalize(q) / sqrt(128)
khat  = l2_normalize(k)
pred  = khat^T @ (decay * S)
delta = beta * (v - pred)
S     = decay * S + outer(khat, delta)             # S is [key,value]
y     = qhat^T @ S
out   = W_o(RMSNorm_weighted(y) * sigmoid(z))
```

Match the pinned reference's update order exactly; the compact equations do not
license reassociation. Decode is one recurrent step. Prefill may be chunked or
scanned, but chunk boundaries must be bitwise/state equivalent within the
declared tolerance. `RMSNorm_weighted` is the exception to zero-centered norms:
its stored GDN weight is centered on one and multiplied directly. The converter
must not add one to it. Persistent state per sequence:

- `48 * 128 * 128` float32 recurrence values per GDN layer;
- three prior Q/K/V convolution samples per channel;
- position and validity/generation metadata.

There are 36 GDN layers. The recurrence alone is about `108 MiB` derived as
`36*48*128*128*4`; convolution history adds roughly `4 MiB`. Never serialize
this state as BF16 solely to save memory without a quality and long-run drift
decision.

## 5. Qwen Sparse Attention (QSA)

### 5.1 Main attention

The Q projection contains an interleaved query and per-head gate. Split it by
head, apply per-head Q/K zero-centered RMSNorm, partial RoPE to the first 64 of
256 dimensions, causal sparse attention with 24 Q and 2 KV heads, then:

```text
head_out <- head_out * sigmoid(query_gate)
out      <- W_o(concat(head_out))
```

The main KV cache stores two heads of 256 values for K and V. At 262,144 tokens,
12 QSA layers in BF16 require exactly `12*262144*2*256*2*2 = 6 GiB` before
alignment and metadata.

### 5.2 Index keys and complete groups

The indexer computes four query heads and one raw key of width 128. Cache the
raw key before pooling, normalization or RoPE. For token position `p`, only
complete groups of four historical tokens are candidates. For group `j`:

```text
raw_group_key_j = mean(raw_key[4j : 4j+4])
key_j = partial_rope(ZCRMSNorm(raw_group_key_j), position=4j)
score_j = sum(h=0..3, ReLU(dot(query_h, key_j))) / sqrt(128)
```

Select at most 512 complete groups (2,048 tokens), expand each selected group to
its four token indices, and append the current incomplete causal tail. Thus the
actual attention width is at most `2048 + 3`, and less near sequence start.
Selection ties need one deterministic policy shared by Metal and the oracle.
Specify it as descending score, then ascending group index unless the pinned
reference fixture proves another stable ordering.

The separate raw index-key cache is about `12*262144*128*2 = 0.75 GiB` in BF16.
Its slot/copy/remove semantics must mirror the main KV cache exactly. Never pool
four adjacent physical cache cells: group by logical sequence and absolute
position, since cache storage can wrap or be compacted.

After the raw-cache reference path is qualified, production may compact each
completed logical group into one already pooled+normalized+RoPE key and retain
only the raw tail of at most three keys. This is exact because a completed group
is immutable. It reduces full-context BF16 index storage from about 0.75 GiB to
about 0.19 GiB. The optimized cache still mirrors KV logical positions and must
support rewind through group boundaries; enable it only after raw and pooled
paths select identical blocks.

### 5.3 Sparse means sparse compute

Creating a dense `T x T` mask and calling a dense attention kernel is a
correctness scaffold, not an acceptable production implementation. Production
prefill gathers selected K/V rows into bounded tiles or performs indirect block
loads; decode reads only selected groups plus tail. Instrument selected blocks,
expanded width, gather bytes and kernel time so this property is observable.

## 6. MoE

For every layer and token:

```text
logits  = float32(W_router x)
probs   = softmax(logits over all 512 experts)       # float32
(id,w)  = deterministic_top10(probs)
w       = w / sum(w)
expert_e(x) = W_down_e(SiLU(W_gate_e x) * (W_up_e x))
routed  = sum_j w_j * expert_id_j(x)
shared  = W_down_s(SiLU(W_gate_s x) * (W_up_s x))
output  = routed + sigmoid(W_shared_gate x) * shared
```

Do not select top-10 on raw logits using an approximate order unless an oracle
proves identical IDs; calculate softmax in float32. Keep IDs and weights paired
through grouping, deduplication and SSD planning. The existing top-8 Qwen3.6 API
is not generalized by passing 10 into an eight-wide buffer.

Derived routed geometry:

```text
one expert = 3 * 2560 * 640 = 4,915,200 parameters
all routed = 48 * 512 * 4,915,200 = 120,795,955,200 parameters
active/token = 48 * 10 * 4,915,200 = 2,359,296,000 parameter uses
```

At ideal payload only, all routed experts are 56.25 GiB at 4 bits or 28.125 GiB
at 2 bits. Quant metadata, dense weights and runtime state are additional.

## 7. PLE n-gram embedding

### 7.1 Exact address generation

PLE uses 16 logical heads: eight bigram and eight trigram. Each has its own
successive-prime modulus at or above 20,000,000 and a cumulative row offset.
Token-position multipliers are generated with the reference SplitMix64 constants
and forced odd. Perform all hash operations in unsigned 64-bit wraparound:

```text
h2 = current * m0 XOR previous1 * m1
h3 = h2 XOR previous2 * m2
row = offset[head] + (hash % prime[head])
```

Gather one width-160 row for each of 16 heads and concatenate to width 2,560.
EOS `248044` resets/truncates the predecessor chain for the following token; the
current EOS still hashes with its available predecessors. A missing predecessor
is filled with this EOS sentinel. Store the last
two token IDs per live sequence and include them in reset/copy/rollback.

The table geometry is `320,001,536 * 160 = 51,200,245,760` parameters, about
102.4 GB in BF16 or 25.6 GB at ideal 4 bits. Source weights are physically split
but the runtime row space is logical and contiguous.

### 7.2 PLE injection

PLE executes once, before zero-based decoder layer 1:

```text
e = concat(gathered rows)                           # 2560
k = group_ZCRMSNorm(W_key e)                       # 4*2560
v = W_value e                                      # 2560
q = group_ZCRMSNorm(current wide residual)          # 4*2560
s = dot_group(q, k) / sqrt(2560)
s = sign(s) * sqrt(max(abs(s), 1e-6))
g = sigmoid(s)                                     # four scalar gates
p = concat(g_b * v for b in 0..3)
c = SiLU(depthwise_dilated_conv(group_norm(p), kernel=4, dilation=3))
wide_residual <- wide_residual + p + c
```

The exact reference ordering of normalization, convolution and addition governs
fixtures if the compact notation is ambiguous. PLE convolution state covers the
effective nine prior positions required by `dilation*(kernel-1)`, not merely
three contiguous tokens. This is 9x10,240 state values per live sequence.

### 7.3 Prefetch opportunity

Addresses depend only on token IDs and two-token history, so all rows for an
input batch are known before layer 0. Deduplicate and begin SSD reads while
embedding/layer 0 execute. Decode can calculate the next current-token addresses
only after the sampled token is known, but can overlap the PLE read with the
first layer of that token. This is a graph scheduling contract, not optional
readahead.

## 8. Tokenizer and chat template

Required token IDs include BOS/end-of-text/pad `248044`, EOS set
`{248046,248044}`, image `248056`, and video `248057`; validate the full special
token mapping from the pinned tokenizer, not only this subset. The tokenizer is
Qwen2-style with a new pre-tokenization regex and `model_max_length=262144`.

Tests must freeze:

- Unicode normalization edge cases, combining marks and astral code points;
- whitespace/newline runs and punctuation around control tokens;
- multilingual, code and invalid-byte/replacement-character cases;
- empty/system/user/assistant turns and tool-like text;
- default reasoning effort (upstream template defaults to `xhigh`), explicit
  disabling, and preservation/removal of prior reasoning exactly as configured;
- literal strings resembling special tokens, which cannot acquire control
  semantics through ordinary user text.

Template behavior is part of artifact identity. Do not patch it at runtime based
on a hosted-service API or a future tokenizer revision.

## 9. MTP boundary

The source checkpoint contains one MTP layer. Phase 1 validates its tensor
presence/absence according to the text-artifact manifest but does not execute it.
Base-model logits are the release oracle. A later phase may implement speculative
decoding and reuse base QSA top-block IDs, but must prove identical committed
tokens, transactional state rollback, and positive end-to-end speed after draft
overhead. MTP is never required for base correctness.
