"""Independent transactional RNG oracle for a future DSpark caller.

The production sampler in :mod:`ds4.c` uses xorshift64* and publishes the
updated 64-bit state only when categorical sampling actually requests a
uniform.  Greedy sampling and several sampled fallback paths do not consume a
draw.  This module reproduces that state transition exactly, but it is not a
runtime dependency and does not implement target or proposal sampling.

Why a second draw schedule is necessary
---------------------------------------

One eager sequential stream cannot also be a prefix-rollback stream.  For
example, an eager depth-two round draws::

    pending, proposal[0], proposal[1], acceptance[0], ...

If the caller consumes only ``pending, proposal[0]``, the state needed after
``acceptance[0]`` necessarily includes the discarded ``proposal[1]`` draw.
Stopping before that draw omits the acceptance draw.  Re-running a shallower
round is not equivalent because it assigns a different uniform to
``acceptance[0]`` and can change the accepted token.  Xorshift state is a
linear prefix of calls, so no checkpoint of that same eager stream can satisfy
both requirements.

The oracle therefore assigns one *public ticket* to each potentially sampled
output position, including a terminal token that may not become visible.
``public_draw_consumed[i]`` records whether the scalar target sampling decision
for that position consumes a categorical uniform.  That
schedule is attached only after the verifier has produced a block; it cannot
be an input to eager proposal generation.  The ledger publishes only the first
``observed_count`` tickets at
``commit(adopted_count, observed_count, mode)``.  ``adopted_count`` is always
the block-token prefix the caller adopted into its output before choosing the
disposition.  ``RETAIN`` also preserves that prefix as reusable target/session
state and therefore allows only equal counts or one terminal token sampled but
not adopted.  ``INVALIDATE`` covers byte-level stop strings and delivery
failures: detection may inspect several block tokens past the caller-adopted
output prefix, so it permits any
``adopted_count <= observed_count <= block_count``.  The target/session state
is unusable after that disposition and must be rebuilt, but the public RNG
ledger still publishes exactly ``rng_after[observed_count]``; later block
tickets never leak into it.  Any synthetic terminal framing added after the
loop remains frontend ownership.  Proposal and acceptance uniforms are
explicit domain-separated functions of the transaction's stable starting
state and absolute public-ticket position; calculating them never advances
public RNG state.

For depth zero this construction is byte- and state-identical to the current C
sampler: a consumed public ticket is the exact xorshift64* word and high-24-bit
uniform, while a greedy or sampled fallback ticket leaves the state unchanged.
For depth greater than zero, domain separation intentionally does *not* claim
seeded-stream parity with DeepSpec, PyTorch, or any other implementation.  The
explicit 64-bit mixer gives reproducible pseudorandom substreams, not a proof
of statistical independence.  The exact acceptance and residual equations
remain owned by ``reference.speculative_sample_exact``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import math
import struct
from typing import Iterable


U64_MASK = (1 << 64) - 1
XORSHIFT_ZERO_SEED = 0x9E3779B97F4A7C15
XORSHIFT_MULTIPLIER = 0x2545F4914F6CDD1D
MAX_DSPARK_BLOCK_COUNT = 7
F32_MAX = 3.4028234663852886e38

_MIX_MULTIPLIER_1 = 0xBF58476D1CE4E5B9
_MIX_MULTIPLIER_2 = 0x94D049BB133111EB
_POSITION_MULTIPLIER = 0x9E3779B97F4A7C15
_DOMAIN_KEYS = {
    "proposal": 0x44535041524B5051,  # ASCII-derived, then mixed below.
    "acceptance": 0x44535041524B4143,
}


class TransactionRngError(ValueError):
    """Malformed transaction input or invalid transaction lifecycle."""


class StaleTransactionError(TransactionRngError):
    """A transaction cookie does not identify the active transaction."""


class CommitMode(Enum):
    """Whether the caller may retain the target/session state after commit."""

    RETAIN = "retain"
    INVALIDATE = "invalidate"


def _require_u64(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TransactionRngError(f"{name} must be an unsigned 64-bit integer")
    if value < 0 or value > U64_MASK:
        raise TransactionRngError(f"{name} is outside unsigned 64-bit range")
    return value


def xorshift64star_step(state: int) -> tuple[int, int]:
    """Return ``(next_state, output_word)`` exactly as the C sampler does."""

    x = _require_u64(state, "state")
    if x == 0:
        x = XORSHIFT_ZERO_SEED
    x ^= x >> 12
    x &= U64_MASK
    x ^= (x << 25) & U64_MASK
    x &= U64_MASK
    x ^= x >> 27
    x &= U64_MASK
    return x, (x * XORSHIFT_MULTIPLIER) & U64_MASK


def uniform_f32_from_word(word: int) -> float:
    """Reproduce ``sample_rng_f32`` from one xorshift64* output word."""

    value = _require_u64(word, "word")
    return float((value >> 40) & 0xFFFFFF) / 16777216.0


def _mix64(value: int) -> int:
    """Explicit SplitMix64 finalizer used only for auxiliary subdraws."""

    z = value & U64_MASK
    z = ((z ^ (z >> 30)) * _MIX_MULTIPLIER_1) & U64_MASK
    z = ((z ^ (z >> 27)) * _MIX_MULTIPLIER_2) & U64_MASK
    return (z ^ (z >> 31)) & U64_MASK


@dataclass(frozen=True)
class SamplingPolicy:
    """Sampler settings frozen into one transaction for audit and validation."""

    temperature: float
    top_k: int
    top_p: float
    min_p: float

    def __post_init__(self) -> None:
        numeric = (self.temperature, self.top_p, self.min_p)
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               for value in numeric):
            raise TransactionRngError("sampling floats must be real numbers")
        if not all(math.isfinite(float(value)) for value in numeric):
            raise TransactionRngError("sampling floats must be finite")
        if any(abs(float(value)) > F32_MAX for value in numeric):
            raise TransactionRngError("sampling value is outside finite F32 range")
        temperature = struct.unpack(
            "<f", struct.pack("<f", float(self.temperature))
        )[0]
        top_p = struct.unpack("<f", struct.pack("<f", float(self.top_p)))[0]
        min_p = struct.unpack("<f", struct.pack("<f", float(self.min_p)))[0]
        object.__setattr__(self, "temperature", temperature)
        object.__setattr__(self, "top_p", top_p)
        object.__setattr__(self, "min_p", min_p)
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int):
            raise TransactionRngError("top_k must be an integer")
        if self.top_k < -0x80000000 or self.top_k > 0x7FFFFFFF:
            raise TransactionRngError("top_k is outside C int range")

        # Match sample_top_p_min_p() before any vocabulary-size clamp.  The
        # runtime treats non-positive top_k as full-vocabulary sampling, caps
        # its fixed local top-k arrays at 1024, replaces invalid top_p with 1,
        # and floors min_p at zero.
        top_k = 0 if self.top_k <= 0 else min(self.top_k, 1024)
        top_p = top_p if 0.0 < top_p <= 1.0 else 1.0
        min_p = max(min_p, 0.0)
        object.__setattr__(self, "top_k", top_k)
        object.__setattr__(self, "top_p", top_p)
        object.__setattr__(self, "min_p", min_p)

    @property
    def greedy(self) -> bool:
        return float(self.temperature) <= 0.0

    def fingerprint(self) -> str:
        """Stable F32/int policy identity; it is not RNG seed material."""

        payload = struct.pack(
            "<fiff",
            float(self.temperature),
            self.top_k,
            float(self.top_p),
            float(self.min_p),
        )
        return hashlib.sha256(payload).hexdigest()


def _draw_schedule(values: Iterable[bool]) -> tuple[bool, ...]:
    schedule = tuple(values)
    if not 1 <= len(schedule) <= MAX_DSPARK_BLOCK_COUNT:
        raise TransactionRngError(
            f"draw schedule must contain 1..{MAX_DSPARK_BLOCK_COUNT} tickets"
        )
    if any(type(value) is not bool for value in schedule):
        raise TransactionRngError("draw schedule entries must be booleans")
    return schedule


@dataclass(frozen=True)
class TransactionRngLedger:
    """The publishable RNG states for every inspected block prefix.

    In RETAIN mode the final inspected position may be one terminal token that
    is absent from the caller-adopted output prefix, which is also preserved as
    reusable target/session state.  INVALIDATE may inspect a longer prefix,
    but it destroys the target/session state and requires a rebuild.
    """

    transaction_id: int
    start_state: int
    start_position: int
    block_count: int
    policy_fingerprint: str
    public_draw_consumed: tuple[bool, ...]
    rng_after: tuple[int, ...]

    def state_after_observed(self, observed_count: int) -> int:
        if isinstance(observed_count, bool) or not isinstance(observed_count, int):
            raise TransactionRngError("observed_count must be an integer")
        if observed_count < 0 or observed_count > self.block_count:
            raise TransactionRngError("observed_count is outside the published block")
        return self.rng_after[observed_count]


@dataclass(frozen=True)
class DSparkRoundUniforms:
    """Random inputs for one sampled DSpark round.

    ``None`` marks a categorical position whose prepared distribution selected
    a deterministic fallback and therefore consumed no public draw.
    """

    pending: float | None
    proposal: tuple[float, ...]
    acceptance: tuple[float, ...]
    categorical: tuple[float | None, ...]


@dataclass(frozen=True)
class RngReservation:
    """Private random-access draw reservation for one uncommitted block."""

    transaction_id: int
    start_state: int
    start_position: int
    policy: SamplingPolicy
    capacity: int

    def _check_index(self, output_index: int) -> None:
        if isinstance(output_index, bool) or not isinstance(output_index, int):
            raise TransactionRngError("output_index must be an integer")
        if output_index < 0 or output_index >= self.capacity:
            raise TransactionRngError("output_index is outside the reservation")

    def _public_trace(
        self,
        public_draw_consumed: Iterable[bool],
    ) -> tuple[tuple[bool, ...], tuple[int, ...], tuple[int | None, ...]]:
        schedule = _draw_schedule(public_draw_consumed)
        if len(schedule) > self.capacity:
            raise TransactionRngError("draw schedule exceeds the reservation")
        if self.policy.greedy and any(schedule):
            raise TransactionRngError("greedy sampling cannot consume public RNG draws")
        states = [self.start_state]
        words: list[int | None] = []
        current = self.start_state
        for consumed in schedule:
            if consumed:
                current, word = xorshift64star_step(current)
                words.append(word)
            else:
                words.append(None)
            states.append(current)
        return schedule, tuple(states), tuple(words)

    def canonical_uniforms(
        self,
        public_draw_consumed: Iterable[bool],
    ) -> tuple[float | None, ...]:
        """Return exact public uniforms for an outcome-known draw schedule."""

        _, _, words = self._public_trace(public_draw_consumed)
        return tuple(
            None if word is None else uniform_f32_from_word(word)
            for word in words
        )

    def auxiliary_uniform(self, domain: str, output_index: int) -> float:
        """Return a random-access proposal/acceptance subdraw.

        The draw/no-draw schedule, policy fingerprint, and transaction cookie
        are deliberately absent:
        retrying from the same RNG state and absolute public-ticket position is
        deterministic, and a policy change does not silently change the random
        variate as well as the probability transform.
        """

        self._check_index(output_index)
        if self.policy.greedy:
            raise TransactionRngError(
                "greedy DSpark scheduling has no auxiliary random draws"
            )
        if domain not in _DOMAIN_KEYS:
            raise TransactionRngError(f"unsupported auxiliary domain: {domain}")
        absolute = self.start_position + output_index
        material = (
            self.start_state
            ^ _DOMAIN_KEYS[domain]
            ^ ((absolute + 1) * _POSITION_MULTIPLIER)
        ) & U64_MASK
        return uniform_f32_from_word(_mix64(material))

    def dspark_round_uniforms(
        self,
        draft_depth: int,
        public_draw_consumed: Iterable[bool],
    ) -> DSparkRoundUniforms:
        """Map output-position tickets to the existing exact sampler inputs."""

        if self.policy.greedy:
            raise TransactionRngError(
                "sampled DSpark uniforms are unavailable for a greedy policy"
            )
        if isinstance(draft_depth, bool) or not isinstance(draft_depth, int):
            raise TransactionRngError("draft_depth must be an integer")
        if draft_depth < 0 or draft_depth > 5:
            raise TransactionRngError("draft_depth must be inside 0..5")
        needed = draft_depth + 2
        if self.capacity < needed:
            raise TransactionRngError(
                "reservation is too small for pending, draft rows, and final pending"
            )
        schedule = _draw_schedule(public_draw_consumed)
        if len(schedule) != needed:
            raise TransactionRngError(
                "round draw schedule must cover pending through final pending"
            )
        canonical = self.canonical_uniforms(schedule)
        return DSparkRoundUniforms(
            pending=canonical[0],
            proposal=tuple(
                self.auxiliary_uniform("proposal", index + 1)
                for index in range(draft_depth)
            ),
            acceptance=tuple(
                self.auxiliary_uniform("acceptance", index + 1)
                for index in range(draft_depth)
            ),
            categorical=tuple(
                canonical[index + 1]
                for index in range(draft_depth + 1)
            ),
        )

    def ledger(
        self,
        block_count: int,
        public_draw_consumed: Iterable[bool],
    ) -> TransactionRngLedger:
        if isinstance(block_count, bool) or not isinstance(block_count, int):
            raise TransactionRngError("block_count must be an integer")
        if block_count < 1 or block_count > self.capacity:
            raise TransactionRngError("block_count is outside the reservation")
        schedule, states, _ = self._public_trace(public_draw_consumed)
        if len(schedule) != block_count:
            raise TransactionRngError(
                "published draw schedule must match block_count"
            )
        return TransactionRngLedger(
            transaction_id=self.transaction_id,
            start_state=self.start_state,
            start_position=self.start_position,
            block_count=block_count,
            policy_fingerprint=self.policy.fingerprint(),
            public_draw_consumed=schedule,
            rng_after=states,
        )


def _reservation(
    transaction_id: int,
    start_state: int,
    start_position: int,
    policy: SamplingPolicy,
    capacity: int,
) -> RngReservation:
    state = _require_u64(start_state, "start_state")
    position = _require_u64(start_position, "start_position")
    if not isinstance(policy, SamplingPolicy):
        raise TransactionRngError("policy must be a SamplingPolicy")
    if isinstance(capacity, bool) or not isinstance(capacity, int):
        raise TransactionRngError("capacity must be an integer")
    if capacity < 1 or capacity > MAX_DSPARK_BLOCK_COUNT:
        raise TransactionRngError(
            f"capacity must be inside 1..{MAX_DSPARK_BLOCK_COUNT}"
        )
    if position > U64_MASK - capacity:
        raise TransactionRngError(
            "public-ticket position would overflow unsigned 64-bit range"
        )
    return RngReservation(
        transaction_id=transaction_id,
        start_state=state,
        start_position=position,
        policy=policy,
        capacity=capacity,
    )


class TransactionalRng:
    """Small fail-closed lifecycle oracle for begin/publish/commit.

    This class models ownership only.  A future C session API may use different
    types, but it must preserve the same state boundaries.
    """

    def __init__(self, state: int, stream_position: int = 0) -> None:
        self._state = _require_u64(state, "state")
        self._position = _require_u64(stream_position, "stream_position")
        self._last_transaction_id = 0
        self._active: RngReservation | None = None
        self._ledger: TransactionRngLedger | None = None

    @property
    def state(self) -> int:
        return self._state

    @property
    def stream_position(self) -> int:
        return self._position

    @property
    def active(self) -> bool:
        return self._active is not None

    def begin(
        self,
        policy: SamplingPolicy,
        capacity: int,
    ) -> RngReservation:
        if self._active is not None:
            raise TransactionRngError("an RNG transaction is already active")
        if self._last_transaction_id == U64_MASK:
            raise TransactionRngError("transaction cookie space is exhausted")
        transaction_id = self._last_transaction_id + 1
        reservation = _reservation(
            transaction_id,
            self._state,
            self._position,
            policy,
            capacity,
        )
        # Mutate only after every reservation input has passed validation.
        self._last_transaction_id = transaction_id
        self._active = reservation
        self._ledger = None
        return reservation

    def _require_active(self, transaction_id: int) -> RngReservation:
        cookie = _require_u64(transaction_id, "transaction_id")
        if cookie == 0:
            raise StaleTransactionError("transaction cookie is zero")
        if self._active is None or cookie != self._active.transaction_id:
            raise StaleTransactionError("transaction cookie is stale or not active")
        return self._active

    def publish(
        self,
        transaction_id: int,
        block_count: int,
        public_draw_consumed: Iterable[bool],
    ) -> TransactionRngLedger:
        reservation = self._require_active(transaction_id)
        if self._ledger is not None:
            raise TransactionRngError("the active transaction is already published")
        self._ledger = reservation.ledger(block_count, public_draw_consumed)
        return self._ledger

    def commit(
        self,
        transaction_id: int,
        adopted_count: int,
        observed_count: int,
        mode: CommitMode,
    ) -> int:
        """Publish the exact observed RNG prefix under one target disposition.

        ``adopted_count`` is the block-token prefix adopted into caller-visible
        output before the disposition is chosen.  In ``RETAIN`` mode that
        output prefix is also preserved as reusable target/session state, and
        ``observed_count`` may additionally include only one terminal token
        sampled but not evaluated or adopted.  In ``INVALIDATE`` mode, a
        multi-token byte stop or delivery failure may have inspected any longer
        prefix inside the published block; the disposition destroys the target
        state rather than retaining ``adopted_count`` as a reusable session
        prefix.  The caller must rebuild before further target work.  This
        RNG-only oracle deliberately
        preserves the public ledger across that invalidation and never
        publishes tickets after ``observed_count``.  The absolute public-ticket
        position advances by ``observed_count`` even when deterministic
        fallback tickets consumed no xorshift draw.  Synthetic post-loop
        terminal framing is outside this oracle.
        """

        self._require_active(transaction_id)
        if self._ledger is None:
            raise TransactionRngError("transaction has no published block")
        if not isinstance(mode, CommitMode):
            raise TransactionRngError("mode must be a CommitMode")
        for value, name in (
            (adopted_count, "adopted_count"),
            (observed_count, "observed_count"),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TransactionRngError(f"{name} must be an integer")
        if (adopted_count < 0
                or observed_count < adopted_count
                or observed_count > self._ledger.block_count):
            raise TransactionRngError(
                "counts must satisfy 0 <= adopted_count <= observed_count "
                "<= block_count"
            )
        if mode is CommitMode.RETAIN and observed_count > adopted_count + 1:
            raise TransactionRngError(
                "RETAIN allows at most one unadopted terminal observation"
            )
        state = self._ledger.state_after_observed(observed_count)
        if self._position > U64_MASK - observed_count:
            raise TransactionRngError("committed ticket position would overflow")
        self._state = state
        self._position += observed_count
        self._active = None
        self._ledger = None
        return state


def eager_sequential_draw_order(draft_depth: int) -> tuple[str, ...]:
    """Return the conventional eager single-stream order used in the proof."""

    if isinstance(draft_depth, bool) or not isinstance(draft_depth, int):
        raise TransactionRngError("draft_depth must be an integer")
    if draft_depth < 0 or draft_depth > 5:
        raise TransactionRngError("draft_depth must be inside 0..5")
    return (
        ("pending",)
        + tuple(f"proposal[{index}]" for index in range(draft_depth))
        + tuple(f"acceptance[{index}]" for index in range(draft_depth))
        + tuple(f"categorical[{index}]" for index in range(draft_depth + 1))
    )


def eager_prefix_obstructions(
    draft_depth: int,
    accepted_drafts_consumed: int,
) -> tuple[str, ...]:
    """List discarded eager proposal draws before a needed acceptance draw.

    The caller has consumed pending plus ``accepted_drafts_consumed`` accepted
    proposal tokens.  Every such token requires its proposal and acceptance
    decisions.  An eager depth-D stream has already drawn all D proposals before
    the first acceptance, so proposal suffixes are unavoidable obstructions.
    """

    order = eager_sequential_draw_order(draft_depth)
    if (isinstance(accepted_drafts_consumed, bool)
            or not isinstance(accepted_drafts_consumed, int)):
        raise TransactionRngError("accepted_drafts_consumed must be an integer")
    if accepted_drafts_consumed < 0 or accepted_drafts_consumed > draft_depth:
        raise TransactionRngError("accepted draft prefix is outside the draft")
    if accepted_drafts_consumed == 0:
        return ()
    last_required = order.index(
        f"acceptance[{accepted_drafts_consumed - 1}]"
    )
    required = {"pending"}
    required.update(
        f"proposal[{index}]" for index in range(accepted_drafts_consumed)
    )
    required.update(
        f"acceptance[{index}]" for index in range(accepted_drafts_consumed)
    )
    return tuple(label for label in order[:last_required + 1] if label not in required)
