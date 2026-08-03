#!/usr/bin/env python3
"""Model-free tests for the independent DSpark transactional RNG oracle."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.dspark_oracle.transaction_rng import (  # noqa: E402
    MAX_DSPARK_BLOCK_COUNT,
    SamplingPolicy,
    StaleTransactionError,
    TransactionalRng,
    TransactionRngError,
    eager_prefix_obstructions,
    eager_sequential_draw_order,
    uniform_f32_from_word,
    xorshift64star_step,
)


SAMPLED = SamplingPolicy(temperature=0.7, top_k=0, top_p=0.95, min_p=0.0)
GREEDY = SamplingPolicy(temperature=0.0, top_k=0, top_p=1.0, min_p=0.0)
SEED = 0x0123456789ABCDEF


class XorshiftCompatibilityTests(unittest.TestCase):
    def test_closed_vectors_match_current_c_sampler(self) -> None:
        state = 0
        expected = (
            (0x03F721DFFE39B342, 0x0D83B3E29A21487A, 0.052790820598602295),
            (0x5830920757D41153, 0x54C44C79F1FE9D67, 0.33112025260925293),
            (0x44DA53DEC8EB16D8, 0xA845F342007A0E78, 0.6573173403739929),
        )
        for expected_state, expected_word, expected_uniform in expected:
            state, word = xorshift64star_step(state)
            self.assertEqual(state, expected_state)
            self.assertEqual(word, expected_word)
            self.assertEqual(uniform_f32_from_word(word), expected_uniform)

        state, word = xorshift64star_step(1)
        self.assertEqual(state, 0x0000000002000001)
        self.assertEqual(word, 0x47E4CE4B896CDD1D)
        self.assertEqual(uniform_f32_from_word(word), 0.28083503246307373)

    def test_invalid_xorshift_inputs_fail_closed(self) -> None:
        for value in (-1, 1 << 64, True, 1.5):
            with self.subTest(value=value):
                with self.assertRaises(TransactionRngError):
                    xorshift64star_step(value)  # type: ignore[arg-type]
        with self.assertRaises(TransactionRngError):
            uniform_f32_from_word(-1)


class EagerStreamImpossibilityTests(unittest.TestCase):
    def test_depth_two_partial_prefix_has_unavoidable_discarded_draw(self) -> None:
        self.assertEqual(
            eager_sequential_draw_order(2),
            (
                "pending",
                "proposal[0]",
                "proposal[1]",
                "acceptance[0]",
                "acceptance[1]",
                "categorical[0]",
                "categorical[1]",
                "categorical[2]",
            ),
        )
        self.assertEqual(eager_prefix_obstructions(2, 1), ("proposal[1]",))

    def test_only_full_draft_prefix_avoids_eager_proposal_suffix(self) -> None:
        self.assertEqual(
            eager_prefix_obstructions(5, 2),
            ("proposal[2]", "proposal[3]", "proposal[4]"),
        )
        self.assertEqual(eager_prefix_obstructions(5, 5), ())
        self.assertEqual(eager_prefix_obstructions(5, 0), ())


class TransactionLedgerTests(unittest.TestCase):
    def test_sampled_full_commit_has_one_state_per_visible_prefix(self) -> None:
        rng = TransactionalRng(SEED, stream_position=41)
        reservation = rng.begin(SAMPLED, 4)
        ledger = rng.publish(
            reservation.transaction_id, 4, [True, True, True, True]
        )
        self.assertEqual(
            ledger.rng_after,
            (
                SEED,
                0xA69CF1E9AA4D68FC,
                0xEBFF37A39F313BF6,
                0x4A3B99C8687BF0C0,
                0xA9D6C08A3025649E,
            ),
        )
        self.assertEqual(
            rng.commit(reservation.transaction_id, 4, 4),
            ledger.rng_after[4],
        )
        self.assertEqual(rng.state, 0xA9D6C08A3025649E)
        self.assertEqual(rng.stream_position, 45)
        self.assertFalse(rng.active)

    def test_partial_commit_does_not_publish_suffix_draws(self) -> None:
        rng = TransactionalRng(SEED, stream_position=41)
        first = rng.begin(SAMPLED, 5)
        old_future_uniform = first.canonical_uniforms([True] * 5)[2]
        rng.publish(first.transaction_id, 5, [True] * 5)
        self.assertEqual(
            rng.commit(first.transaction_id, 2, 2),
            0xEBFF37A39F313BF6,
        )
        self.assertEqual(rng.stream_position, 43)

        resumed = rng.begin(SAMPLED, 3)
        self.assertEqual(
            resumed.canonical_uniforms([True, True, True])[0],
            old_future_uniform,
        )
        control = TransactionalRng(rng.state, rng.stream_position)
        same_boundary = control.begin(SAMPLED, 3)
        self.assertEqual(
            resumed.auxiliary_uniform("proposal", 0),
            same_boundary.auxiliary_uniform("proposal", 0),
        )
        rng.publish(resumed.transaction_id, 3, [True, True, True])
        rng.commit(resumed.transaction_id, 0, 0)

    def test_commit_zero_restores_rng_and_ordinary_fallback_draw(self) -> None:
        rng = TransactionalRng(1, stream_position=9)
        reservation = rng.begin(SAMPLED, 3)
        pending_uniform = reservation.canonical_uniforms([True, True, True])[0]
        rng.publish(reservation.transaction_id, 3, [True, True, True])
        self.assertEqual(rng.commit(reservation.transaction_id, 0, 0), 1)
        self.assertEqual(rng.state, 1)
        self.assertEqual(rng.stream_position, 9)

        next_state, next_word = xorshift64star_step(rng.state)
        self.assertEqual(uniform_f32_from_word(next_word), pending_uniform)
        self.assertEqual(next_state, 0x0000000002000001)

    def test_pending_stop_advances_rng_without_adopting_target_row(self) -> None:
        target_frontier = 17
        rng = TransactionalRng(SEED, stream_position=41)
        reservation = rng.begin(SAMPLED, 1)
        ledger = rng.publish(reservation.transaction_id, 1, [True])

        self.assertEqual(
            rng.commit(
                reservation.transaction_id,
                adopted_count=0,
                observed_count=1,
            ),
            ledger.rng_after[1],
        )
        self.assertEqual(rng.state, 0xA69CF1E9AA4D68FC)
        self.assertEqual(rng.stream_position, 42)
        self.assertEqual(target_frontier + 0, target_frontier)

    def test_suffix_stop_publishes_observed_draw_but_not_later_suffix(self) -> None:
        target_frontier = 100
        rng = TransactionalRng(SEED, stream_position=41)
        reservation = rng.begin(SAMPLED, 4)
        ledger = rng.publish(reservation.transaction_id, 4, [True] * 4)

        self.assertEqual(
            rng.commit(
                reservation.transaction_id,
                adopted_count=2,
                observed_count=3,
            ),
            ledger.rng_after[3],
        )
        self.assertEqual(rng.state, 0x4A3B99C8687BF0C0)
        self.assertEqual(rng.stream_position, 44)
        self.assertEqual(target_frontier + 2, 102)

        # The fourth sampled row was never inspected.  Its xorshift draw is
        # therefore still the next canonical public draw at the new boundary.
        resumed = rng.begin(SAMPLED, 1)
        fourth_uniform = reservation.canonical_uniforms([True] * 4)[3]
        self.assertEqual(resumed.canonical_uniforms([True])[0], fourth_uniform)

    def test_mixed_draw_and_no_draw_schedule(self) -> None:
        rng = TransactionalRng(SEED, stream_position=100)
        reservation = rng.begin(SAMPLED, 4)
        schedule = [False, True, False, True]
        ledger = rng.publish(reservation.transaction_id, 4, schedule)
        self.assertEqual(
            ledger.rng_after,
            (
                SEED,
                SEED,
                0xA69CF1E9AA4D68FC,
                0xA69CF1E9AA4D68FC,
                0xEBFF37A39F313BF6,
            ),
        )
        uniforms = reservation.canonical_uniforms(schedule)
        self.assertIsNone(uniforms[0])
        self.assertEqual(uniforms[1], 0.4866410493850708)
        self.assertIsNone(uniforms[2])
        self.assertEqual(uniforms[3], 0.8337453603744507)

        self.assertEqual(
            rng.commit(reservation.transaction_id, 3, 3),
            0xA69CF1E9AA4D68FC,
        )
        self.assertEqual(rng.stream_position, 103)

    def test_greedy_and_sampled_no_draw_paths_leave_state_unchanged(self) -> None:
        greedy_rng = TransactionalRng(SEED)
        greedy = greedy_rng.begin(GREEDY, 3)
        ledger = greedy_rng.publish(
            greedy.transaction_id, 3, [False, False, False]
        )
        self.assertEqual(ledger.rng_after, (SEED, SEED, SEED, SEED))
        self.assertEqual(
            greedy.canonical_uniforms([False, False, False]),
            (None, None, None),
        )
        with self.assertRaises(TransactionRngError):
            greedy.auxiliary_uniform("proposal", 0)
        with self.assertRaises(TransactionRngError):
            greedy.dspark_round_uniforms(1, [False, False, False])
        greedy_rng.commit(greedy.transaction_id, 3, 3)
        self.assertEqual(greedy_rng.state, SEED)

        fallback_rng = TransactionalRng(SEED)
        fallback = fallback_rng.begin(SAMPLED, 2)
        fallback_rng.publish(fallback.transaction_id, 2, [False, False])
        fallback_rng.commit(fallback.transaction_id, 2, 2)
        self.assertEqual(fallback_rng.state, SEED)
        self.assertEqual(fallback_rng.stream_position, 2)

        with self.assertRaises(TransactionRngError):
            invalid = TransactionalRng(SEED)
            reservation = invalid.begin(GREEDY, 2)
            invalid.publish(reservation.transaction_id, 2, [False, True])

    def test_depth_zero_is_word_uniform_and_state_identical(self) -> None:
        rng = TransactionalRng(SEED)
        reservation = rng.begin(SAMPLED, 2)
        first_state, first_word = xorshift64star_step(SEED)
        second_state, second_word = xorshift64star_step(first_state)
        self.assertEqual(
            reservation.canonical_uniforms([True, True])[0],
            uniform_f32_from_word(first_word),
        )
        self.assertEqual(
            reservation.canonical_uniforms([True, True])[1],
            uniform_f32_from_word(second_word),
        )
        ledger = rng.publish(reservation.transaction_id, 2, [True, True])
        self.assertEqual(ledger.rng_after, (SEED, first_state, second_state))
        self.assertEqual(
            rng.commit(reservation.transaction_id, 2, 2),
            second_state,
        )

    def test_domain_subdraws_are_deterministic_distinct_and_closed(self) -> None:
        first_rng = TransactionalRng(SEED, stream_position=41)
        first = first_rng.begin(SAMPLED, 4)
        second_rng = TransactionalRng(SEED, stream_position=41)
        second = second_rng.begin(SAMPLED, 4)

        expected_proposal = (
            0.7413484454154968,
            0.17584848403930664,
            0.08323049545288086,
            0.7316870093345642,
        )
        expected_acceptance = (
            0.09651243686676025,
            0.35172927379608154,
            0.491601824760437,
            0.5626910328865051,
        )
        for index in range(4):
            proposal = first.auxiliary_uniform("proposal", index)
            acceptance = first.auxiliary_uniform("acceptance", index)
            self.assertEqual(proposal, expected_proposal[index])
            self.assertEqual(acceptance, expected_acceptance[index])
            self.assertEqual(proposal, second.auxiliary_uniform("proposal", index))
            self.assertEqual(
                acceptance,
                second.auxiliary_uniform("acceptance", index),
            )
            self.assertNotEqual(proposal, acceptance)

    def test_round_uniform_mapping_keeps_categorical_on_public_tickets(self) -> None:
        rng = TransactionalRng(SEED, stream_position=41)
        reservation = rng.begin(SAMPLED, 4)
        draws = reservation.dspark_round_uniforms(
            2, [True, False, True, True]
        )
        self.assertEqual(draws.pending, 0.4866410493850708)
        self.assertEqual(
            draws.proposal,
            (0.17584848403930664, 0.08323049545288086),
        )
        self.assertEqual(
            draws.acceptance,
            (0.35172927379608154, 0.491601824760437),
        )
        self.assertEqual(
            draws.categorical,
            (None, 0.8337453603744507, 0.5340441465377808),
        )

    def test_policy_change_starts_cleanly_at_committed_boundary(self) -> None:
        rng = TransactionalRng(SEED, stream_position=5)
        first = rng.begin(SAMPLED, 3)
        first_fingerprint = first.policy.fingerprint()
        rng.publish(first.transaction_id, 3, [True, False, True])
        rng.commit(first.transaction_id, 2, 2)

        changed = SamplingPolicy(temperature=1.0, top_k=32, top_p=0.8, min_p=0.1)
        second = rng.begin(changed, 2)
        self.assertNotEqual(first_fingerprint, second.policy.fingerprint())
        self.assertEqual(second.start_state, 0xA69CF1E9AA4D68FC)
        self.assertEqual(second.start_position, 7)

        control = TransactionalRng(second.start_state, second.start_position)
        same_boundary = control.begin(SAMPLED, 2)
        self.assertEqual(
            second.auxiliary_uniform("proposal", 0),
            same_boundary.auxiliary_uniform("proposal", 0),
        )


class TransactionLifecycleTests(unittest.TestCase):
    def test_stale_and_out_of_order_operations_fail_closed(self) -> None:
        rng = TransactionalRng(SEED)
        reservation = rng.begin(SAMPLED, 2)
        with self.assertRaises(TransactionRngError):
            rng.begin(SAMPLED, 1)
        with self.assertRaises(StaleTransactionError):
            rng.publish(reservation.transaction_id + 1, 1, [True])
        with self.assertRaises(TransactionRngError):
            rng.commit(reservation.transaction_id, 0, 0)
        with self.assertRaises(TransactionRngError):
            rng.publish(True, 1, [True])
        with self.assertRaises(TransactionRngError):
            rng.publish(1.0, 1, [True])  # type: ignore[arg-type]
        with self.assertRaises(TransactionRngError):
            rng.publish(reservation.transaction_id, 3, [True, True, True])

        rng.publish(reservation.transaction_id, 2, [True, True])
        with self.assertRaises(TransactionRngError):
            rng.publish(reservation.transaction_id, 2, [True, True])
        with self.assertRaises(TransactionRngError):
            rng.commit(reservation.transaction_id, 3, 3)
        rng.commit(reservation.transaction_id, 1, 1)
        with self.assertRaises(StaleTransactionError):
            rng.commit(reservation.transaction_id, 0, 0)

        next_reservation = rng.begin(SAMPLED, 1)
        with self.assertRaises(StaleTransactionError):
            rng.publish(reservation.transaction_id, 1, [True])
        rng.publish(next_reservation.transaction_id, 1, [True])
        rng.commit(next_reservation.transaction_id, 0, 0)

    def test_adopted_and_observed_counts_fail_closed(self) -> None:
        invalid_counts = (
            (-1, 0),
            (1, 0),
            (0, 2),
            (1, 3),
            (2, 3),
            (True, 1),
            (0, False),
        )
        for adopted_count, observed_count in invalid_counts:
            with self.subTest(
                adopted_count=adopted_count,
                observed_count=observed_count,
            ):
                rng = TransactionalRng(SEED)
                reservation = rng.begin(SAMPLED, 2)
                rng.publish(reservation.transaction_id, 2, [True, True])
                snapshot = dict(rng.__dict__)
                with self.assertRaises(TransactionRngError):
                    rng.commit(
                        reservation.transaction_id,
                        adopted_count,  # type: ignore[arg-type]
                        observed_count,  # type: ignore[arg-type]
                    )
                self.assertEqual(rng.__dict__, snapshot)

    def test_cookie_exhaustion_and_invalid_begin_are_byte_identical(self) -> None:
        rng = TransactionalRng(SEED)
        for policy, capacity in ((SAMPLED, 0), (None, 1), ("bad", 1)):
            with self.subTest(policy=policy, capacity=capacity):
                snapshot = dict(rng.__dict__)
                with self.assertRaises(TransactionRngError):
                    rng.begin(policy, capacity)  # type: ignore[arg-type]
                self.assertEqual(rng.__dict__, snapshot)

        rng._last_transaction_id = (1 << 64) - 2
        final = rng.begin(SAMPLED, 1)
        self.assertEqual(final.transaction_id, (1 << 64) - 1)
        rng.publish(final.transaction_id, 1, [False])
        rng.commit(final.transaction_id, 0, 0)

        snapshot = dict(rng.__dict__)
        with self.assertRaises(TransactionRngError):
            rng.begin(SAMPLED, 1)
        self.assertEqual(rng.__dict__, snapshot)

    def test_malformed_policy_schedule_counts_and_domains_fail_closed(self) -> None:
        bad_policies = (
            dict(temperature=math.nan, top_k=0, top_p=1.0, min_p=0.0),
            dict(temperature=1.0e300, top_k=0, top_p=1.0, min_p=0.0),
            dict(temperature=1.0, top_k=1 << 31, top_p=1.0, min_p=0.0),
        )
        for values in bad_policies:
            with self.subTest(values=values):
                with self.assertRaises(TransactionRngError):
                    SamplingPolicy(**values)

        self.assertTrue(
            SamplingPolicy(temperature=1.0e-50, top_k=0, top_p=1.0, min_p=0.0).greedy
        )
        normalized = SamplingPolicy(
            temperature=1.0, top_k=-7, top_p=0.0, min_p=-0.5
        )
        self.assertEqual(normalized.top_k, 0)
        self.assertEqual(normalized.top_p, 1.0)
        self.assertEqual(normalized.min_p, 0.0)
        self.assertEqual(
            normalized.fingerprint(),
            SamplingPolicy(
                temperature=1.0,
                top_k=0,
                top_p=1.0,
                min_p=0.0,
            ).fingerprint(),
        )
        capped = SamplingPolicy(
            temperature=1.0, top_k=4096, top_p=2.0, min_p=1.25
        )
        self.assertEqual(capped.top_k, 1024)
        self.assertEqual(capped.top_p, 1.0)
        self.assertEqual(capped.min_p, 1.25)

        for capacity in (0, MAX_DSPARK_BLOCK_COUNT + 1, True):
            with self.subTest(capacity=capacity):
                with self.assertRaises(TransactionRngError):
                    TransactionalRng(SEED).begin(SAMPLED, capacity)

        with self.assertRaises(TransactionRngError):
            TransactionalRng(-1)
        with self.assertRaises(TransactionRngError):
            TransactionalRng(SEED, stream_position=(1 << 64) - 1).begin(
                SAMPLED, 1
            )

        rng = TransactionalRng(SEED)
        reservation = rng.begin(SAMPLED, 2)
        with self.assertRaises(TransactionRngError):
            reservation.canonical_uniforms([True, True, True])
        with self.assertRaises(TransactionRngError):
            reservation.canonical_uniforms([True, 1])
        with self.assertRaises(TransactionRngError):
            reservation.auxiliary_uniform("unknown", 0)
        with self.assertRaises(TransactionRngError):
            reservation.dspark_round_uniforms(1, [True, True, True])
        with self.assertRaises(TransactionRngError):
            reservation.dspark_round_uniforms(6, [True, True])
        with self.assertRaises(TransactionRngError):
            reservation.ledger(0, [True])


if __name__ == "__main__":
    unittest.main()
