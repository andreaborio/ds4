#!/usr/bin/env python3
"""Model-free fail-closed tests for the DSpark 8K cohort validator."""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.dspark_oracle import benchmark_cohort


MANIFEST_PATH = ROOT / "tools/dspark_oracle/benchmark_8k_abba.json"
RUNNER_PATH = ROOT / "speed-bench/run_m5_dsflash_arm.sh"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class BenchmarkCohortValidatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = benchmark_cohort.load_manifest(MANIFEST_PATH)
        benchmark_cohort.validate_manifest(cls.manifest, local=False)

    def make_cohort(self, root: Path) -> None:
        copied_manifest = root / "manifest.json"
        copied_manifest.write_text(
            json.dumps(self.manifest, indent=2) + "\n", encoding="utf-8"
        )
        (root / "plan.sha256").write_text(
            f"{sha256(copied_manifest)}  {copied_manifest}\n"
            f"{sha256(RUNNER_PATH)}  {RUNNER_PATH}\n",
            encoding="utf-8",
        )
        baseline_head = "a" * 40
        candidate_head = baseline_head
        (root / "identity.tsv").write_text(
            "baseline_head\tbaseline_main_head\tcandidate_head\n"
            f"{baseline_head}\t{baseline_head}\t{candidate_head}\n",
            encoding="utf-8",
        )
        support = self.manifest["artifacts"]["candidate"]
        support_identity = {
            "device": "16777234",
            "inode": "987654321",
            "bytes": str(support["support_bytes"]),
            "mtime_epoch": "1785781200",
        }
        model_hash = root / "model-hash"
        model_hash.mkdir()
        (model_hash / "support.txt").write_text(
            "schema=1\n"
            f"model_path_sha256={'7' * 64}\n"
            f"model_device={support_identity['device']}\n"
            f"model_inode={support_identity['inode']}\n"
            f"model_bytes={support_identity['bytes']}\n"
            f"model_mtime_epoch={support_identity['mtime_epoch']}\n"
            f"model_sha256_expected={support['support_sha256']}\n"
            f"model_sha256_actual={support['support_sha256']}\n"
            "verified_at=2026-08-03T20:19:58+02:00\n",
            encoding="utf-8",
        )
        identity_values = "\t".join(support_identity.values())
        (root / "oracle-support-identity.tsv").write_text(
            "timestamp\tphase\tdevice\tinode\tbytes\tmtime_epoch\n"
            f"2026-08-03T20:20:00+02:00\tbefore\t{identity_values}\n"
            f"2026-08-03T20:20:01+02:00\tafter\t{identity_values}\n",
            encoding="utf-8",
        )
        oracle_logs = {
            "system": (
                "........................................................................\n"
                "----------------------------------------------------------------------\n"
                "Ran 73 tests in 5.000s\n\n"
                "OK (skipped=1)\n"
            ),
            "mlx": (
                "........................................................................\n"
                "----------------------------------------------------------------------\n"
                "Ran 73 tests in 6.000s\n\n"
                "OK\n"
            ),
        }
        oracle_rows = [
            "timestamp\tlane\ttests\tskipped\tlog_sha256\tresult"
        ]
        for offset, lane in enumerate(("system", "mlx"), 2):
            log = root / f"oracle-{lane}.log"
            log.write_text(oracle_logs[lane], encoding="utf-8")
            skipped = "1" if lane == "system" else "0"
            oracle_rows.append(
                f"2026-08-03T20:20:0{offset}+02:00\t{lane}\t73\t"
                f"{skipped}\t{sha256(log)}\tpass"
            )
        (root / "oracle-results.tsv").write_text(
            "\n".join(oracle_rows) + "\n", encoding="utf-8"
        )

        order = ["timestamp\tarm\tvariant\tphase\tevent"]
        timestamp = dt.datetime.fromisoformat("2026-08-03T20:20:00+02:00")
        for arm, variant in benchmark_cohort.EXPECTED_ORDER:
            for phase in ("warmup", "retained"):
                for event in ("start", "complete"):
                    order.append(
                        f"{timestamp.isoformat()}\t{arm}\t{variant}\t{phase}\t{event}"
                    )
                    timestamp += dt.timedelta(seconds=1)
        (root / "order.tsv").write_text("\n".join(order) + "\n", encoding="utf-8")

        for arm, variant in benchmark_cohort.EXPECTED_ORDER:
            artifact = self.manifest["artifacts"][variant]
            is_baseline = variant == "baseline"
            repo_head = baseline_head if is_baseline else candidate_head
            prefix = root / arm
            summary = {
                "label": arm,
                "mode": "auto",
                "residency": "auto",
                "cache": "auto",
                "cache_state": "warm",
                "abort_reason": "none",
                "result_error": "none",
                "process_contamination": "none",
                "process_rc": "0",
                "rc": "0",
                "swapout_pages_delta": "0",
                "repo_untracked_count": "0",
                "host_memory_bytes": str(64 * 1024**3),
                "pressure_min": "40",
                "power_source": "AC Power",
                "model_sha256_expected": artifact["sha256"],
                "model_sha256_actual": artifact["sha256"],
                "model_bytes": str(artifact["bytes"]),
                "prompt_sha256": self.manifest["workload"]["prompt_sha256"],
                "ctx_start": "8192",
                "ctx_max": "8192",
                "ctx_alloc": "8321",
                "gen_tokens": "128",
                "bin": f"/fixture/{variant}/build/metal-arm64/bin/ds4-bench",
                "post_bin_sha256": "3" * 64,
                "repo_head": repo_head,
                "post_repo_head": repo_head,
                "repo_diff_sha256": benchmark_cohort.EMPTY_GIT_DIFF_SHA256,
                "post_repo_diff_sha256": benchmark_cohort.EMPTY_GIT_DIFF_SHA256,
                "post_repo_untracked_count": "0",
                "repo_untracked_manifest_sha256": "2" * 64,
                "post_repo_untracked_manifest_sha256": "2" * 64,
                "repo_source_state_sha256": "1" * 64,
                "post_repo_source_state_sha256": "1" * 64,
                "bin_sha256": "3" * 64,
                "metal_file_set_manifest_sha256": "5" * 64,
                "post_metal_file_set_manifest_sha256": "5" * 64,
                "prompt_source_sha256": "8" * 64,
                "post_prompt_source_sha256": "8" * 64,
                "post_prompt_sha256": self.manifest["workload"]["prompt_sha256"],
                "prompt_bytes": "65536",
                "post_prompt_bytes": "65536",
                "os_build": "fixture-build",
            }
            Path(f"{prefix}.summary").write_text(
                "".join(f"{key}={value}\n" for key, value in summary.items()),
                encoding="utf-8",
            )
            plan = (
                f"ds4: build git={repo_head[:12]} compiled=metal-arm64 "
                "runtime=metal\n"
                "ds4: residency requested=auto resolved=ssd: fixture\n"
            )
            if is_baseline:
                plan += "ds4:   cached expert count: 4129 (fixture)\n"
            else:
                plan += (
                    "ds4:   DSpark static payload 553290668 bytes; 16K page union "
                    "TARGET 8.20 GiB + SUPPORT delta 0.52 GiB = 8.71 GiB; "
                    "speculative runtime 28.30 MiB (capture 1.00 MiB)\n"
                    "ds4:   DSpark cache: cycle-aligned parent 4160 = TARGET 4129 "
                    "+ SUPPORT 31\n"
                )
            Path(f"{prefix}.stderr").write_text(plan, encoding="utf-8")
            Path(f"{prefix}.resolved-plan").write_text(plan, encoding="utf-8")
            Path(f"{prefix}.build-info").write_text(
                "ds4 build\n"
                f"git:     {repo_head[:12]}\n"
                "backend: metal\n"
                "arch:    arm64\n",
                encoding="utf-8",
            )
            Path(f"{prefix}.metal-library").write_text(
                "ds4: metal_library "
                f"source_sha256={'6' * 64} overrides=0 tensor=on "
                "norm_unify=off kv_raw_f32=off math=fast\n",
                encoding="utf-8",
            )

            if is_baseline:
                values = {
                    "prefill_tps": 100.0,
                    "prefill_wall_ms": 1000.0,
                    "ttft_ms": 1000.0,
                    "gen_tps": 10.0,
                    "gen_wall_ms": 1000.0,
                    "gen_tpot_p50_ms": 100.0,
                    "gen_tpot_p95_ms": 110.0,
                    "prefill_pread_gib": 1.0,
                    "prefill_pread_gib_per_tok": 1.0 / 8192.0,
                    "gen_pread_gib": 0.1,
                    "gen_pread_gib_per_tok": 0.1 / 128.0,
                }
            else:
                values = {
                    "prefill_tps": 105.0,
                    "prefill_wall_ms": 900.0,
                    "ttft_ms": 900.0,
                    "gen_tps": 11.0,
                    "gen_wall_ms": 900.0,
                    "gen_tpot_p50_ms": 90.0,
                    "gen_tpot_p95_ms": 100.0,
                    "prefill_pread_gib": 0.8,
                    "prefill_pread_gib_per_tok": 0.8 / 8192.0,
                    "gen_pread_gib": 0.08,
                    "gen_pread_gib_per_tok": 0.08 / 128.0,
                }
            csv_values = {
                "ctx_tokens": 8192,
                "prefill_tokens": 8192,
                "gen_tokens": 128,
                **values,
            }
            Path(f"{prefix}.csv").write_text(
                ",".join(csv_values) + "\n"
                + ",".join(str(value) for value in csv_values.values()) + "\n",
                encoding="utf-8",
            )

            logits_dir = Path(f"{prefix}.logits")
            evidence_dir = Path(f"{prefix}.evidence")
            logits_dir.mkdir()
            evidence_dir.mkdir()
            (logits_dir / "frontier_008192.logits.json").write_text(
                json.dumps({
                    "source": arm,
                    "model": f"fixture-{variant}.gguf",
                    "frontier_tokens": 8192,
                    "argmax_id": 1,
                    "logits": [0.0, 1.0],
                }) + "\n",
                encoding="utf-8",
            )
            (evidence_dir / "frontier_008192.decode.json").write_text(
                json.dumps({
                    "schema": "ds4.qwen.decode-evidence/1",
                    "frontier_tokens": 8192,
                    "token_ids": [1] * 128,
                    "final_argmax_id": 1,
                    "final_logits": [0.0, 1.0],
                }) + "\n",
                encoding="utf-8",
            )

    def validate(self, root: Path) -> tuple[list[dict[str, object]], list[str]]:
        return benchmark_cohort.validate_results(root, self.manifest)

    def test_valid_synthetic_cohort_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_cohort(root)
            rows, errors = self.validate(root)
            self.assertEqual([row["arm"] for row in rows], ["A1", "B1", "B2", "A2"])
            self.assertEqual(errors, [])

    def test_nontransactional_or_terminal_eos_policy_fails_closed(self) -> None:
        for key, value in (
            ("implementation", "ds4-bench argmax excluding EOS"),
            ("eos_policy", "terminal"),
        ):
            manifest = json.loads(json.dumps(self.manifest))
            manifest["workload"]["sampling"][key] = value
            with self.assertRaises(benchmark_cohort.ManifestError):
                benchmark_cohort.validate_manifest(manifest, local=False)

    def test_order_mutation_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_cohort(root)
            path = root / "order.tsv"
            path.write_text(path.read_text().replace("\tB1\t", "\tX1\t", 1))
            with self.assertRaisesRegex(benchmark_cohort.ManifestError, "A1,B1,B2,A2"):
                self.validate(root)

    def test_parity_mutation_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_cohort(root)
            path = root / "B1.evidence/frontier_008192.decode.json"
            payload = json.loads(path.read_text())
            payload["token_ids"][0] = 2
            path.write_text(json.dumps(payload) + "\n")
            _, errors = self.validate(root)
            self.assertTrue(any("decode_output_sha256 parity failed" in error for error in errors))

    def test_swap_and_abort_mutation_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_cohort(root)
            path = root / "B1.summary"
            text = path.read_text().replace("abort_reason=none", "abort_reason=timeout")
            text = text.replace("swapout_pages_delta=0", "swapout_pages_delta=1")
            path.write_text(text)
            _, errors = self.validate(root)
            self.assertTrue(any("abort_reason" in error for error in errors))
            self.assertTrue(any("swapout_pages_delta" in error for error in errors))

    def test_plan_hash_mutation_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_cohort(root)
            path = root / "plan.sha256"
            lines = path.read_text().splitlines()
            lines[1] = "0" * 64 + "  " + str(RUNNER_PATH)
            path.write_text("\n".join(lines) + "\n")
            with self.assertRaisesRegex(benchmark_cohort.ManifestError, "runner hash"):
                self.validate(root)

    def test_stale_binary_build_identity_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_cohort(root)
            stale = "c" * 12
            for arm in ("B1", "B2"):
                build_info = Path(f"{root / arm}.build-info")
                build_info.write_text(
                    build_info.read_text().replace("a" * 12, stale)
                )
                for suffix in ("stderr", "resolved-plan"):
                    plan = Path(f"{root / arm}.{suffix}")
                    plan.write_text(plan.read_text().replace("a" * 12, stale))
            _, errors = self.validate(root)
            self.assertTrue(any(
                "compiled build identity does not match clean repo_head" in error
                for error in errors
            ))
            self.assertTrue(any(
                "runtime build line does not match repo_head" in error
                for error in errors
            ))

    def test_runtime_metal_library_identity_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_cohort(root)
            identity = root / "B1.metal-library"
            identity.write_text(identity.read_text().replace("6" * 64, "7" * 64))
            _, errors = self.validate(root)
            self.assertTrue(any(
                "executable/repository/Metal identity differs" in error
                for error in errors
            ))

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_cohort(root)
            identity = root / "B1.metal-library"
            identity.write_text(identity.read_text() * 2)
            with self.assertRaisesRegex(
                benchmark_cohort.ManifestError,
                "runtime Metal identity must be one complete line",
            ):
                self.validate(root)

        mutations = {
            "post_bin_sha256": "4" * 64,
            "post_repo_head": "b" * 40,
            "post_repo_diff_sha256": "9" * 64,
            "post_repo_untracked_count": "1",
            "post_repo_untracked_manifest_sha256": "4" * 64,
            "post_repo_source_state_sha256": "4" * 64,
            "post_metal_file_set_manifest_sha256": "4" * 64,
            "post_prompt_source_sha256": "4" * 64,
            "post_prompt_sha256": "4" * 64,
            "post_prompt_bytes": "65537",
        }
        for field, replacement in mutations.items():
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                self.make_cohort(root)
                summary = root / "A2.summary"
                lines = summary.read_text(encoding="utf-8").splitlines()
                changed = False
                for index, line in enumerate(lines):
                    if line.startswith(field + "="):
                        lines[index] = f"{field}={replacement}"
                        changed = True
                        break
                self.assertTrue(changed)
                summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
                _, errors = self.validate(root)
                self.assertTrue(
                    any(
                        "post-arm" in error or field in error
                        for error in errors
                    ),
                    errors,
                )

    def test_different_a_b_revision_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_cohort(root)
            identity = root / "identity.tsv"
            identity.write_text(
                identity.read_text().replace("\t" + "a" * 40 + "\n", "\t" + "b" * 40 + "\n")
            )
            with self.assertRaisesRegex(
                benchmark_cohort.ManifestError,
                "one identical product revision",
            ):
                self.validate(root)

    def test_support_oracle_hash_and_identity_mutations_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_cohort(root)
            evidence = root / "model-hash/support.txt"
            evidence.write_text(evidence.read_text().replace(
                self.manifest["artifacts"]["candidate"]["support_sha256"],
                "0" * 64,
                1,
            ))
            with self.assertRaisesRegex(
                benchmark_cohort.ManifestError,
                "SUPPORT oracle hash evidence",
            ):
                self.validate(root)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_cohort(root)
            identity = root / "oracle-support-identity.tsv"
            lines = identity.read_text().splitlines()
            lines[2] = lines[2].replace("1785781200", "1785781201")
            identity.write_text("\n".join(lines) + "\n")
            with self.assertRaisesRegex(
                benchmark_cohort.ManifestError,
                "changed during oracle runs",
            ):
                self.validate(root)

    def test_mlx_skip_and_oracle_count_drift_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_cohort(root)
            log = root / "oracle-mlx.log"
            log.write_text(log.read_text().replace("\nOK\n", "\nOK (skipped=1)\n"))
            with self.assertRaisesRegex(
                benchmark_cohort.ManifestError,
                "mlx oracle skipped 1 tests, expected 0",
            ):
                self.validate(root)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_cohort(root)
            log = root / "oracle-system.log"
            log.write_text(log.read_text().replace("Ran 73 tests", "Ran 72 tests"))
            with self.assertRaisesRegex(
                benchmark_cohort.ManifestError,
                "system oracle ran 72 tests, expected 73",
            ):
                self.validate(root)

    def test_repeated_tracked_diff_still_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_cohort(root)
            for arm in ("A1", "A2"):
                summary = root / f"{arm}.summary"
                summary.write_text(summary.read_text().replace(
                    "repo_diff_sha256=" + benchmark_cohort.EMPTY_GIT_DIFF_SHA256,
                    "repo_diff_sha256=" + "9" * 64,
                ))
            _, errors = self.validate(root)
            self.assertEqual(
                sum("tracked worktree diff" in error for error in errors), 2
            )
            self.assertFalse(any("A1/A2" in error for error in errors))

    def test_dry_run_uses_only_canonical_worktree_binaries(self) -> None:
        script = benchmark_cohort.render_dry_run(MANIFEST_PATH, self.manifest)
        runner = RUNNER_PATH.read_text(encoding="utf-8")
        self.assertNotIn("HEBRUS_DSPARK_BASELINE_BIN", script)
        self.assertNotIn("HEBRUS_DSPARK_CANDIDATE_BIN", script)
        self.assertIn(
            "BASELINE_BIN=$HEBRUS_DSPARK_BASELINE_ROOT/"
            "build/metal-arm64/bin/ds4-bench",
            script,
        )
        self.assertIn("bin=$BASELINE_BIN", script)
        self.assertIn("bin=$CANDIDATE_BIN", script)
        self.assertIn("model-hash/support.txt", script)
        self.assertIn("check_transactional_bench", script)
        self.assertLess(
            script.index("cohort blocked: production DSpark N>1"),
            script.index("--prepare-model-hash-evidence"),
        )
        self.assertIn("oracle-system.log", script)
        self.assertIn("--validate-oracle-log mlx", script)
        self.assertNotIn("grep -m 1 '^ds4: metal_library '", runner)
        self.assertIn("metal_library_identity_count", runner)
        for field in (
            "post_bin_sha256",
            "post_repo_head",
            "post_repo_diff_sha256",
            "post_repo_untracked_count",
            "post_repo_untracked_manifest_sha256",
            "post_repo_source_state_sha256",
            "post_metal_file_set_manifest_sha256",
            "post_prompt_source_sha256",
            "post_prompt_sha256",
            "post_prompt_bytes",
        ):
            self.assertIn(field, runner)
        self.assertIn("abort_reason=runtime_input_identity_changed_during_arm", runner)

    def test_cache_split_mutation_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_cohort(root)
            for suffix in ("stderr", "resolved-plan"):
                path = Path(f"{root / 'B1'}.{suffix}")
                path.write_text(path.read_text().replace("parent 4160", "parent 4159"))
            _, errors = self.validate(root)
            self.assertTrue(any("cache-record split" in error for error in errors))

    def test_csv_and_identity_mutations_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_cohort(root)
            csv_path = root / "B2.csv"
            csv_path.write_text(csv_path.read_text().replace("8192,8192,128", "8191,8192,128", 1))
            summary_path = root / "A2.summary"
            summary_path.write_text(summary_path.read_text().replace("repo_head=" + "a" * 40, "repo_head=" + "c" * 40))
            _, errors = self.validate(root)
            self.assertTrue(any("CSV ctx_tokens" in error for error in errors))
            self.assertTrue(any("identity drift" in error for error in errors))

    def test_accounting_table_is_derived_from_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_cohort(root)
            rows, errors = self.validate(root)
            output = io.StringIO()
            with redirect_stdout(output):
                benchmark_cohort.print_results(rows, errors, self.manifest)
            rendered = output.getvalue()
            self.assertIn("29671424", rendered)
            self.assertIn("582959104", rendered)
            self.assertNotIn("29655040", rendered)


if __name__ == "__main__":
    unittest.main()
