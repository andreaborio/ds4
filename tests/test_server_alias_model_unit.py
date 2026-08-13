#!/usr/bin/env python3
"""Model-free regression tests for the server alias release gate helpers."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import shutil
import socket
import sys
import tempfile
import types
import unittest
import urllib.error
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tests" / "test_server_alias_model.py"
SPEC = importlib.util.spec_from_file_location("server_alias_model", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import {MODULE_PATH}")
SERVER_ALIAS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SERVER_ALIAS
SPEC.loader.exec_module(SERVER_ALIAS)


def capability(engine_id: str, backend: str = "metal") -> dict[str, object]:
    return {
        "schema_version": 1,
        "engine_id": engine_id,
        "build_git_sha": "123456789abc",
        "backend": backend,
        "executable_role": "server",
        "model_families": ["qwen35moe"],
    }


def completion(content: object) -> dict[str, object]:
    return {
        "id": "chatcmpl-volatile",
        "created": 123,
        "choices": [{"message": {"role": "assistant", "content": content}}],
    }


class FakeProcess:
    returncode = None

    def poll(self) -> None:
        return None


class ServerAliasModelUnitTests(unittest.TestCase):
    def test_completion_normalization_preserves_nested_and_model_ids(self) -> None:
        document = {
            "id": "chatcmpl-volatile",
            "created": 123,
            "model": "qwen3.6-35b-a3b",
            "choices": [
                {
                    "message": {
                        "tool_calls": [{"id": "stable-tool-id"}],
                    }
                }
            ],
        }
        normalized = SERVER_ALIAS.normalize_completion(document)
        self.assertNotIn("id", normalized)
        self.assertNotIn("created", normalized)
        self.assertEqual(normalized["model"], "qwen3.6-35b-a3b")
        self.assertEqual(
            normalized["choices"][0]["message"]["tool_calls"][0]["id"],
            "stable-tool-id",
        )

    def test_completion_content_requires_exact_nonempty_marker(self) -> None:
        valid = completion(f"  {SERVER_ALIAS.EXPECTED_COMPLETION_MARKER}\n")
        self.assertEqual(
            SERVER_ALIAS.validate_completion_content(valid, "hebrus-server"),
            SERVER_ALIAS.EXPECTED_COMPLETION_MARKER,
        )

        invalid_documents = (
            completion(""),
            completion("   \n"),
            completion(None),
            completion("HEBRUS ALIAS PARITY"),
            completion(f"prefix {SERVER_ALIAS.EXPECTED_COMPLETION_MARKER}"),
            {"choices": []},
            {"choices": [{"message": {}}]},
        )
        for document in invalid_documents:
            with self.subTest(document=document):
                with self.assertRaises(AssertionError):
                    SERVER_ALIAS.validate_completion_content(
                        document, "hebrus-server"
                    )

    def test_environment_preserves_lock_and_reports_only_removed_names(self) -> None:
        environment, removed = SERVER_ALIAS.build_environment(
            {
                "PATH": "/usr/bin",
                "DS4_LOCK_FILE": "/tmp/qualified.lock",
                "DS4_TOKEN_TIMING": "1",
                "HEBRUS_EXPERIMENT": "1",
            }
        )
        self.assertEqual(environment["DS4_LOCK_FILE"], "/tmp/qualified.lock")
        self.assertNotIn("DS4_TOKEN_TIMING", environment)
        self.assertNotIn("HEBRUS_EXPERIMENT", environment)
        self.assertEqual(removed, ["DS4_TOKEN_TIMING", "HEBRUS_EXPERIMENT"])

    def test_environment_sets_explicit_default_lock(self) -> None:
        environment, removed = SERVER_ALIAS.build_environment({"PATH": "/usr/bin"})
        self.assertEqual(environment["DS4_LOCK_FILE"], SERVER_ALIAS.DEFAULT_LOCK_FILE)
        self.assertEqual(removed, [])

    def test_host_power_parsers_select_the_active_source(self) -> None:
        battery = SERVER_ALIAS.parse_battery_status(
            "Now drawing from 'AC Power'\n"
            " -InternalBattery-0\t100%; charged; 0:00 remaining present: true\n"
        )
        self.assertEqual(
            battery,
            {
                "source": "AC Power",
                "battery_percent": 100,
                "battery_state": "charged",
            },
        )
        custom = (
            "Battery Power:\n powermode 1\n"
            "AC Power:\n powermode 2\n"
        )
        self.assertEqual(
            SERVER_ALIAS.parse_powermode(custom, battery["source"]), 2
        )
        self.assertIsNone(SERVER_ALIAS.parse_powermode(custom, "UPS Power"))
        self.assertEqual(
            SERVER_ALIAS.parse_battery_status(None)["source"], None
        )

    def test_host_metadata_collection_omits_sensitive_network_identity(self) -> None:
        values = {
            ("pmset", "-g", "batt"): (
                "Now drawing from 'AC Power'\n"
                " -InternalBattery-0 87%; charging; 0:10 remaining\n"
            ),
            ("pmset", "-g", "custom"): "AC Power:\n powermode 2\n",
            ("sysctl", "-n", "hw.model"): "Mac17,1",
            ("sysctl", "-n", "machdep.cpu.brand_string"): "Apple M5 Pro",
            ("sysctl", "-n", "hw.memsize"): "68719476736",
            ("sw_vers", "-productVersion"): "26.5.2",
            ("sw_vers", "-buildVersion"): "25F90",
        }
        host = SERVER_ALIAS.collect_host_metadata(
            lambda argv: values.get(tuple(argv)),
            system_name="Darwin",
            architecture="arm64",
        )
        self.assertEqual(host["architecture"], "arm64")
        self.assertEqual(host["physical_memory_bytes"], 68719476736)
        self.assertEqual(host["os"]["name"], "macOS")
        self.assertEqual(host["os"]["build"], "25F90")
        self.assertEqual(host["power"]["powermode"], 2)
        self.assertNotIn("hostname", host)
        self.assertNotIn("address", json.dumps(host).lower())

    def test_qualified_host_validation_fails_closed(self) -> None:
        host = {
            "architecture": "arm64",
            "physical_memory_bytes": 68719476736,
            "os": {"name": "macOS", "version": "26.5.2", "build": "25F90"},
        }
        SERVER_ALIAS.validate_qualified_host(host, "metal")
        with self.assertRaisesRegex(AssertionError, "Metal"):
            SERVER_ALIAS.validate_qualified_host(host, "cpu")
        with self.assertRaisesRegex(AssertionError, "arm64"):
            SERVER_ALIAS.validate_qualified_host(
                {**host, "architecture": "x86_64"}, "metal"
            )
        with self.assertRaisesRegex(AssertionError, "macOS"):
            SERVER_ALIAS.validate_qualified_host(
                {**host, "os": {"name": "Linux", "version": "1", "build": "1"}},
                "metal",
            )
        with self.assertRaisesRegex(AssertionError, "physical memory"):
            SERVER_ALIAS.validate_qualified_host(
                {**host, "physical_memory_bytes": None}, "metal"
            )

    def test_capability_validation_rejects_wrong_backend_or_dirty_build(self) -> None:
        valid = capability("hebrus")
        SERVER_ALIAS.validate_capability_document(
            valid,
            binary_name="hebrus-server",
            engine_id="hebrus",
            expected_backend="metal",
            expected_build_sha="123456789abcdef0",
        )

        wrong_backend = dict(valid, backend="cpu")
        with self.assertRaisesRegex(AssertionError, "expected backend"):
            SERVER_ALIAS.validate_capability_document(
                wrong_backend,
                binary_name="hebrus-server",
                engine_id="hebrus",
                expected_backend="metal",
                expected_build_sha="123456789abcdef0",
            )

        dirty = dict(valid, build_git_sha="123456789abc-dirty")
        with self.assertRaisesRegex(AssertionError, "does not match candidate"):
            SERVER_ALIAS.validate_capability_document(
                dirty,
                binary_name="hebrus-server",
                engine_id="hebrus",
                expected_backend="metal",
                expected_build_sha="123456789abcdef0",
            )

    def test_build_identity_requires_a_clean_hex_commit_prefix(self) -> None:
        self.assertTrue(
            SERVER_ALIAS.build_sha_matches(
                "123456789abc", "123456789abc" + ("d" * 28)
            )
        )
        self.assertFalse(
            SERVER_ALIAS.build_sha_matches("123456789abc-dirty", "123456789abc")
        )
        self.assertFalse(SERVER_ALIAS.build_sha_matches("unknown", "123456789abc"))

    def test_repository_cleanliness_rejects_tracked_or_untracked_changes(self) -> None:
        clean = SERVER_ALIAS.subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        dirty = SERVER_ALIAS.subprocess.CompletedProcess(
            args=[], returncode=0, stdout="?? release-notes.md\n", stderr=""
        )
        with mock.patch.object(SERVER_ALIAS.subprocess, "run", return_value=clean):
            SERVER_ALIAS.assert_clean_repository(pathlib.Path("/candidate"))
        with mock.patch.object(SERVER_ALIAS.subprocess, "run", return_value=dirty):
            with self.assertRaisesRegex(AssertionError, "exact clean committed tree"):
                SERVER_ALIAS.assert_clean_repository(pathlib.Path("/candidate"))

    def test_capability_pair_rejects_drift_beyond_engine_identity(self) -> None:
        canonical = capability("hebrus")
        legacy = capability("ds4")
        SERVER_ALIAS.validate_capability_pair_documents(
            canonical,
            legacy,
            canonical_name="hebrus-server",
            legacy_name="ds4-server",
            expected_backend="metal",
            expected_build_sha="123456789abcdef0",
        )
        legacy["model_families"] = ["deepseek4"]
        with self.assertRaisesRegex(AssertionError, "beyond engine_id"):
            SERVER_ALIAS.validate_capability_pair_documents(
                canonical,
                legacy,
                canonical_name="hebrus-server",
                legacy_name="ds4-server",
                expected_backend="metal",
                expected_build_sha="123456789abcdef0",
            )

    def test_contract_is_loaded_from_manifest_and_rejects_invalid_digest(self) -> None:
        with tempfile.TemporaryDirectory(prefix="alias-contract-") as directory:
            path = pathlib.Path(directory) / "contract.json"
            document = {
                "schemaVersion": 2,
                "publishedArtifact": {
                    "status": "published",
                    "filename": "qualified.gguf",
                    "bytes": 3,
                    "sha256": "a" * 64,
                    "revision": "b" * 40,
                    "runtimeCommit": "c" * 40,
                },
            }
            path.write_text(json.dumps(document), encoding="utf-8")
            artifact = SERVER_ALIAS.load_published_artifact(path)
            self.assertEqual(artifact.filename, "qualified.gguf")
            self.assertEqual(artifact.size, 3)

            document["schemaVersion"] = 1
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(AssertionError, "schemaVersion 2"):
                SERVER_ALIAS.load_published_artifact(path)

            document["schemaVersion"] = 2
            document["publishedArtifact"]["sha256"] = "not-a-digest"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(AssertionError, "SHA-256"):
                SERVER_ALIAS.load_published_artifact(path)

    def test_checked_in_contract_is_accepted_by_alias_gate_loader(self) -> None:
        artifact = SERVER_ALIAS.load_published_artifact()
        self.assertGreater(artifact.size, 0)
        self.assertEqual(len(artifact.sha256), 64)

    def test_model_verification_requires_manifest_filename_and_hash(self) -> None:
        with tempfile.TemporaryDirectory(prefix="alias-model-") as directory:
            model = pathlib.Path(directory) / "qualified.gguf"
            model.write_bytes(b"abc")
            artifact = SERVER_ALIAS.PublishedArtifact(
                filename="qualified.gguf",
                size=3,
                sha256="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
                revision="b" * 40,
                runtime_commit="c" * 40,
            )
            resolved, digest, signature = SERVER_ALIAS.verify_model(model, artifact)
            self.assertEqual(resolved, model.resolve())
            self.assertEqual(digest, artifact.sha256)
            self.assertEqual(signature.size, 3)

            renamed = pathlib.Path(directory) / "renamed.gguf"
            renamed.write_bytes(b"abc")
            with self.assertRaisesRegex(AssertionError, "filename"):
                SERVER_ALIAS.verify_model(renamed, artifact)

    def test_default_evidence_directory_is_persistent(self) -> None:
        evidence = SERVER_ALIAS.prepare_evidence_dir(None)
        try:
            self.assertTrue(evidence.is_dir())
            marker = evidence / "marker"
            marker.write_text("retained", encoding="utf-8")
            self.assertEqual(marker.read_text(encoding="utf-8"), "retained")
        finally:
            shutil.rmtree(evidence)

    def test_evidence_manifest_is_deterministic_and_non_self_referential(self) -> None:
        with tempfile.TemporaryDirectory(prefix="alias-evidence-") as directory:
            evidence = pathlib.Path(directory)
            (evidence / "server-alias-parity.json").write_text(
                '{"status":"PASS"}\n', encoding="utf-8"
            )
            (evidence / "hebrus-server.log").write_text(
                "canonical log\n", encoding="utf-8"
            )
            first = SERVER_ALIAS.build_evidence_manifest(evidence)
            manifest_path = SERVER_ALIAS.write_evidence_manifest(evidence)
            second = SERVER_ALIAS.build_evidence_manifest(evidence)

            self.assertEqual(first, second)
            self.assertEqual(
                [entry["path"] for entry in first["artifacts"]],
                ["hebrus-server.log", "server-alias-parity.json"],
            )
            self.assertNotIn(
                SERVER_ALIAS.EVIDENCE_MANIFEST_NAME,
                [entry["path"] for entry in first["artifacts"]],
            )
            written = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(written, first)
            core = {
                "schema_version": written["schema_version"],
                "artifacts": written["artifacts"],
            }
            canonical = json.dumps(
                core, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            ).encode("utf-8")
            self.assertEqual(
                written["bundle_sha256"], hashlib.sha256(canonical).hexdigest()
            )

            (evidence / "hebrus-server.log").write_text(
                "changed log\n", encoding="utf-8"
            )
            changed = SERVER_ALIAS.build_evidence_manifest(evidence)
            self.assertNotEqual(changed["bundle_sha256"], first["bundle_sha256"])

    def test_port_selection_and_collision_preflight(self) -> None:
        selected = SERVER_ALIAS.select_port(0)
        self.assertGreaterEqual(selected, 1024)
        SERVER_ALIAS.assert_port_available(selected)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as held:
            held.bind(("127.0.0.1", 0))
            occupied = int(held.getsockname()[1])
            with self.assertRaisesRegex(AssertionError, "already in use"):
                SERVER_ALIAS.assert_port_available(occupied)

    def test_startup_http_probe_is_bounded_by_remaining_deadline(self) -> None:
        with tempfile.TemporaryDirectory(prefix="alias-listener-") as directory:
            log_path = pathlib.Path(directory) / "server.log"
            log_path.write_text(
                "ds4-server: listening on http://127.0.0.1:18081\n",
                encoding="utf-8",
            )
            with mock.patch.object(
                SERVER_ALIAS,
                "http_json",
                side_effect=urllib.error.URLError("not ready"),
            ) as request:
                started = SERVER_ALIAS.time.monotonic()
                with self.assertRaisesRegex(AssertionError, "startup timed out"):
                    SERVER_ALIAS.wait_for_models(
                        "http://127.0.0.1:18081",
                        FakeProcess(),
                        log_path,
                        18081,
                        0.05,
                    )
                elapsed = SERVER_ALIAS.time.monotonic() - started
            self.assertLess(elapsed, 0.5)
            self.assertTrue(request.called)
            self.assertLessEqual(request.call_args.kwargs["timeout"], 0.05)

    def test_local_http_opener_disables_environment_proxies(self) -> None:
        proxy_handlers = [
            handler
            for handler in SERVER_ALIAS.LOCAL_HTTP_OPENER.handlers
            if isinstance(handler, SERVER_ALIAS.urllib.request.ProxyHandler)
        ]
        self.assertFalse(any(handler.proxies for handler in proxy_handlers))

    def test_graceful_process_cleanup_uses_sigterm(self) -> None:
        process = mock.Mock()
        process.poll.return_value = None
        process.returncode = 0
        process.wait.return_value = 0
        self.assertEqual(SERVER_ALIAS.terminate_process(process, "server", 1.0), 0)
        process.send_signal.assert_called_once_with(SERVER_ALIAS.signal.SIGTERM)
        process.kill.assert_not_called()

    def test_cleanup_kills_after_graceful_timeout(self) -> None:
        process = mock.Mock()
        process.poll.return_value = None
        process.returncode = -9
        process.wait.side_effect = [
            SERVER_ALIAS.subprocess.TimeoutExpired("server", 0.01),
            -9,
        ]
        with self.assertRaisesRegex(AssertionError, "graceful shutdown timed out"):
            SERVER_ALIAS.terminate_process(process, "server", 0.01)
        process.kill.assert_called_once_with()
        self.assertEqual(process.wait.call_count, 2)

    def test_run_server_cleans_up_after_keyboard_interrupt(self) -> None:
        with tempfile.TemporaryDirectory(prefix="alias-interrupt-") as directory:
            evidence = pathlib.Path(directory)
            process = mock.Mock()
            with (
                mock.patch.object(SERVER_ALIAS, "assert_port_available"),
                mock.patch.object(
                    SERVER_ALIAS.subprocess, "Popen", return_value=process
                ),
                mock.patch.object(
                    SERVER_ALIAS, "wait_for_models", side_effect=KeyboardInterrupt()
                ),
                mock.patch.object(
                    SERVER_ALIAS, "terminate_process", return_value=0
                ) as terminate,
            ):
                with self.assertRaises(KeyboardInterrupt):
                    SERVER_ALIAS.run_server(
                        pathlib.Path("/candidate/hebrus-server"),
                        pathlib.Path("/models/qualified.gguf"),
                        18081,
                        1.0,
                        1.0,
                        1.0,
                        evidence,
                        {},
                    )
            terminate.assert_called_once_with(process, "hebrus-server", 1.0)

    def test_main_records_fail_and_manifest_for_keyboard_interrupt(self) -> None:
        with tempfile.TemporaryDirectory(prefix="alias-main-interrupt-") as directory:
            root = pathlib.Path(directory)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            (bin_dir / SERVER_ALIAS.CANONICAL_SERVER_NAME).write_bytes(b"binary")
            (bin_dir / SERVER_ALIAS.LEGACY_SERVER_NAME).write_bytes(b"binary")
            evidence = root / "evidence"
            args = types.SimpleNamespace(
                evidence_dir=evidence,
                bin_dir=bin_dir,
                expected_backend="metal",
                expected_build_sha="123456789abc",
                model=root / "qualified.gguf",
                port=0,
                startup_timeout=1.0,
                request_timeout=1.0,
                shutdown_timeout=1.0,
            )
            host = {
                "architecture": "arm64",
                "physical_memory_bytes": 68719476736,
                "os": {
                    "name": "macOS",
                    "version": "26.5.2",
                    "build": "25F90",
                },
                "power": {},
            }
            with (
                mock.patch.object(SERVER_ALIAS, "parse_args", return_value=args),
                mock.patch.object(SERVER_ALIAS, "install_termination_signal_handlers"),
                mock.patch.object(SERVER_ALIAS, "collect_host_metadata", return_value=host),
                mock.patch.object(SERVER_ALIAS.os.path, "samefile", return_value=True),
                mock.patch.object(
                    SERVER_ALIAS, "repository_head", side_effect=KeyboardInterrupt()
                ),
            ):
                with self.assertRaises(KeyboardInterrupt):
                    SERVER_ALIAS.main()

            report = json.loads(
                (evidence / "server-alias-parity.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["schema_version"], 3)
            self.assertEqual(report["status"], "FAIL")
            self.assertEqual(report["error"], "KeyboardInterrupt: interrupted")
            self.assertEqual(report["host"], host)
            manifest = json.loads(
                (evidence / SERVER_ALIAS.EVIDENCE_MANIFEST_NAME).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                [entry["path"] for entry in manifest["artifacts"]],
                ["server-alias-parity.json"],
            )

    def test_termination_signal_is_converted_to_cleanup_exception(self) -> None:
        with self.assertRaisesRegex(InterruptedError, "received signal"):
            SERVER_ALIAS.termination_signal_handler(SERVER_ALIAS.signal.SIGTERM, None)

    def test_interrupt_handlers_include_sigint(self) -> None:
        with mock.patch.object(SERVER_ALIAS.signal, "signal") as register:
            SERVER_ALIAS.install_termination_signal_handlers()
        register.assert_any_call(
            SERVER_ALIAS.signal.SIGINT, SERVER_ALIAS.termination_signal_handler
        )
        register.assert_any_call(
            SERVER_ALIAS.signal.SIGTERM, SERVER_ALIAS.termination_signal_handler
        )


if __name__ == "__main__":
    unittest.main()
