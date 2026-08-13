#!/usr/bin/env python3
"""Model-backed Hebrus/DS4 server route, output, and shutdown parity gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import platform
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL_CONTRACT = ROOT / "docs" / "contracts" / "qwen-release.json"
DEFAULT_LOCK_FILE = "/tmp/ds4.lock"
CANONICAL_SERVER_NAME = "hebrus-server"
LEGACY_SERVER_NAME = "ds4-server"
EXPECTED_COMPLETION_MARKER = "HEBRUS ALIAS PARITY OK"
EVIDENCE_MANIFEST_NAME = "server-alias-evidence-manifest.json"
VOLATILE_COMPLETION_KEYS = frozenset({"id", "created"})
LOCAL_HTTP_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


@dataclass(frozen=True)
class PublishedArtifact:
    filename: str
    size: int
    sha256: str
    revision: str
    runtime_commit: str


@dataclass(frozen=True)
class FileSignature:
    device: int
    inode: int
    size: int
    mtime_ns: int


def termination_signal_handler(signum: int, _frame: Any) -> None:
    raise InterruptedError(f"received signal {signum}")


def install_termination_signal_handlers() -> None:
    signal.signal(signal.SIGINT, termination_signal_handler)
    signal.signal(signal.SIGTERM, termination_signal_handler)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, termination_signal_handler)


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_signature(path: pathlib.Path) -> FileSignature:
    stat = path.stat()
    return FileSignature(stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def load_published_artifact(path: pathlib.Path = MODEL_CONTRACT) -> PublishedArtifact:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AssertionError(f"cannot read Qwen release contract {path}: {exc}") from exc
    if (
        not isinstance(document, dict)
        or type(document.get("schemaVersion")) is not int
        or document.get("schemaVersion") != 2
    ):
        raise AssertionError("Qwen release contract must use schemaVersion 2")
    published = document.get("publishedArtifact")
    if not isinstance(published, dict) or published.get("status") != "published":
        raise AssertionError("Qwen release contract has no published artifact")

    required = ("filename", "bytes", "sha256", "revision", "runtimeCommit")
    missing = [key for key in required if key not in published]
    if missing:
        raise AssertionError(
            "Qwen published artifact is missing: " + ", ".join(missing)
        )
    filename = published["filename"]
    size = published["bytes"]
    digest = published["sha256"]
    revision = published["revision"]
    runtime_commit = published["runtimeCommit"]
    if not isinstance(filename, str) or not filename:
        raise AssertionError("Qwen published filename is invalid")
    if type(size) is not int or size <= 0:
        raise AssertionError("Qwen published byte count is invalid")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise AssertionError("Qwen published SHA-256 is invalid")
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise AssertionError("Qwen published revision is invalid")
    if (
        not isinstance(runtime_commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", runtime_commit) is None
    ):
        raise AssertionError("Qwen compatible runtime commit is invalid")
    return PublishedArtifact(filename, size, digest, revision, runtime_commit)


def verify_model(
    requested_path: pathlib.Path, artifact: PublishedArtifact
) -> tuple[pathlib.Path, str, FileSignature]:
    expanded = requested_path.expanduser()
    if expanded.is_symlink():
        raise AssertionError("release model path must not be a symlink")
    model = expanded.resolve(strict=True)
    if not model.is_file():
        raise AssertionError(f"release model is not a regular file: {model}")
    if model.name != artifact.filename:
        raise AssertionError(
            f"unexpected Qwen filename: {model.name!r}; expected {artifact.filename!r}"
        )
    before = file_signature(model)
    if before.size != artifact.size:
        raise AssertionError(f"unexpected Qwen size: {before.size}")
    digest = sha256(model)
    after = file_signature(model)
    if before != after:
        raise AssertionError("Qwen artifact changed while its SHA-256 was computed")
    if digest != artifact.sha256:
        raise AssertionError(f"unexpected Qwen SHA-256: {digest}")
    return model, digest, after


def normalize_completion(document: dict[str, Any]) -> dict[str, Any]:
    """Drop only request-instance fields; preserve every nested/public id."""
    return {
        key: value for key, value in document.items() if key not in VOLATILE_COMPLETION_KEYS
    }


def validate_completion_content(
    document: dict[str, Any], binary_name: str
) -> str:
    choices = document.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise AssertionError(
            f"{binary_name}: completion must contain exactly one choice"
        )
    choice = choices[0]
    if not isinstance(choice, dict):
        raise AssertionError(f"{binary_name}: completion choice is not an object")
    message = choice.get("message")
    if not isinstance(message, dict):
        raise AssertionError(f"{binary_name}: completion choice has no message object")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise AssertionError(f"{binary_name}: completion content is empty or missing")
    normalized = content.strip()
    if normalized != EXPECTED_COMPLETION_MARKER:
        raise AssertionError(
            f"{binary_name}: completion content does not exactly match "
            f"{EXPECTED_COMPLETION_MARKER!r}"
        )
    return normalized


def command_output(argv: Sequence[str]) -> str | None:
    try:
        result = subprocess.run(
            list(argv),
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    output = result.stdout.strip()
    return output or None


def parse_battery_status(output: str | None) -> dict[str, Any]:
    status: dict[str, Any] = {
        "source": None,
        "battery_percent": None,
        "battery_state": None,
    }
    if not output:
        return status
    source_match = re.search(r"Now drawing from '([^']+)'", output)
    if source_match is not None:
        status["source"] = source_match.group(1)
    percent_match = re.search(r"\b(\d{1,3})%;\s*([^;\n]+)", output)
    if percent_match is not None:
        percent = int(percent_match.group(1))
        if 0 <= percent <= 100:
            status["battery_percent"] = percent
        state = percent_match.group(2).strip()
        status["battery_state"] = state or None
    return status


def parse_powermode(output: str | None, power_source: object) -> int | None:
    if not output:
        return None
    modes: dict[str, int] = {}
    current_source: str | None = None
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.endswith("Power:"):
            current_source = stripped[:-1]
            continue
        match = re.fullmatch(r"powermode\s+(\d+)", stripped)
        if match is not None:
            modes[current_source or ""] = int(match.group(1))
    if isinstance(power_source, str) and power_source in modes:
        return modes[power_source]
    unique_modes = set(modes.values())
    if len(unique_modes) == 1:
        return unique_modes.pop()
    return None


def parse_positive_integer(value: str | None) -> int | None:
    if value is None or re.fullmatch(r"[1-9][0-9]*", value) is None:
        return None
    return int(value)


def collect_host_metadata(
    reader: Callable[[Sequence[str]], str | None] = command_output,
    *,
    system_name: str | None = None,
    architecture: str | None = None,
) -> dict[str, Any]:
    system = system_name if system_name is not None else platform.system()
    machine = architecture if architecture is not None else platform.machine()
    battery = parse_battery_status(reader(("pmset", "-g", "batt")))
    power_source = battery["source"]
    battery["powermode"] = parse_powermode(
        reader(("pmset", "-g", "custom")), power_source
    )
    return {
        "architecture": machine,
        "hardware_model": reader(("sysctl", "-n", "hw.model")),
        "chip": reader(("sysctl", "-n", "machdep.cpu.brand_string")),
        "physical_memory_bytes": parse_positive_integer(
            reader(("sysctl", "-n", "hw.memsize"))
        ),
        "os": {
            "name": "macOS" if system == "Darwin" else system,
            "version": reader(("sw_vers", "-productVersion")),
            "build": reader(("sw_vers", "-buildVersion")),
        },
        "power": battery,
    }


def validate_qualified_host(host: Mapping[str, Any], expected_backend: str) -> None:
    operating_system = host.get("os")
    if expected_backend != "metal":
        raise AssertionError("model-backed server parity requires the Metal backend")
    if host.get("architecture") != "arm64":
        raise AssertionError("model-backed server parity requires an arm64 host")
    if not isinstance(operating_system, Mapping) or operating_system.get("name") != "macOS":
        raise AssertionError("model-backed server parity requires macOS")
    if not operating_system.get("version") or not operating_system.get("build"):
        raise AssertionError("qualified host OS version/build metadata is unavailable")
    memory = host.get("physical_memory_bytes")
    if type(memory) is not int or memory <= 0:
        raise AssertionError("qualified host physical memory metadata is unavailable")


def build_environment(source: Mapping[str, str]) -> tuple[dict[str, str], list[str]]:
    environment: dict[str, str] = {}
    removed: list[str] = []
    for key, value in source.items():
        if key == "DS4_LOCK_FILE":
            environment[key] = value
        elif key.startswith(("DS4_", "HEBRUS_")):
            removed.append(key)
        else:
            environment[key] = value
    if not environment.get("DS4_LOCK_FILE"):
        environment["DS4_LOCK_FILE"] = DEFAULT_LOCK_FILE
    return environment, sorted(removed)


def prepare_evidence_dir(requested: pathlib.Path | None) -> pathlib.Path:
    if requested is None:
        return pathlib.Path(tempfile.mkdtemp(prefix="hebrus-server-alias-")).resolve()
    evidence_dir = requested.expanduser().resolve()
    evidence_dir.mkdir(parents=True, exist_ok=False)
    return evidence_dir


def write_report(path: pathlib.Path, report: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def build_evidence_manifest(evidence_dir: pathlib.Path) -> dict[str, Any]:
    artifacts: list[dict[str, Any]] = []
    for path in sorted(evidence_dir.iterdir(), key=lambda candidate: candidate.name):
        if (
            not path.is_file()
            or path.name == EVIDENCE_MANIFEST_NAME
            or path.name.startswith(".")
        ):
            continue
        artifacts.append(
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    core: dict[str, Any] = {
        "schema_version": 1,
        "artifacts": artifacts,
    }
    canonical = json.dumps(
        core, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return {
        **core,
        "bundle_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def write_evidence_manifest(evidence_dir: pathlib.Path) -> pathlib.Path:
    manifest_path = evidence_dir / EVIDENCE_MANIFEST_NAME
    write_report(manifest_path, build_evidence_manifest(evidence_dir))
    return manifest_path


def validate_requested_port(port: int) -> None:
    if port != 0 and (port < 1024 or port > 65535):
        raise AssertionError("--port must be 0 or an unprivileged TCP port")


def assert_port_available(port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError as exc:
            raise AssertionError(f"TCP port {port} is already in use") from exc


def select_port(requested: int) -> int:
    validate_requested_port(requested)
    if requested:
        assert_port_available(requested)
        return requested
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        selected = int(probe.getsockname()[1])
    if selected < 1024:
        raise AssertionError(f"OS selected a privileged TCP port: {selected}")
    return selected


def wait_for_port_closed(port: int, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while True:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                pass
        except OSError:
            return
        if time.monotonic() >= deadline:
            raise AssertionError(f"TCP port {port} still accepts connections after shutdown")
        time.sleep(0.1)


def http_json(
    url: str,
    payload: dict[str, Any] | None = None,
    *,
    timeout: float,
) -> dict[str, Any]:
    data = None
    method = "GET"
    headers: dict[str, str] = {}
    if payload is not None:
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        method = "POST"
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with LOCAL_HTTP_OPENER.open(request, timeout=timeout) as response:
        if response.status != 200:
            raise AssertionError(f"{method} {url} returned HTTP {response.status}")
        document = json.loads(response.read())
    if not isinstance(document, dict):
        raise AssertionError(f"{method} {url} did not return a JSON object")
    return document


def build_sha_matches(actual: object, expected: str) -> bool:
    if not isinstance(actual, str):
        return False
    if re.fullmatch(r"[0-9a-f]{12,40}", actual) is None:
        return False
    if re.fullmatch(r"[0-9a-f]{12,40}", expected) is None:
        return False
    return actual.startswith(expected) or expected.startswith(actual)


def repository_head(root: pathlib.Path = ROOT) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    head = result.stdout.strip()
    if result.returncode != 0 or re.fullmatch(r"[0-9a-f]{40}", head) is None:
        raise AssertionError(f"cannot resolve candidate repository HEAD: {result.stderr.strip()}")
    return head


def assert_clean_repository(root: pathlib.Path = ROOT) -> None:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"cannot inspect candidate repository state: {result.stderr.strip()}"
        )
    if result.stdout:
        raise AssertionError(
            "candidate repository must be an exact clean committed tree"
        )


def validate_capability_document(
    document: dict[str, Any],
    *,
    binary_name: str,
    engine_id: str,
    expected_backend: str,
    expected_build_sha: str,
) -> None:
    if type(document.get("schema_version")) is not int or document.get("schema_version") != 1:
        raise AssertionError(f"{binary_name}: capability schema is not 1")
    if document.get("engine_id") != engine_id:
        raise AssertionError(f"{binary_name}: expected engine_id {engine_id!r}")
    if document.get("executable_role") != "server":
        raise AssertionError(f"{binary_name}: capability role is not server")
    if document.get("backend") != expected_backend:
        raise AssertionError(
            f"{binary_name}: expected backend {expected_backend!r}, "
            f"got {document.get('backend')!r}"
        )
    if not build_sha_matches(document.get("build_git_sha"), expected_build_sha):
        raise AssertionError(
            f"{binary_name}: build {document.get('build_git_sha')!r} does not "
            f"match candidate {expected_build_sha!r}"
        )


def read_capabilities(
    binary: pathlib.Path, environment: Mapping[str, str]
) -> dict[str, Any]:
    result = subprocess.run(
        [str(binary), "--capabilities=json"],
        check=False,
        capture_output=True,
        text=True,
        env=dict(environment),
        timeout=30,
    )
    if result.returncode != 0 or result.stderr:
        raise AssertionError(
            f"{binary.name}: capability preflight failed with {result.returncode}: "
            f"{result.stderr.strip()}"
        )
    document = json.loads(result.stdout)
    if not isinstance(document, dict):
        raise AssertionError(f"{binary.name}: capability output is not a JSON object")
    return document


def validate_capability_pair_documents(
    canonical_document: dict[str, Any],
    legacy_document: dict[str, Any],
    *,
    canonical_name: str,
    legacy_name: str,
    expected_backend: str,
    expected_build_sha: str,
) -> None:
    validate_capability_document(
        canonical_document,
        binary_name=canonical_name,
        engine_id="hebrus",
        expected_backend=expected_backend,
        expected_build_sha=expected_build_sha,
    )
    validate_capability_document(
        legacy_document,
        binary_name=legacy_name,
        engine_id="ds4",
        expected_backend=expected_backend,
        expected_build_sha=expected_build_sha,
    )
    canonical_common = {
        key: value for key, value in canonical_document.items() if key != "engine_id"
    }
    legacy_common = {
        key: value for key, value in legacy_document.items() if key != "engine_id"
    }
    if canonical_common != legacy_common:
        raise AssertionError("server capability documents differ beyond engine_id")


def capability_preflight(
    canonical: pathlib.Path,
    legacy: pathlib.Path,
    *,
    expected_backend: str,
    expected_build_sha: str,
    environment: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
    canonical_document = read_capabilities(canonical, environment)
    legacy_document = read_capabilities(legacy, environment)
    validate_capability_pair_documents(
        canonical_document,
        legacy_document,
        canonical_name=canonical.name,
        legacy_name=legacy.name,
        expected_backend=expected_backend,
        expected_build_sha=expected_build_sha,
    )
    return {
        canonical.name: canonical_document,
        legacy.name: legacy_document,
    }


def wait_for_listener(
    log_path: pathlib.Path,
    process: subprocess.Popen[bytes],
    port: int,
    deadline: float,
) -> None:
    marker = f"listening on http://127.0.0.1:{port}"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise AssertionError(
                f"server exited during startup with code {process.returncode}"
            )
        try:
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
        except FileNotFoundError:
            log_text = ""
        if marker in log_text:
            return
        time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
    raise AssertionError(f"server startup timed out before listener ownership: {marker}")


def wait_for_models(
    base_url: str,
    process: subprocess.Popen[bytes],
    log_path: pathlib.Path,
    port: int,
    timeout: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    wait_for_listener(log_path, process, port, deadline)
    last_error = "server did not accept a request"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise AssertionError(
                f"server exited during startup with code {process.returncode}"
            )
        remaining = deadline - time.monotonic()
        try:
            return http_json(
                f"{base_url}/v1/models",
                timeout=max(0.05, min(2.0, remaining)),
            )
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            last_error = str(exc)
            time.sleep(min(0.25, max(0.0, deadline - time.monotonic())))
    raise AssertionError(f"server startup timed out: {last_error}")


def terminate_process(
    process: subprocess.Popen[bytes], binary_name: str, timeout: float
) -> int:
    if process.poll() is None:
        process.send_signal(signal.SIGTERM)
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            process.kill()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired as kill_exc:
                raise AssertionError(
                    f"{binary_name}: process survived SIGKILL"
                ) from kill_exc
            raise AssertionError(f"{binary_name}: graceful shutdown timed out") from exc
    if process.returncode is None:
        raise AssertionError(f"{binary_name}: process has no exit status")
    return process.returncode


def run_server(
    binary: pathlib.Path,
    model: pathlib.Path,
    port: int,
    startup_timeout: float,
    request_timeout: float,
    shutdown_timeout: float,
    evidence_dir: pathlib.Path,
    environment: Mapping[str, str],
) -> dict[str, Any]:
    assert_port_available(port)
    log_path = evidence_dir / f"{binary.name}.log"
    command = [
        str(binary),
        "-m", str(model),
        "--ctx", "8192",
        "--tokens", "32",
        "--threads", "8",
        "--host", "127.0.0.1",
        "--port", str(port),
    ]
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=dict(environment),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            base_url = f"http://127.0.0.1:{port}"
            models = wait_for_models(
                base_url, process, log_path, port, startup_timeout
            )
            entries = models.get("data")
            if not isinstance(entries, list) or len(entries) != 1:
                raise AssertionError(f"{binary.name}: unexpected model list: {models}")
            entry = entries[0]
            if not isinstance(entry, dict):
                raise AssertionError(f"{binary.name}: model list entry is not an object")
            model_id = entry.get("id")
            if not isinstance(model_id, str) or not model_id:
                raise AssertionError(f"{binary.name}: model list has no usable id")
            model_document = http_json(
                f"{base_url}/v1/models/{urllib.parse.quote(model_id, safe='')}",
                timeout=request_timeout,
            )
            completion = http_json(
                f"{base_url}/v1/chat/completions",
                {
                    "model": model_id,
                    "messages": [
                        {
                            "role": "user",
                            "content": "Reply with exactly: HEBRUS ALIAS PARITY OK",
                        }
                    ],
                    "max_tokens": 16,
                    "temperature": 0,
                    "seed": 20260721,
                    "thinking": False,
                    "stream": False,
                },
                timeout=request_timeout,
            )
            completion_content = validate_completion_content(
                completion, binary.name
            )
        finally:
            exit_code = terminate_process(process, binary.name, shutdown_timeout)
    wait_for_port_closed(port)
    if exit_code != 0:
        raise AssertionError(
            f"{binary.name}: shutdown returned {exit_code}; see {log_path}"
        )

    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    for marker in (
        f"listening on http://127.0.0.1:{port}",
        "shutdown requested, draining requests",
    ):
        if marker not in log_text:
            raise AssertionError(f"{binary.name}: missing log marker {marker!r}")
    return {
        "command": command,
        "models": models,
        "model": model_document,
        "completion": normalize_completion(completion),
        "completion_content": completion_content,
        "log": str(log_path),
        "exit_code": exit_code,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=pathlib.Path)
    parser.add_argument("--bin-dir", default=ROOT, type=pathlib.Path)
    parser.add_argument("--expected-backend", required=True, choices=("metal", "cpu"))
    parser.add_argument("--expected-build-sha", required=True)
    parser.add_argument(
        "--port",
        default=0,
        type=int,
        help="unprivileged port, or 0 to select an unused local port",
    )
    parser.add_argument("--startup-timeout", default=180.0, type=float)
    parser.add_argument("--request-timeout", default=300.0, type=float)
    parser.add_argument("--shutdown-timeout", default=90.0, type=float)
    parser.add_argument("--evidence-dir", type=pathlib.Path)
    args = parser.parse_args()
    for name in ("startup_timeout", "request_timeout", "shutdown_timeout"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if sys.platform == "darwin" and args.expected_backend != "metal":
        parser.error("model-backed server parity on macOS requires the Metal backend")
    return args


def main() -> int:
    args = parse_args()
    evidence_dir = prepare_evidence_dir(args.evidence_dir)
    report_path = evidence_dir / "server-alias-parity.json"
    report: dict[str, Any] = {
        "schema_version": 3,
        "status": "RUNNING",
        "evidence_manifest": EVIDENCE_MANIFEST_NAME,
        "results": {},
    }
    write_report(report_path, report)
    print(f"server-alias-model: evidence {report_path}", file=sys.stderr)
    install_termination_signal_handlers()

    try:
        host = collect_host_metadata()
        report["host"] = host
        write_report(report_path, report)
        validate_qualified_host(host, args.expected_backend)

        bin_dir = args.bin_dir.expanduser().resolve(strict=True)
        canonical = bin_dir / CANONICAL_SERVER_NAME
        legacy = bin_dir / LEGACY_SERVER_NAME
        if (
            not canonical.is_file()
            or not legacy.is_file()
            or not os.path.samefile(canonical, legacy)
        ):
            raise AssertionError(
                f"{CANONICAL_SERVER_NAME} and {LEGACY_SERVER_NAME} "
                "are not one executable surface"
            )

        candidate_head = repository_head()
        assert_clean_repository()
        if not build_sha_matches(candidate_head, args.expected_build_sha):
            raise AssertionError(
                f"candidate repository HEAD {candidate_head!r} does not match "
                f"expected build {args.expected_build_sha!r}"
            )
        environment, removed_environment = build_environment(os.environ)
        binary_hash = sha256(canonical)
        report.update(
            {
                "candidate": {
                    "expected_backend": args.expected_backend,
                    "expected_build_sha": args.expected_build_sha,
                    "repository_head": candidate_head,
                    "repository_clean": True,
                    "binary_sha256": binary_hash,
                },
                "environment": {
                    "preserved_names": ["DS4_LOCK_FILE"],
                    "removed_names": removed_environment,
                },
                "log_paths": {
                    canonical.name: str(evidence_dir / f"{canonical.name}.log"),
                    legacy.name: str(evidence_dir / f"{legacy.name}.log"),
                },
            }
        )
        write_report(report_path, report)
        capabilities = capability_preflight(
            canonical,
            legacy,
            expected_backend=args.expected_backend,
            expected_build_sha=args.expected_build_sha,
            environment=environment,
        )
        if sha256(canonical) != binary_hash or not os.path.samefile(canonical, legacy):
            raise AssertionError("server executable changed during capability preflight")
        report["candidate"]["capabilities"] = capabilities
        write_report(report_path, report)

        artifact = load_published_artifact()
        model, model_hash, model_signature = verify_model(args.model, artifact)
        port = select_port(args.port)
        report.update(
            {
                "model": {
                    "contract": str(MODEL_CONTRACT),
                    "filename": artifact.filename,
                    "path": str(model),
                    "bytes": model_signature.size,
                    "sha256": model_hash,
                    "revision": artifact.revision,
                    "runtime_commit": artifact.runtime_commit,
                },
                "port": port,
            }
        )
        write_report(report_path, report)

        results: dict[str, Any] = report["results"]
        for binary in (canonical, legacy):
            if sha256(canonical) != binary_hash or not os.path.samefile(canonical, legacy):
                raise AssertionError("server executable changed during alias parity gate")
            results[binary.name] = run_server(
                binary,
                model,
                port,
                args.startup_timeout,
                args.request_timeout,
                args.shutdown_timeout,
                evidence_dir,
                environment,
            )
            if sha256(canonical) != binary_hash:
                raise AssertionError("server executable changed during alias parity gate")
            if file_signature(model) != model_signature:
                raise AssertionError("Qwen artifact changed during alias parity gate")
            write_report(report_path, report)

        for field in (
            "models",
            "model",
            "completion",
            "completion_content",
            "exit_code",
        ):
            if results[canonical.name][field] != results[legacy.name][field]:
                raise AssertionError(
                    f"server aliases differ for {field}; see {report_path}"
                )
        if repository_head() != candidate_head:
            raise AssertionError("candidate repository HEAD changed during alias parity gate")
        assert_clean_repository()

        report["status"] = "PASS"
        write_report(report_path, report)
        manifest_path = write_evidence_manifest(evidence_dir)
        print(
            f"server-alias-model: PASS "
            f"({canonical.name}/{legacy.name}, {report_path}, {manifest_path})"
        )
        return 0
    except BaseException as exc:
        report["status"] = "FAIL"
        detail = str(exc) or "interrupted"
        report["error"] = f"{type(exc).__name__}: {detail}"
        write_report(report_path, report)
        try:
            write_evidence_manifest(evidence_dir)
        except Exception as manifest_exc:
            print(
                f"server-alias-model: failed to write evidence manifest: "
                f"{manifest_exc}",
                file=sys.stderr,
            )
        print(f"server-alias-model: evidence retained at {report_path}", file=sys.stderr)
        raise


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt as exc:
        print(f"server-alias-model: FAIL: {exc or 'interrupted'}", file=sys.stderr)
        raise SystemExit(130)
    except (
        AssertionError,
        OSError,
        subprocess.SubprocessError,
        urllib.error.URLError,
        json.JSONDecodeError,
    ) as exc:
        print(f"server-alias-model: FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
