import argparse
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

IDENTITY_KEYS = [
    "schema_version",
    "semantic_version",
    "source_sha",
    "source_state",
]
FULL_SHA_RE = re.compile(r"[0-9a-f]{40}\Z")


def run(command, cwd, *, env=None):
    result = subprocess.run(
        [str(part) for part in command],
        cwd=cwd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"command failed with exit {result.returncode}: {command}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def canonical_identity(version, sha, state):
    return json.dumps(
        {
            "schema_version": 1,
            "semantic_version": version,
            "source_sha": sha,
            "source_state": state,
        },
        separators=(",", ":"),
    ) + "\n"


def parse_identity(raw_identity):
    try:
        identity = json.loads(raw_identity)
    except json.JSONDecodeError as error:
        raise AssertionError(f"invalid build-identity JSON: {raw_identity!r}") from error
    if list(identity) != IDENTITY_KEYS:
        raise AssertionError(
            f"expected ordered identity keys {IDENTITY_KEYS}, got {list(identity)}"
        )
    if identity["schema_version"] != 1:
        raise AssertionError(f"unsupported identity schema: {identity!r}")
    if identity["source_state"] not in {"clean", "dirty", "unavailable"}:
        raise AssertionError(f"invalid identity source state: {identity!r}")
    if identity["source_state"] == "unavailable":
        if identity["source_sha"] is not None:
            raise AssertionError(f"unavailable identity claimed a SHA: {identity!r}")
    elif not isinstance(identity["source_sha"], str) or not FULL_SHA_RE.fullmatch(
        identity["source_sha"]
    ):
        raise AssertionError(f"identity did not contain a full lowercase SHA: {identity!r}")
    expected_raw = canonical_identity(
        identity["semantic_version"],
        identity["source_sha"],
        identity["source_state"],
    )
    if raw_identity != expected_raw:
        raise AssertionError(
            f"identity is valid JSON but not canonical:\n{raw_identity!r}\n"
            f"expected:\n{expected_raw!r}"
        )
    return identity


def assert_receipt(path, *, version, sha, state):
    raw_identity = path.read_text(encoding="utf-8")
    identity = parse_identity(raw_identity)
    expected = {
        "schema_version": 1,
        "semantic_version": version,
        "source_sha": sha,
        "source_state": state,
    }
    if identity != expected:
        raise AssertionError(f"expected receipt {expected!r}, got {identity!r}")
    return raw_identity


def configure_fixture(cmake, source, build, module, git=None):
    command = [
        cmake,
        "-S",
        source,
        "-B",
        build,
        f"-DDEAC_BUILD_IDENTITY_MODULE:FILEPATH={module}",
    ]
    if git is not None:
        command.append(f"-DGIT_EXECUTABLE:FILEPATH={git}")
    run(command, source)


def write_fixture(source):
    (source / "src").mkdir(parents=True)
    (source / "VERSION").write_text("2.0.0-rc1\n", encoding="utf-8")
    (source / "src" / "probe.txt").write_text("baseline\n", encoding="utf-8")
    (source / "CMakeLists.txt").write_text(
        """cmake_minimum_required(VERSION 3.18)
project(deac_build_identity_fixture LANGUAGES NONE)
if(NOT DEFINED DEAC_BUILD_IDENTITY_MODULE)
    message(FATAL_ERROR "DEAC_BUILD_IDENTITY_MODULE is required")
endif()
include("${DEAC_BUILD_IDENTITY_MODULE}")
deac_configure_build_identity(
    SOURCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_HEADER "${CMAKE_CURRENT_BINARY_DIR}/identity.hpp"
    OUTPUT_RECEIPT "${CMAKE_CURRENT_BINARY_DIR}/identity.json")
""",
        encoding="utf-8",
    )


def git_environment():
    environment = os.environ.copy()
    environment.update(
        {
            "GIT_AUTHOR_DATE": "2000-01-01T00:00:00+0000",
            "GIT_COMMITTER_DATE": "2000-01-01T00:00:00+0000",
            "GIT_CONFIG_NOSYSTEM": "1",
        }
    )
    return environment


def test_git_reconfiguration(cmake, git, workdir, module):
    source = workdir / "git-source"
    build = workdir / "git-build"
    write_fixture(source)
    environment = git_environment()
    run([git, "init", "--quiet"], source, env=environment)
    run([git, "config", "user.name", "DEAC identity test"], source, env=environment)
    run(
        [git, "config", "user.email", "identity-test@example.invalid"],
        source,
        env=environment,
    )
    run([git, "add", "CMakeLists.txt", "VERSION", "src/probe.txt"], source, env=environment)
    run([git, "commit", "--quiet", "-m", "fixture baseline"], source, env=environment)
    initial_sha = run([git, "rev-parse", "HEAD"], source, env=environment).stdout.strip()

    configure_fixture(cmake, source, build, module, git)
    receipt = build / "identity.json"
    first_raw = assert_receipt(
        receipt, version="2.0.0-rc1", sha=initial_sha, state="clean"
    )
    run([cmake, "--build", build], source)
    if receipt.read_text(encoding="utf-8") != first_raw:
        raise AssertionError("unchanged rebuild changed the canonical identity")

    # CMake's dependency checks are timestamp based.  Keep this regression
    # deterministic on filesystems whose mtimes have only one-second
    # resolution, where an immediate edit could otherwise look unchanged.
    time.sleep(1.05)
    probe = source / "src" / "probe.txt"
    probe.write_text("dirty\n", encoding="utf-8")
    run([cmake, "--build", build], source)
    assert_receipt(receipt, version="2.0.0-rc1", sha=initial_sha, state="dirty")

    run([git, "add", "src/probe.txt"], source, env=environment)
    run([cmake, "--build", build], source)
    assert_receipt(receipt, version="2.0.0-rc1", sha=initial_sha, state="dirty")

    time.sleep(1.05)
    probe.write_text("baseline\n", encoding="utf-8")
    run([git, "add", "src/probe.txt"], source, env=environment)
    run([cmake, "--build", build], source)
    assert_receipt(receipt, version="2.0.0-rc1", sha=initial_sha, state="clean")

    time.sleep(1.05)
    probe.write_text("committed successor\n", encoding="utf-8")
    run([git, "add", "src/probe.txt"], source, env=environment)
    run([git, "commit", "--quiet", "-m", "fixture successor"], source, env=environment)
    successor_sha = run([git, "rev-parse", "HEAD"], source, env=environment).stdout.strip()
    if successor_sha == initial_sha:
        raise AssertionError("fixture commit did not update HEAD")
    run([cmake, "--build", build], source)
    assert_receipt(
        receipt, version="2.0.0-rc1", sha=successor_sha, state="clean"
    )

    # An empty commit changes only Git metadata.  The normal build must still
    # regenerate and embed its new exact HEAD without a manual configure.
    time.sleep(1.05)
    run(
        [git, "commit", "--quiet", "--allow-empty", "-m", "fixture empty successor"],
        source,
        env=environment,
    )
    metadata_only_sha = run(
        [git, "rev-parse", "HEAD"], source, env=environment
    ).stdout.strip()
    run([cmake, "--build", build], source)
    assert_receipt(
        receipt, version="2.0.0-rc1", sha=metadata_only_sha, state="clean"
    )

    return source


def test_archive_fallback(cmake, git_source, workdir, module, git=None):
    archive_source = workdir / "archive-source"
    archive_build = workdir / "archive-build"
    shutil.copytree(git_source, archive_source, ignore=shutil.ignore_patterns(".git"))
    if (archive_source / ".git").exists():
        raise AssertionError("archive fixture unexpectedly retained Git metadata")
    configure_fixture(cmake, archive_source, archive_build, module, git)
    assert_receipt(
        archive_build / "identity.json",
        version="2.0.0-rc1",
        sha=None,
        state="unavailable",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--identity-module", required=True)
    parser.add_argument("--cmake", required=True)
    parser.add_argument("--git")
    parser.add_argument("--expected-version", required=True)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument(
        "--expected-state", choices=("clean", "dirty", "unavailable"), required=True
    )
    args = parser.parse_args()

    expected_sha = None if args.expected_sha == "null" else args.expected_sha
    expected_raw = canonical_identity(
        args.expected_version, expected_sha, args.expected_state
    )

    first = run([args.exe, "--build-identity"], Path(args.workdir).parent)
    second = run([args.exe, "--build-identity"], Path(args.workdir).parent)
    if first.stderr or second.stderr:
        raise AssertionError(
            "--build-identity wrote to stderr:\n"
            f"first: {first.stderr!r}\nsecond: {second.stderr!r}"
        )
    if first.stdout != expected_raw or second.stdout != expected_raw:
        raise AssertionError(
            "executable identity was not byte-stable or did not match CMake:\n"
            f"expected: {expected_raw!r}\n"
            f"first: {first.stdout!r}\nsecond: {second.stdout!r}"
        )
    parse_identity(first.stdout)
    receipt_raw = assert_receipt(
        Path(args.receipt),
        version=args.expected_version,
        sha=expected_sha,
        state=args.expected_state,
    )
    if receipt_raw != first.stdout:
        raise AssertionError("build receipt and executable identity differ byte-for-byte")

    workdir = Path(args.workdir)
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    module = Path(args.identity_module).resolve()
    cmake = Path(args.cmake).resolve()
    if args.git is not None:
        git = Path(args.git).resolve()
        git_source = test_git_reconfiguration(cmake, git, workdir, module)
    else:
        git_source = workdir / "archive-template"
        write_fixture(git_source)
        git = None
    test_archive_fallback(cmake, git_source, workdir, module, git)


if __name__ == "__main__":
    main()
