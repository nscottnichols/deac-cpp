import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

IDENTITY_KEYS = [
    "schema_version",
    "semantic_version",
    "source_sha",
    "source_state",
]
FULL_SHA_RE = re.compile(r"[0-9a-f]{40}\Z")
GIT_REDIRECTION_ENVIRONMENT = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_COMMON_DIR",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_QUARANTINE_PATH",
    "GIT_NAMESPACE",
    "GIT_SHALLOW_FILE",
    "GIT_CEILING_DIRECTORIES",
    "GIT_DISCOVERY_ACROSS_FILESYSTEM",
    "GIT_PREFIX",
    "GIT_EXEC_PATH",
    "GIT_CONFIG_COUNT",
    "GIT_CONFIG_PARAMETERS",
    "GIT_CONFIG_GLOBAL",
    "GIT_CONFIG_SYSTEM",
    "GIT_CONFIG_NOSYSTEM",
    "GIT_REPLACE_REF_BASE",
    "GIT_LITERAL_PATHSPECS",
    "GIT_GLOB_PATHSPECS",
    "GIT_NOGLOB_PATHSPECS",
    "GIT_ICASE_PATHSPECS",
)


def sanitized_environment(environment=None):
    sanitized = (os.environ if environment is None else environment).copy()
    for variable in GIT_REDIRECTION_ENVIRONMENT:
        sanitized.pop(variable, None)
    return sanitized


def run(command, cwd, *, env=None, check=True):
    result = subprocess.run(
        [str(part) for part in command],
        cwd=cwd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
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


def repository_identity(source, git):
    version = (source / "VERSION").read_text(encoding="utf-8").strip()
    if git is None:
        return version, None, "unavailable"
    environment = sanitized_environment()
    top_level = run(
        [git, "-C", source, "rev-parse", "--show-toplevel"],
        source,
        env=environment,
        check=False,
    )
    if top_level.returncode != 0 or Path(top_level.stdout.strip()).resolve() != source:
        return version, None, "unavailable"
    sha_result = run(
        [git, "-C", source, "rev-parse", "--verify", "HEAD^{commit}"],
        source,
        env=environment,
        check=False,
    )
    sha = sha_result.stdout.strip().lower()
    if sha_result.returncode != 0 or not FULL_SHA_RE.fullmatch(sha):
        return version, None, "unavailable"
    status = run(
        [
            git,
            "-C",
            source,
            "status",
            "--porcelain=v1",
            "--untracked-files=normal",
            "--",
            "VERSION",
            "src",
        ],
        source,
        env=environment,
        check=False,
    )
    if status.returncode != 0:
        return version, None, "unavailable"
    return version, sha, "dirty" if status.stdout.strip() else "clean"


def assert_executable_identity(exe, receipt, *, version, sha, state, repeat=False):
    expected_raw = canonical_identity(version, sha, state)
    repetitions = 2 if repeat else 1
    for _ in range(repetitions):
        result = run([exe, "--build-identity"], exe.parent)
        if result.stderr:
            raise AssertionError(f"--build-identity wrote to stderr: {result.stderr!r}")
        if result.stdout != expected_raw:
            raise AssertionError(
                "executable identity did not match expected canonical bytes:\n"
                f"expected: {expected_raw!r}\nactual: {result.stdout!r}"
            )
        parse_identity(result.stdout)
    receipt_raw = receipt.read_text(encoding="utf-8")
    if receipt_raw != expected_raw:
        raise AssertionError(
            "build receipt and executable identity differ byte-for-byte:\n"
            f"expected: {expected_raw!r}\nreceipt: {receipt_raw!r}"
        )


def fixture_probe_source(marker="baseline"):
    return f"""#include \"build_identity.hpp\"

#include <iostream>
#include <string>

// {marker}
int main(int argc, char* argv[]) {{
    if (argc == 2 && std::string(argv[1]) == \"--build-identity\") {{
        std::cout << deac_build_identity::canonical_json() << '\\n';
        return 0;
    }}
    if (argc == 2 && std::string(argv[1]) == \"--version\") {{
        std::cout << deac_build_identity::semantic_version() << '\\n';
        return 0;
    }}
    return 2;
}}
"""


def write_fixture(source):
    (source / "src").mkdir(parents=True)
    (source / "VERSION").write_text("2.0.0-rc1\n", encoding="utf-8")
    (source / ".gitignore").write_text("src/*.ignored\n", encoding="utf-8")
    (source / "src" / "probe.cpp").write_text(
        fixture_probe_source(), encoding="utf-8"
    )
    (source / "CMakeLists.txt").write_text(
        """cmake_minimum_required(VERSION 3.18)
project(deac_build_identity_fixture LANGUAGES CXX)
foreach(required_variable DEAC_BUILD_IDENTITY_MODULE)
    if(NOT DEFINED ${required_variable})
        message(FATAL_ERROR "${required_variable} is required")
    endif()
endforeach()
include("${DEAC_BUILD_IDENTITY_MODULE}")
deac_add_build_identity(
    SOURCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}"
    GENERATED_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated"
    IDENTITY_NAME fixture
    RECEIPT "${CMAKE_CURRENT_BINARY_DIR}/deac-build-identity.json")
add_executable(identity-probe src/probe.cpp)
deac_target_add_build_identity(identity-probe)
""",
        encoding="utf-8",
    )


def configure_fixture(
    cmake,
    source,
    build,
    module,
    cxx_compiler,
    generator,
    make_program,
    git=None,
    env=None,
):
    command = [
        cmake,
        "-S",
        source,
        "-B",
        build,
        "-G",
        generator,
        f"-DCMAKE_CXX_COMPILER:FILEPATH={cxx_compiler}",
        f"-DDEAC_BUILD_IDENTITY_MODULE:FILEPATH={module}",
    ]
    if make_program is not None:
        command.append(f"-DCMAKE_MAKE_PROGRAM:FILEPATH={make_program}")
    if git is not None:
        command.append(f"-DGIT_EXECUTABLE:FILEPATH={git}")
    run(command, source, env=env)


def build_fixture(
    cmake,
    source,
    build,
    *,
    build_config=None,
    env=None,
    require_ref_watch=False,
):
    command = [cmake, "--build", build, "--parallel", "2"]
    if build_config is not None:
        command.extend(["--config", build_config])
    result = run(command, source, env=env)
    output = result.stdout + result.stderr
    if "Refreshing canonical DEAC build identity" not in output:
        raise AssertionError(
            "ordinary build did not execute the symbolic identity refresh:\n" + output
        )
    if "build_identity.cpp" not in output:
        raise AssertionError(
            "ordinary build did not compile the identity-only source:\n" + output
        )
    if "Linking CXX executable" not in output or "identity-probe" not in output:
        raise AssertionError(
            "ordinary build did not link the refreshed identity object:\n" + output
        )
    if (
        require_ref_watch
        and "Configuring done" not in output
        and "GLOB mismatch" not in output
    ):
        raise AssertionError(
            "creation of the formerly absent loose ref was not detected by CMake:\n"
            + output
        )
    return result


def fixture_executable(build, build_config=None):
    suffix = ".exe" if os.name == "nt" else ""
    executable_directory = build if build_config is None else build / build_config
    return executable_directory / f"identity-probe{suffix}"


def git_environment():
    environment = sanitized_environment()
    environment.update(
        {
            "GIT_AUTHOR_DATE": "2000-01-01T00:00:00+0000",
            "GIT_COMMITTER_DATE": "2000-01-01T00:00:00+0000",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_DEFAULT_HASH": "sha1",
            "LC_ALL": "C",
        }
    )
    return environment


def test_git_transitions(
    cmake,
    git,
    workdir,
    module,
    cxx_compiler,
    generator,
    make_program,
    build_config,
):
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
    run(
        [git, "add", ".gitignore", "CMakeLists.txt", "VERSION", "src/probe.cpp"],
        source,
        env=environment,
    )
    run([git, "commit", "--quiet", "-m", "fixture baseline"], source, env=environment)
    initial_sha = run([git, "rev-parse", "HEAD"], source, env=environment).stdout.strip()

    configure_fixture(
        cmake,
        source,
        build,
        module,
        cxx_compiler,
        generator,
        make_program,
        git,
    )
    build_fixture(cmake, source, build, build_config=build_config)
    exe = fixture_executable(build, build_config)
    receipt = build / "deac-build-identity.json"
    assert_executable_identity(
        exe, receipt, version="2.0.0-rc1", sha=initial_sha, state="clean", repeat=True
    )

    ignored_product = source / "src" / "compiler-cache.ignored"
    ignored_product.write_text("ignored build output\n", encoding="utf-8")
    build_fixture(cmake, source, build, build_config=build_config)
    assert_executable_identity(
        exe, receipt, version="2.0.0-rc1", sha=initial_sha, state="clean"
    )
    ignored_product.unlink()

    untracked_source = source / "src" / "untracked.hpp"
    untracked_source.write_text("// untracked source\n", encoding="utf-8")
    build_fixture(cmake, source, build, build_config=build_config)
    assert_executable_identity(
        exe, receipt, version="2.0.0-rc1", sha=initial_sha, state="dirty"
    )
    untracked_source.unlink()
    build_fixture(cmake, source, build, build_config=build_config)
    assert_executable_identity(
        exe, receipt, version="2.0.0-rc1", sha=initial_sha, state="clean"
    )

    probe = source / "src" / "probe.cpp"
    probe.write_text(fixture_probe_source("unstaged dirty"), encoding="utf-8")
    build_fixture(cmake, source, build, build_config=build_config)
    assert_executable_identity(
        exe, receipt, version="2.0.0-rc1", sha=initial_sha, state="dirty"
    )

    run([git, "add", "src/probe.cpp"], source, env=environment)
    build_fixture(cmake, source, build, build_config=build_config)
    assert_executable_identity(
        exe, receipt, version="2.0.0-rc1", sha=initial_sha, state="dirty"
    )

    probe.write_text(fixture_probe_source(), encoding="utf-8")
    run([git, "add", "src/probe.cpp"], source, env=environment)
    build_fixture(cmake, source, build, build_config=build_config)
    assert_executable_identity(
        exe, receipt, version="2.0.0-rc1", sha=initial_sha, state="clean"
    )

    probe.write_text(fixture_probe_source("committed successor"), encoding="utf-8")
    run([git, "add", "src/probe.cpp"], source, env=environment)
    run([git, "commit", "--quiet", "-m", "fixture successor"], source, env=environment)
    successor_sha = run([git, "rev-parse", "HEAD"], source, env=environment).stdout.strip()
    build_fixture(cmake, source, build, build_config=build_config)
    assert_executable_identity(
        exe, receipt, version="2.0.0-rc1", sha=successor_sha, state="clean"
    )

    symbolic_ref = run(
        [git, "symbolic-ref", "--quiet", "HEAD"], source, env=environment
    ).stdout.strip()
    run([git, "pack-refs", "--all", "--prune"], source, env=environment)
    loose_ref_result = run(
        [git, "rev-parse", "--git-path", symbolic_ref], source, env=environment
    )
    loose_ref = Path(loose_ref_result.stdout.strip())
    if not loose_ref.is_absolute():
        loose_ref = source / loose_ref
    if loose_ref.exists():
        raise AssertionError(f"pack-refs did not remove loose ref {loose_ref}")
    build_fixture(cmake, source, build, build_config=build_config)
    assert_executable_identity(
        exe, receipt, version="2.0.0-rc1", sha=successor_sha, state="clean"
    )

    run(
        [git, "commit", "--quiet", "--allow-empty", "-m", "metadata-only successor"],
        source,
        env=environment,
    )
    metadata_only_sha = run(
        [git, "rev-parse", "HEAD"], source, env=environment
    ).stdout.strip()
    if not loose_ref.exists():
        raise AssertionError(f"metadata-only commit did not create loose ref {loose_ref}")
    build_fixture(
        cmake,
        source,
        build,
        build_config=build_config,
        require_ref_watch=True,
    )
    assert_executable_identity(
        exe, receipt, version="2.0.0-rc1", sha=metadata_only_sha, state="clean"
    )

    return source


def test_archive_spoof(
    cmake,
    git,
    git_source,
    workdir,
    module,
    cxx_compiler,
    generator,
    make_program,
    build_config,
):
    archive_source = workdir / "archive-source"
    archive_build = workdir / "archive-build"
    shutil.copytree(git_source, archive_source, ignore=shutil.ignore_patterns(".git"))
    if (archive_source / ".git").exists():
        raise AssertionError("archive fixture unexpectedly retained Git metadata")

    git_dir = git_source / ".git"
    spoofed_environment = os.environ.copy()
    spoofed_environment.update(
        {
            "GIT_DIR": str(git_dir),
            "GIT_WORK_TREE": str(archive_source),
            "GIT_INDEX_FILE": str(git_dir / "index"),
            "GIT_COMMON_DIR": str(git_dir),
            "GIT_OBJECT_DIRECTORY": str(git_dir / "objects"),
            "GIT_ALTERNATE_OBJECT_DIRECTORIES": str(git_dir / "objects"),
        }
    )
    spoof_probe = run(
        [git, "-C", archive_source, "rev-parse", "--show-toplevel"],
        archive_source,
        env=spoofed_environment,
    )
    if Path(spoof_probe.stdout.strip()).resolve() != archive_source:
        raise AssertionError("test environment did not successfully spoof archive Git root")

    configure_fixture(
        cmake,
        archive_source,
        archive_build,
        module,
        cxx_compiler,
        generator,
        make_program,
        git,
        spoofed_environment,
    )
    build_fixture(
        cmake,
        archive_source,
        archive_build,
        build_config=build_config,
        env=spoofed_environment,
    )
    assert_executable_identity(
        fixture_executable(archive_build, build_config),
        archive_build / "deac-build-identity.json",
        version="2.0.0-rc1",
        sha=None,
        state="unavailable",
        repeat=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--identity-module", required=True)
    parser.add_argument("--cmake", required=True)
    parser.add_argument("--cxx-compiler", required=True)
    parser.add_argument("--generator", required=True)
    parser.add_argument("--make-program")
    parser.add_argument("--build-config")
    parser.add_argument("--git")
    args = parser.parse_args()

    source_root = Path(args.source_root).resolve()
    git = Path(args.git).resolve() if args.git is not None else None
    version, sha, state = repository_identity(source_root, git)
    assert_executable_identity(
        Path(args.exe),
        Path(args.receipt),
        version=version,
        sha=sha,
        state=state,
        repeat=True,
    )

    workdir = Path(args.workdir)
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    module = Path(args.identity_module).resolve()
    cmake = Path(args.cmake).resolve()
    cxx_compiler = Path(args.cxx_compiler).resolve()

    if git is not None:
        git_source = test_git_transitions(
            cmake,
            git,
            workdir,
            module,
            cxx_compiler,
            args.generator,
            args.make_program,
            args.build_config,
        )
        test_archive_spoof(
            cmake,
            git,
            git_source,
            workdir,
            module,
            cxx_compiler,
            args.generator,
            args.make_program,
            args.build_config,
        )


if __name__ == "__main__":
    main()
