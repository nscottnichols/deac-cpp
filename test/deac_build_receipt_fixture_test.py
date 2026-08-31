import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
from pathlib import Path


def run(command, cwd, *, check=True):
    result = subprocess.run(
        [str(part) for part in command],
        cwd=cwd,
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


def write(path, contents, *, executable=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")
    if executable:
        path.chmod(0o755)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_fixture_modules(solver_source, fixture_source):
    module_names = [
        "DeacBuildIdentity.cmake",
        "DeacBuildReceipt.cmake",
        "GenerateDeacBuildReceipt.cmake",
        "VerifyDeacBuildReceiptTools.cmake",
        "deac_build_receipt_data.cpp.in",
    ]
    module_directory = fixture_source / "src" / "cmake"
    module_directory.mkdir(parents=True)
    for name in module_names:
        shutil.copy2(solver_source / "cmake" / name, module_directory / name)
    support_directory = fixture_source / "src" / "deac" / "src"
    support_directory.mkdir(parents=True)
    shutil.copy2(
        solver_source / "deac" / "src" / "build_identity.hpp",
        support_directory / "build_identity.hpp",
    )


def create_fixture_source(solver_source, fixture_source):
    fixture_source.mkdir(parents=True)
    copy_fixture_modules(solver_source, fixture_source)
    write(fixture_source / "VERSION", "1.2.3\n")
    write(
        fixture_source / "src" / "dependency.cpp",
        "int receipt_dependency() { return 17; }\n",
    )
    write(
        fixture_source / "src" / "probe.cpp",
        """#include "build_identity.hpp"
#include <iostream>

int receipt_dependency();

int main() {
    if (receipt_dependency() != 17) {
        return 2;
    }
    std::cout << deac_build_identity::build_receipt_json() << '\\n';
    return 0;
}
""",
    )
    write(
        fixture_source / "src" / "dependency_two.cpp",
        "int receipt_dependency_two() { return 23; }\n",
    )
    write(
        fixture_source / "src" / "probe_two.cpp",
        """#include "build_identity.hpp"
#include <iostream>

int receipt_dependency_two();

int main() {
    if (receipt_dependency_two() != 23) {
        return 2;
    }
    std::cout << deac_build_identity::build_receipt_json() << '\\n';
    return 0;
}
""",
    )
    write(
        fixture_source / "src" / "CMakeLists.txt",
        """cmake_minimum_required(VERSION 3.27)
project(deac_build_receipt_fixture LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(GPU_BACKEND none CACHE STRING "fixture backend")
set(DEAC_FIXTURE_ARCHIVER "" CACHE FILEPATH "effective fixture CMAKE_AR")
set(DEAC_FIXTURE_RULE_ATTACK "" CACHE STRING "fixture rule attack")
set(DEAC_FIXTURE_NATIVE_LANGUAGE_SHAPE "" CACHE STRING "native language shape")
option(DEAC_FIXTURE_PARALLEL_CONSUMERS "add a second receipt consumer" OFF)
option(DEAC_FIXTURE_DEPENDENCY_IPO_RELEASE "enable dependency Release IPO" OFF)

# CMake's persisted compiler-information file restores CMAKE_AR as a normal
# variable on every reconfigure, shadowing a changed -DCMAKE_AR cache entry.
# Rebind both scopes so the fixture can exercise a genuine effective archiver
# transition in one build tree.
if(NOT DEAC_FIXTURE_ARCHIVER STREQUAL "")
    set(CMAKE_AR "${DEAC_FIXTURE_ARCHIVER}" CACHE FILEPATH
        "effective fixture CMAKE_AR" FORCE)
    set(CMAKE_AR "${DEAC_FIXTURE_ARCHIVER}")
endif()

add_library(receipt_dependency STATIC dependency.cpp)
add_executable(receipt_probe probe.cpp)
target_link_libraries(receipt_probe PRIVATE receipt_dependency)
if(DEAC_FIXTURE_DEPENDENCY_IPO_RELEASE)
    set_property(TARGET receipt_dependency PROPERTY
        INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
endif()
if(DEAC_FIXTURE_PARALLEL_CONSUMERS)
    add_library(receipt_dependency_two STATIC dependency_two.cpp)
    add_executable(receipt_probe_two probe_two.cpp)
    target_link_libraries(receipt_probe_two PRIVATE receipt_dependency_two)
endif()

if(DEAC_FIXTURE_RULE_ATTACK STREQUAL "compile-and")
    string(APPEND CMAKE_CXX_COMPILE_OBJECT
        " && \\\"${CMAKE_COMMAND}\\\" -E false")
elseif(DEAC_FIXTURE_RULE_ATTACK STREQUAL "link-or")
    string(APPEND CMAKE_CXX_LINK_EXECUTABLE
        " || \\\"${CMAKE_COMMAND}\\\" -E true")
elseif(DEAC_FIXTURE_RULE_ATTACK STREQUAL "archive-pipe")
    string(APPEND CMAKE_CXX_ARCHIVE_CREATE
        " | \\\"${CMAKE_COMMAND}\\\" -E true")
elseif(DEAC_FIXTURE_RULE_ATTACK STREQUAL "quoted-pipe")
    string(APPEND CMAKE_CXX_COMPILE_OBJECT " \\\"literal|argument\\\"")
elseif(NOT DEAC_FIXTURE_RULE_ATTACK STREQUAL "")
    message(FATAL_ERROR "unknown fixture rule attack")
endif()

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/DeacBuildReceipt.cmake")
if(DEAC_FIXTURE_NATIVE_LANGUAGE_SHAPE MATCHES "^(CUDA|HIP)$")
    _deac_build_receipt_reject_unsupported_languages(
        CXX "${DEAC_FIXTURE_NATIVE_LANGUAGE_SHAPE}")
elseif(NOT DEAC_FIXTURE_NATIVE_LANGUAGE_SHAPE STREQUAL "")
    message(FATAL_ERROR "invalid native language shape")
endif()
set(DEAC_FIXTURE_CACHE_KEYS
    CMAKE_AR
    CMAKE_BUILD_TYPE
    CMAKE_CONFIGURATION_TYPES
    CMAKE_CXX_COMPILER
    CMAKE_CXX_FLAGS
    CMAKE_EXE_LINKER_FLAGS
    CMAKE_GENERATOR
    CMAKE_HOME_DIRECTORY
    CMAKE_PREFIX_PATH
    CMAKE_RANLIB
    GPU_BACKEND)
deac_target_add_build_receipt(receipt_probe
    SOURCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.."
    GENERATED_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated/receipt"
    IDENTITY_NAME fixture
    RECEIPT
        "${CMAKE_CURRENT_BINARY_DIR}/receipt/$<CONFIG>/build-receipt.json"
    BACKEND "${GPU_BACKEND}"
    CACHE_KEYS ${DEAC_FIXTURE_CACHE_KEYS}
    DEPENDENCY_TARGETS receipt_dependency)
if(DEAC_FIXTURE_PARALLEL_CONSUMERS)
    deac_target_add_build_receipt(receipt_probe_two
        SOURCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.."
        GENERATED_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated/receipt"
        IDENTITY_NAME fixture
        RECEIPT
            "${CMAKE_CURRENT_BINARY_DIR}/receipt/$<CONFIG>/build-receipt-two.json"
        BACKEND "${GPU_BACKEND}"
        CACHE_KEYS ${DEAC_FIXTURE_CACHE_KEYS}
        DEPENDENCY_TARGETS receipt_dependency_two)
endif()
""",
    )


def compiler_shim_contents(real_compiler, marker):
    return (
        "#!/bin/sh\n"
        f"# receipt fixture compiler marker: {marker}\n"
        f"exec {shlex.quote(str(real_compiler))} \"$@\"\n"
    )


def archive_shim_contents(real_archiver, marker):
    return (
        "#!/bin/sh\n"
        f"# receipt fixture archiver marker: {marker}\n"
        f"exec {shlex.quote(str(real_archiver))} \"$@\"\n"
    )


def configure(cmake, source, build, compiler, *extra, check=True):
    return run(
        [
            cmake,
            "-S",
            source / "src",
            "-B",
            build,
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_CXX_COMPILER={compiler}",
            "-DGPU_BACKEND=none",
            *extra,
        ],
        source,
        check=check,
    )


def build(cmake, build_directory, *, config=None, parallel=1, check=True):
    command = [
        cmake,
        "--build",
        build_directory,
        "--parallel",
        str(parallel),
        "--verbose",
    ]
    if config is not None:
        command.extend(["--config", config])
    return run(command, build_directory, check=check)


def parse_receipt(path):
    raw = path.read_text(encoding="utf-8")
    document = json.loads(raw)
    canonical = json.dumps(document, ensure_ascii=False, separators=(",", ":"))
    if raw != canonical + "\n":
        raise AssertionError("fixture receipt is not canonical JSON")
    payload = json.dumps(
        document["receipt"], ensure_ascii=False, separators=(",", ":")
    )
    expected = hashlib.sha256(payload.encode()).hexdigest()
    if document["receipt_sha256"] != expected:
        raise AssertionError("fixture receipt digest does not bind its payload")
    expected_keys = [
        "archive_tools",
        "backend",
        "build_system",
        "cache_entries",
        "compile_groups",
        "link",
        "source_identity",
        "target",
        "target_dependencies",
        "toolchains",
    ]
    if list(document["receipt"]) != expected_keys:
        raise AssertionError("fixture receipt payload keys are not canonical")
    return document


def archive_tool(document, name):
    tools = document["receipt"]["archive_tools"]
    if [tool.get("name") for tool in tools] != ["CMAKE_AR", "CMAKE_RANLIB"]:
        raise AssertionError(f"unexpected archive tools: {tools!r}")
    tool = next(tool for tool in tools if tool["name"] == name)
    if list(tool) != ["name", "path", "real_path", "sha256"]:
        raise AssertionError(f"noncanonical archive tool: {tool!r}")
    if len(tool["sha256"]) != 64 or any(
        character not in "0123456789abcdef" for character in tool["sha256"]
    ):
        raise AssertionError(f"invalid archive tool digest: {tool!r}")
    if not tool["real_path"].startswith("<"):
        if (
            not tool["path"].startswith("<")
            and Path(tool["path"]).resolve() != Path(tool["real_path"])
        ):
            raise AssertionError(f"archive tool paths disagree: {tool!r}")
        if sha256_file(tool["real_path"]) != tool["sha256"]:
            raise AssertionError(f"archive tool digest disagrees: {tool!r}")
    return tool


def assert_no_git_source(document, version):
    expected = {
        "schema_version": 1,
        "semantic_version": version,
        "source_sha": None,
        "source_state": "unavailable",
    }
    if document["receipt"]["source_identity"] != expected:
        raise AssertionError(
            "no-Git fixture acquired unexpected source identity: "
            f"{document['receipt']['source_identity']!r}"
        )


def assert_static_dependency(
    document, fingerprint, expected_name="receipt_dependency"
):
    dependencies = document["receipt"]["target_dependencies"]
    if len(dependencies) != 1:
        raise AssertionError(f"unexpected fixture dependencies: {dependencies!r}")
    dependency = dependencies[0]
    if dependency["name"] != expected_name:
        raise AssertionError(f"unexpected fixture dependency: {dependency!r}")
    if dependency["type"] != "STATIC_LIBRARY":
        raise AssertionError(f"dependency is not a static library: {dependency!r}")
    if dependency["link"] is not None or not isinstance(dependency["archive"], dict):
        raise AssertionError(f"static dependency archive is missing: {dependency!r}")
    if not isinstance(dependency["archive"].get("command_fragments"), list):
        raise TypeError("static dependency archive fragments are malformed")
    groups = dependency["compile_groups"]
    sources = [source for group in groups for source in group["sources"]]
    expected_source = expected_name.removeprefix("receipt_") + ".cpp"
    if not any(source.endswith("/" + expected_source) for source in sources):
        raise AssertionError(
            f"static dependency source {expected_source!r} is absent from the receipt"
        )
    definition = f"DEAC_BUILD_TOOLCHAIN_FINGERPRINT_SHA256={fingerprint}"
    if any(group["definitions"].count(definition) != 1 for group in groups):
        raise AssertionError("static dependency does not bind the tool fingerprint")


def assert_embedded_matches(executable, receipt_path):
    endpoint = run([executable], executable.parent)
    if endpoint.stderr:
        raise AssertionError(f"fixture endpoint wrote stderr: {endpoint.stderr!r}")
    if endpoint.stdout != receipt_path.read_text(encoding="utf-8"):
        raise AssertionError("fixture embedded and adjacent receipts disagree")


def assert_build_actions(result, required):
    output = result.stdout + result.stderr
    missing = [needle for needle in required if needle not in output]
    if missing:
        raise AssertionError(
            f"build output omitted required actions {missing!r}:\n{output}"
        )


def receipt_fingerprint(document):
    return document["receipt"]["target"]["toolchain_fingerprint_sha256"]


def cache_value(document, name):
    matches = [
        entry["value"]
        for entry in document["receipt"]["cache_entries"]
        if entry["name"] == name
    ]
    if len(matches) != 1:
        raise AssertionError(f"expected one {name} cache entry, got {matches!r}")
    return matches[0]


def test_single_config_refresh_and_replacement(
    cmake, source, workdir, compiler_shim, real_compiler
):
    build_directory = workdir / "single-build"
    write(
        compiler_shim,
        compiler_shim_contents(real_compiler, "first"),
        executable=True,
    )
    configure(cmake, source, build_directory, compiler_shim)
    receipt_actions = [
        "Embedding effective build receipt",
        (
            "Building CXX object "
            "CMakeFiles/receipt_probe.dir/generated/receipt/Release/"
            "fixture_receipt_probe_build_receipt.cpp.o"
        ),
        "Linking CXX executable receipt_probe",
    ]
    first_build = build(cmake, build_directory)
    assert_build_actions(first_build, receipt_actions)
    receipt_path = build_directory / "receipt" / "Release" / "build-receipt.json"
    first_receipt = parse_receipt(receipt_path)
    first_fingerprint = first_receipt["receipt"]["target"][
        "toolchain_fingerprint_sha256"
    ]
    archive_tool(first_receipt, "CMAKE_AR")
    archive_tool(first_receipt, "CMAKE_RANLIB")
    assert_no_git_source(first_receipt, "1.2.3")
    assert_static_dependency(first_receipt, first_fingerprint)
    executable = build_directory / "receipt_probe"
    assert_embedded_matches(executable, receipt_path)

    second_build = build(cmake, build_directory)
    assert_build_actions(second_build, receipt_actions)
    assert_embedded_matches(executable, receipt_path)

    # Refresh a material build-time input without reconfiguring.  The same
    # generated graph must compile the new receipt object before relinking;
    # comparing only the freely replaceable adjacent JSON would miss a stale
    # embedded receipt.
    write(source / "VERSION", "1.2.4\n")
    changed_build = build(cmake, build_directory)
    assert_build_actions(changed_build, receipt_actions)
    changed_receipt = parse_receipt(receipt_path)
    if changed_receipt["receipt_sha256"] == first_receipt["receipt_sha256"]:
        raise AssertionError("material receipt refresh did not change its digest")
    if changed_receipt["receipt"]["source_identity"]["semantic_version"] != "1.2.4":
        raise AssertionError("material receipt refresh retained the old version")
    assert_no_git_source(changed_receipt, "1.2.4")
    assert_embedded_matches(executable, receipt_path)

    write(
        compiler_shim,
        compiler_shim_contents(real_compiler, "replacement"),
        executable=True,
    )
    rejected = build(cmake, build_directory, check=False)
    if rejected.returncode == 0:
        raise AssertionError("persistent compiler replacement was accepted")
    rejected_output = rejected.stdout + rejected.stderr
    if "compiler changed after configuration" not in rejected_output:
        raise AssertionError(
            "compiler replacement failed for an unexpected reason:\n" + rejected_output
        )

    configure(cmake, source, build_directory, compiler_shim)
    replacement_build = build(cmake, build_directory)
    assert_build_actions(
        replacement_build,
        [
            "dependency.cpp",
            "probe.cpp",
            "build_receipt.cpp",
            "receipt_dependency",
            "receipt_probe",
        ],
    )
    replacement_receipt = parse_receipt(receipt_path)
    replacement_fingerprint = replacement_receipt["receipt"]["target"][
        "toolchain_fingerprint_sha256"
    ]
    if replacement_fingerprint == first_fingerprint:
        raise AssertionError("compiler replacement did not change the fingerprint")
    assert_no_git_source(replacement_receipt, "1.2.4")
    assert_static_dependency(replacement_receipt, replacement_fingerprint)
    assert_embedded_matches(build_directory / "receipt_probe", receipt_path)


def test_single_config_ninja(cmake, ninja, source, workdir, real_compiler):
    build_directory = workdir / "ninja-single-build"
    configure(
        cmake,
        source,
        build_directory,
        real_compiler,
        "-G",
        "Ninja",
        f"-DCMAKE_MAKE_PROGRAM={ninja}",
    )
    required = [
        "GenerateDeacBuildReceipt.cmake",
        "fixture_receipt_probe_build_receipt.cpp.o",
        "-o receipt_probe",
    ]
    first_build = build(cmake, build_directory)
    assert_build_actions(first_build, required)
    second_build = build(cmake, build_directory)
    assert_build_actions(second_build, required)

    receipt_path = build_directory / "receipt" / "Release" / "build-receipt.json"
    document = parse_receipt(receipt_path)
    generator = document["receipt"]["build_system"]["generator"]
    if generator["name"] != "Ninja" or generator["multi_config"]:
        raise AssertionError(f"unexpected single-config Ninja receipt: {generator!r}")
    assert_no_git_source(document, "1.2.4")
    assert_static_dependency(document, receipt_fingerprint(document))
    assert_embedded_matches(build_directory / "receipt_probe", receipt_path)


def test_material_flag_reconfiguration(cmake, source, workdir, real_compiler):
    build_directory = workdir / "material-flag-build"
    configure(cmake, source, build_directory, real_compiler)
    build(cmake, build_directory)
    receipt_path = build_directory / "receipt" / "Release" / "build-receipt.json"
    first_receipt = parse_receipt(receipt_path)

    flag = "-DDEAC_FIXTURE_MATERIAL_FLAG=17"
    configure(
        cmake,
        source,
        build_directory,
        real_compiler,
        f"-DCMAKE_CXX_FLAGS={flag}",
    )
    changed_build = build(cmake, build_directory)
    assert_build_actions(
        changed_build,
        [
            flag,
            "dependency.cpp.o",
            "probe.cpp.o",
            "fixture_receipt_probe_build_receipt.cpp.o",
            "Linking CXX static library libreceipt_dependency.a",
            "Linking CXX executable receipt_probe",
        ],
    )
    changed_receipt = parse_receipt(receipt_path)
    if changed_receipt["receipt_sha256"] == first_receipt["receipt_sha256"]:
        raise AssertionError("material flag reconfigure did not change the receipt")
    if receipt_fingerprint(changed_receipt) != receipt_fingerprint(first_receipt):
        raise AssertionError("material flags unexpectedly changed the tool fingerprint")
    if cache_value(changed_receipt, "CMAKE_CXX_FLAGS") != flag:
        raise AssertionError("material flag is absent from the receipt cache")
    if flag not in json.dumps(changed_receipt, separators=(",", ":")):
        raise AssertionError("material flag is absent from effective command data")
    assert_no_git_source(changed_receipt, "1.2.4")
    assert_embedded_matches(build_directory / "receipt_probe", receipt_path)


def test_archive_tool_reconfiguration(
    cmake, source, workdir, real_compiler, real_archiver
):
    build_directory = workdir / "archive-tool-build"
    tool_directory = workdir / "archive-tools"
    ar_one = tool_directory / "ar-one"
    ar_two = tool_directory / "ar-two"
    write(
        ar_one,
        archive_shim_contents(real_archiver, "one"),
        executable=True,
    )
    write(
        ar_two,
        archive_shim_contents(real_archiver, "two"),
        executable=True,
    )

    configure(
        cmake,
        source,
        build_directory,
        real_compiler,
        f"-DDEAC_FIXTURE_ARCHIVER={ar_one}",
    )
    first_build = build(cmake, build_directory)
    assert_build_actions(
        first_build,
        [str(ar_one), "Linking CXX static library libreceipt_dependency.a"],
    )
    receipt_path = build_directory / "receipt" / "Release" / "build-receipt.json"
    first_receipt = parse_receipt(receipt_path)
    first_ar = archive_tool(first_receipt, "CMAKE_AR")
    if first_ar["path"] != str(ar_one):
        raise AssertionError(f"receipt recorded the wrong first archiver: {first_ar!r}")
    if cache_value(first_receipt, "CMAKE_AR") != str(ar_one):
        raise AssertionError("receipt cache omitted the first effective archiver")
    assert_embedded_matches(build_directory / "receipt_probe", receipt_path)

    write(
        ar_one,
        archive_shim_contents(real_archiver, "tampered"),
        executable=True,
    )
    rejected = build(cmake, build_directory, check=False)
    if rejected.returncode == 0:
        raise AssertionError("persistent CMAKE_AR replacement was accepted")
    rejected_output = rejected.stdout + rejected.stderr
    if "CMAKE_AR executable changed after configuration" not in rejected_output:
        raise AssertionError(
            "CMAKE_AR replacement failed for an unexpected reason:\n"
            + rejected_output
        )

    configure(
        cmake,
        source,
        build_directory,
        real_compiler,
        f"-DDEAC_FIXTURE_ARCHIVER={ar_two}",
    )
    second_build = build(cmake, build_directory)
    assert_build_actions(
        second_build,
        [
            str(ar_two),
            "dependency.cpp.o",
            "Linking CXX static library libreceipt_dependency.a",
            "Linking CXX executable receipt_probe",
        ],
    )
    second_receipt = parse_receipt(receipt_path)
    second_ar = archive_tool(second_receipt, "CMAKE_AR")
    if second_ar["path"] != str(ar_two) or second_ar["sha256"] != sha256_file(ar_two):
        raise AssertionError(f"receipt recorded the wrong second archiver: {second_ar!r}")
    if cache_value(second_receipt, "CMAKE_AR") != str(ar_two):
        raise AssertionError("receipt cache omitted the second effective archiver")
    if receipt_fingerprint(second_receipt) == receipt_fingerprint(first_receipt):
        raise AssertionError("archiver reconfigure did not invalidate the fingerprint")
    if second_receipt["receipt_sha256"] == first_receipt["receipt_sha256"]:
        raise AssertionError("archiver reconfigure did not change the receipt identity")
    assert_static_dependency(second_receipt, receipt_fingerprint(second_receipt))
    assert_no_git_source(second_receipt, "1.2.4")
    assert_embedded_matches(build_directory / "receipt_probe", receipt_path)


def graph_configurations(ninja, build_directory, configuration):
    result = run(
        [
            ninja,
            "-C",
            build_directory,
            "-f",
            f"build-{configuration}.ninja",
            "-t",
            "commands",
            "receipt_probe",
        ],
        build_directory,
    )
    marker = "DEAC_BUILD_RECEIPT_CONFIGURATION:STRING="
    configurations = set()
    for line in result.stdout.splitlines():
        if marker in line:
            configurations.add(line.split(marker, 1)[1].split()[0])
    return configurations


def test_ninja_multi_config(cmake, ninja, source, workdir, compiler_shim):
    build_directory = workdir / "multi-build"
    run(
        [
            cmake,
            "-S",
            source / "src",
            "-B",
            build_directory,
            "-G",
            "Ninja Multi-Config",
            f"-DCMAKE_MAKE_PROGRAM={ninja}",
            f"-DCMAKE_CXX_COMPILER={compiler_shim}",
            "-DCMAKE_CONFIGURATION_TYPES=Debug;Release;RelWithDebInfo",
            (
                "-DCMAKE_PREFIX_PATH="
                f"{source / 'src' / 'prefix-a'};"
                f"{build_directory / 'prefix-b'};/external"
            ),
            "-DGPU_BACKEND=none",
        ],
        source,
    )
    for configuration in ("Release", "Debug"):
        graph = graph_configurations(ninja, build_directory, configuration)
        if graph != {configuration}:
            raise AssertionError(
                f"{configuration} graph crosses receipt configurations: {graph!r}"
            )

    build(cmake, build_directory, config="Release")
    generated_root = build_directory / "generated" / "receipt"
    receipt_root = build_directory / "receipt"
    release_source = generated_root / "Release" / "fixture_receipt_probe_build_receipt.cpp"
    release_receipt = receipt_root / "Release" / "build-receipt.json"
    if not release_source.is_file() or not release_receipt.is_file():
        raise AssertionError("Release receipt outputs were not generated")
    for other in ("Debug", "RelWithDebInfo"):
        if (generated_root / other).exists() or (receipt_root / other).exists():
            raise AssertionError(f"Release build mutated {other} receipt outputs")
    release_bytes = (release_source.read_bytes(), release_receipt.read_bytes())
    release_times = (release_source.stat().st_mtime_ns, release_receipt.stat().st_mtime_ns)

    build(cmake, build_directory, config="Debug")
    debug_source = generated_root / "Debug" / "fixture_receipt_probe_build_receipt.cpp"
    debug_receipt = receipt_root / "Debug" / "build-receipt.json"
    if not debug_source.is_file() or not debug_receipt.is_file():
        raise AssertionError("Debug receipt outputs were not generated")
    if release_bytes != (release_source.read_bytes(), release_receipt.read_bytes()):
        raise AssertionError("Debug build changed Release receipt bytes")
    if release_times != (
        release_source.stat().st_mtime_ns,
        release_receipt.stat().st_mtime_ns,
    ):
        raise AssertionError("Debug build touched Release receipt outputs")
    release_document = parse_receipt(release_receipt)
    debug_document = parse_receipt(debug_receipt)
    if release_document["receipt"]["build_system"]["configuration"] != "Release":
        raise AssertionError("Release receipt records another configuration")
    if debug_document["receipt"]["build_system"]["configuration"] != "Debug":
        raise AssertionError("Debug receipt records another configuration")
    for document in (release_document, debug_document):
        entries = {
            entry["name"]: entry["value"]
            for entry in document["receipt"]["cache_entries"]
        }
        expected_configurations = "Debug;Release;RelWithDebInfo"
        if entries.get("CMAKE_CONFIGURATION_TYPES") != expected_configurations:
            raise AssertionError(
                "multi-config cache value lost list separators: "
                f"{entries.get('CMAKE_CONFIGURATION_TYPES')!r}"
            )
        expected_prefix_path = (
            "<SOURCE_ROOT>/prefix-a;<BUILD_ROOT>/prefix-b;/external"
        )
        if entries.get("CMAKE_PREFIX_PATH") != expected_prefix_path:
            raise AssertionError(
                "path-list cache value was not normalized elementwise: "
                f"{entries.get('CMAKE_PREFIX_PATH')!r}"
            )
        assert_no_git_source(document, "1.2.4")
        assert_static_dependency(document, receipt_fingerprint(document))
    assert_embedded_matches(
        build_directory / "Release" / "receipt_probe", release_receipt
    )
    assert_embedded_matches(
        build_directory / "Debug" / "receipt_probe", debug_receipt
    )


def test_parallel_receipt_consumers(
    cmake, ninja, solver_source, workdir, real_compiler
):
    source = workdir / "parallel-source"
    create_fixture_source(solver_source, source)
    build_directory = workdir / "parallel-build"
    configure(
        cmake,
        source,
        build_directory,
        real_compiler,
        "-G",
        "Ninja",
        f"-DCMAKE_MAKE_PROGRAM={ninja}",
        "-DDEAC_FIXTURE_PARALLEL_CONSUMERS=ON",
    )
    result = build(cmake, build_directory, parallel=4)
    assert_build_actions(
        result,
        [
            "fixture_receipt_probe_build_receipt.cpp.o",
            "fixture_receipt_probe_two_build_receipt.cpp.o",
            "-o receipt_probe ",
            "-o receipt_probe_two",
        ],
    )

    receipt_root = build_directory / "receipt" / "Release"
    checks = [
        (
            "receipt_probe",
            "receipt_dependency",
            receipt_root / "build-receipt.json",
        ),
        (
            "receipt_probe_two",
            "receipt_dependency_two",
            receipt_root / "build-receipt-two.json",
        ),
    ]
    for executable_name, dependency_name, receipt_path in checks:
        document = parse_receipt(receipt_path)
        if document["receipt"]["target"]["name"] != executable_name:
            raise AssertionError(f"parallel receipt names the wrong target: {document!r}")
        assert_no_git_source(document, "1.2.3")
        assert_static_dependency(
            document,
            receipt_fingerprint(document),
            expected_name=dependency_name,
        )
        assert_embedded_matches(build_directory / executable_name, receipt_path)


def assert_configure_rejected(cmake, source, build_directory, compiler, arguments, text):
    result = configure(
        cmake,
        source,
        build_directory,
        compiler,
        *arguments,
        check=False,
    )
    if result.returncode == 0:
        raise AssertionError(f"unsupported configure route was accepted: {arguments!r}")
    output = result.stdout + result.stderr
    if text not in output:
        raise AssertionError(
            f"configure rejection did not mention {text!r}:\n{output}"
        )


def test_rule_override_rejections(
    cmake, ninja, source, workdir, real_compiler
):
    compile_rule = (
        "/usr/bin/env <CMAKE_CXX_COMPILER> <DEFINES> <INCLUDES> <FLAGS> "
        "-o <OBJECT> -c <SOURCE>"
    )
    link_rule = (
        "/usr/bin/env <CMAKE_CXX_COMPILER> <FLAGS> <CMAKE_CXX_LINK_FLAGS> "
        "<LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>"
    )
    attacks = [
        ("compile", [f"-DCMAKE_CXX_COMPILE_OBJECT={compile_rule}"], "CMAKE_CXX_COMPILE_OBJECT"),
        ("link", [f"-DCMAKE_CXX_LINK_EXECUTABLE={link_rule}"], "CMAKE_CXX_LINK_EXECUTABLE"),
        (
            "cuda",
            ["-DCMAKE_CUDA_COMPILE_OBJECT=/usr/bin/env <CMAKE_CUDA_COMPILER>"],
            "CMAKE_CUDA_COMPILE_OBJECT",
        ),
        (
            "hip",
            ["-DCMAKE_HIP_LINK_EXECUTABLE=/usr/bin/env <CMAKE_HIP_COMPILER>"],
            "CMAKE_HIP_LINK_EXECUTABLE",
        ),
        (
            "cuda-device-link",
            [
                (
                    "-DCMAKE_CUDA_DEVICE_LINK_EXECUTABLE="
                    "/usr/bin/env <CMAKE_CUDA_COMPILER>"
                )
            ],
            "CMAKE_CUDA_DEVICE_LINK_EXECUTABLE",
        ),
    ]
    for name, arguments, expected in attacks:
        assert_configure_rejected(
            cmake,
            source,
            workdir / f"rejected-{name}",
            real_compiler,
            arguments,
            expected,
        )

    project_rule_attacks = [
        ("compile-and", "CMAKE_CXX_COMPILE_OBJECT"),
        ("link-or", "CMAKE_CXX_LINK_EXECUTABLE"),
        ("archive-pipe", "CMAKE_CXX_ARCHIVE_CREATE"),
    ]
    for attack, expected in project_rule_attacks:
        assert_configure_rejected(
            cmake,
            source,
            workdir / f"rejected-project-rule-{attack}",
            real_compiler,
            [f"-DDEAC_FIXTURE_RULE_ATTACK={attack}"],
            expected,
        )

    # A literal operator inside a quoted argument remains one represented
    # command and must not be confused with an active shell pipeline.
    configure(
        cmake,
        source,
        workdir / "accepted-quoted-pipe",
        real_compiler,
        "-DDEAC_FIXTURE_RULE_ATTACK=quoted-pipe",
    )

    assert_configure_rejected(
        cmake,
        source,
        workdir / "rejected-generic-ipo",
        real_compiler,
        ["-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON"],
        "INTERPROCEDURAL_OPTIMIZATION",
    )
    assert_configure_rejected(
        cmake,
        source,
        workdir / "rejected-release-ipo",
        real_compiler,
        ["-DCMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE=ON"],
        "INTERPROCEDURAL_OPTIMIZATION_RELEASE",
    )
    assert_configure_rejected(
        cmake,
        source,
        workdir / "rejected-dependency-release-ipo",
        real_compiler,
        ["-DDEAC_FIXTURE_DEPENDENCY_IPO_RELEASE=ON"],
        "INTERPROCEDURAL_OPTIMIZATION_RELEASE",
    )
    assert_configure_rejected(
        cmake,
        source,
        workdir / "rejected-multi-release-ipo",
        real_compiler,
        [
            "-G",
            "Ninja Multi-Config",
            f"-DCMAKE_MAKE_PROGRAM={ninja}",
            "-DCMAKE_CONFIGURATION_TYPES=Debug;Release;RelWithDebInfo",
            "-DCMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE=ON",
        ],
        "INTERPROCEDURAL_OPTIMIZATION_RELEASE",
    )

    for language in ("CUDA", "HIP"):
        assert_configure_rejected(
            cmake,
            source,
            workdir / f"rejected-native-{language.lower()}",
            real_compiler,
            [f"-DDEAC_FIXTURE_NATIVE_LANGUAGE_SHAPE={language}"],
            f"native CMake {language} language",
        )

    override = workdir / "user-rules.cmake"
    write(override, "# intentionally empty override route\n")
    assert_configure_rejected(
        cmake,
        source,
        workdir / "rejected-user-rules",
        real_compiler,
        [f"-DCMAKE_USER_MAKE_RULES_OVERRIDE={override}"],
        "CMAKE_USER_MAKE_RULES_OVERRIDE",
    )
    toolchain = workdir / "toolchain.cmake"
    write(toolchain, f"set(CMAKE_CXX_COMPILER {json.dumps(str(real_compiler))})\n")
    assert_configure_rejected(
        cmake,
        source,
        workdir / "rejected-toolchain",
        real_compiler,
        [f"-DCMAKE_TOOLCHAIN_FILE={toolchain}"],
        "CMAKE_TOOLCHAIN_FILE",
    )
    assert_configure_rejected(
        cmake,
        source,
        workdir / "rejected-module-path",
        real_compiler,
        [f"-DCMAKE_MODULE_PATH={workdir}"],
        "CMAKE_MODULE_PATH",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmake", required=True)
    parser.add_argument("--ninja", required=True)
    parser.add_argument("--solver-source-root", required=True)
    parser.add_argument("--cxx-compiler", required=True)
    parser.add_argument("--archiver", required=True)
    parser.add_argument("--workdir", required=True)
    args = parser.parse_args()

    workdir = Path(args.workdir).resolve()
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    source = workdir / "fixture-source"
    solver_source = Path(args.solver_source_root).resolve()
    real_compiler = Path(args.cxx_compiler).resolve()
    real_archiver = Path(args.archiver).resolve()
    ninja = Path(args.ninja).resolve()
    compiler_shim = workdir / "compiler-shim"
    create_fixture_source(solver_source, source)

    test_single_config_refresh_and_replacement(
        args.cmake, source, workdir, compiler_shim, real_compiler
    )
    test_single_config_ninja(
        args.cmake, ninja, source, workdir, real_compiler
    )
    test_material_flag_reconfiguration(
        args.cmake, source, workdir, real_compiler
    )
    test_archive_tool_reconfiguration(
        args.cmake, source, workdir, real_compiler, real_archiver
    )
    test_ninja_multi_config(
        args.cmake, ninja, source, workdir, compiler_shim
    )
    test_parallel_receipt_consumers(
        args.cmake, ninja, solver_source, workdir, real_compiler
    )
    test_rule_override_rejections(
        args.cmake, ninja, source, workdir, real_compiler
    )


if __name__ == "__main__":
    main()
