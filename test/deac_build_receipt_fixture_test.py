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
        fixture_source / "src" / "CMakeLists.txt",
        """cmake_minimum_required(VERSION 3.27)
project(deac_build_receipt_fixture LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(GPU_BACKEND none CACHE STRING "fixture backend")

add_library(receipt_dependency STATIC dependency.cpp)
add_executable(receipt_probe probe.cpp)
target_link_libraries(receipt_probe PRIVATE receipt_dependency)

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/DeacBuildReceipt.cmake")
deac_target_add_build_receipt(receipt_probe
    SOURCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.."
    GENERATED_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated/receipt"
    IDENTITY_NAME fixture
    RECEIPT
        "${CMAKE_CURRENT_BINARY_DIR}/receipt/$<CONFIG>/build-receipt.json"
    BACKEND "${GPU_BACKEND}"
    CACHE_KEYS
        CMAKE_BUILD_TYPE
        CMAKE_CONFIGURATION_TYPES
        CMAKE_CXX_COMPILER
        CMAKE_CXX_FLAGS
        CMAKE_EXE_LINKER_FLAGS
        CMAKE_GENERATOR
        CMAKE_HOME_DIRECTORY
        CMAKE_PREFIX_PATH
        GPU_BACKEND
    DEPENDENCY_TARGETS receipt_dependency)
""",
    )


def compiler_shim_contents(real_compiler, marker):
    return (
        "#!/bin/sh\n"
        f"# receipt fixture compiler marker: {marker}\n"
        f"exec {shlex.quote(str(real_compiler))} \"$@\"\n"
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


def build(cmake, build_directory, *, config=None, check=True):
    command = [cmake, "--build", build_directory, "--parallel", "2", "--verbose"]
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
    return document


def assert_static_dependency(document, fingerprint):
    dependencies = document["receipt"]["target_dependencies"]
    if len(dependencies) != 1:
        raise AssertionError(f"unexpected fixture dependencies: {dependencies!r}")
    dependency = dependencies[0]
    if dependency["name"] != "receipt_dependency":
        raise AssertionError(f"unexpected fixture dependency: {dependency!r}")
    if dependency["type"] != "STATIC_LIBRARY":
        raise AssertionError(f"dependency is not a static library: {dependency!r}")
    if dependency["link"] is not None or not isinstance(dependency["archive"], dict):
        raise AssertionError(f"static dependency archive is missing: {dependency!r}")
    if not isinstance(dependency["archive"].get("command_fragments"), list):
        raise TypeError("static dependency archive fragments are malformed")
    groups = dependency["compile_groups"]
    sources = [source for group in groups for source in group["sources"]]
    if not any(source.endswith("/dependency.cpp") for source in sources):
        raise AssertionError("static dependency source is absent from the receipt")
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
    first_build = build(cmake, build_directory)
    assert_build_actions(
        first_build,
        [
            "Embedding effective build receipt",
            "build_receipt.cpp",
            "receipt_probe",
        ],
    )
    receipt_path = build_directory / "receipt" / "Release" / "build-receipt.json"
    first_receipt = parse_receipt(receipt_path)
    first_fingerprint = first_receipt["receipt"]["target"][
        "toolchain_fingerprint_sha256"
    ]
    assert_static_dependency(first_receipt, first_fingerprint)
    executable = build_directory / "receipt_probe"
    assert_embedded_matches(executable, receipt_path)

    second_build = build(cmake, build_directory)
    assert_build_actions(
        second_build,
        [
            "Embedding effective build receipt",
            (
                "Building CXX object "
                "CMakeFiles/receipt_probe.dir/generated/receipt/Release/"
                "fixture_receipt_probe_build_receipt.cpp.o"
            ),
            "Linking CXX executable receipt_probe",
        ],
    )

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
    assert_static_dependency(replacement_receipt, replacement_fingerprint)
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
    assert_embedded_matches(
        build_directory / "Release" / "receipt_probe", release_receipt
    )
    assert_embedded_matches(
        build_directory / "Debug" / "receipt_probe", debug_receipt
    )


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


def test_rule_override_rejections(cmake, source, workdir, real_compiler):
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
    parser.add_argument("--workdir", required=True)
    args = parser.parse_args()

    workdir = Path(args.workdir).resolve()
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    source = workdir / "fixture-source"
    solver_source = Path(args.solver_source_root).resolve()
    real_compiler = Path(args.cxx_compiler).resolve()
    compiler_shim = workdir / "compiler-shim"
    create_fixture_source(solver_source, source)

    test_single_config_refresh_and_replacement(
        args.cmake, source, workdir, compiler_shim, real_compiler
    )
    test_ninja_multi_config(
        args.cmake, Path(args.ninja).resolve(), source, workdir, compiler_shim
    )
    test_rule_override_rejections(
        args.cmake, source, workdir, real_compiler
    )


if __name__ == "__main__":
    main()
