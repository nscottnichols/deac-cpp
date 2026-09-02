import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path

TOP_LEVEL_KEYS = ["schema_version", "receipt_sha256", "receipt"]
RECEIPT_KEYS = [
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
SOURCE_IDENTITY_KEYS = [
    "schema_version",
    "semantic_version",
    "source_sha",
    "source_state",
]
INVALID_NORMALIZATION_DEFINITION = (
    "DEAC_TEST_FORCE_INVALID_NORMALIZATION_TRIALS=1"
)
POISON_GPU_FITNESS_DEFINITION = "DEAC_TEST_POISON_GPU_FITNESS=1"
HIPBLAS_CACHE_KEYS = {
    "CMAKE_DISABLE_FIND_PACKAGE_hipblas",
    "DEAC_HIPBLAS_INCLUDE_DIR",
    "DEAC_HIPBLAS_LIBRARY",
    "HIP_RUNTIME_INCLUDE_DIR",
    "hipblas_DIR",
    "hipblas_ROOT",
}


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


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def normalized_absolute_native_arguments(fragment):
    if os.name != "posix":
        raise AssertionError(
            "native-command link-fragment validation requires POSIX shell syntax"
        )
    try:
        arguments = shlex.split(fragment, posix=True)
    except ValueError as error:
        raise AssertionError(
            f"invalid native-command link fragment: {fragment!r}"
        ) from error
    return [
        os.path.normpath(argument)
        for argument in arguments
        if os.path.isabs(argument)
    ]


def validate_expected_link_library(command_fragments, expected_link_library):
    library_arguments = [
        argument
        for fragment in command_fragments
        if fragment["role"] == "libraries"
        for argument in normalized_absolute_native_arguments(fragment["fragment"])
    ]
    if expected_link_library is None:
        return library_arguments
    if not os.path.isabs(expected_link_library):
        raise AssertionError(
            "expected link-library path must be absolute: "
            f"{expected_link_library!r}"
        )
    expected_library = os.path.normpath(expected_link_library)
    if library_arguments.count(expected_library) != 1:
        raise AssertionError(
            "HIP+BLAS receipt must contain its resolved provider library "
            f"exactly once; expected {expected_link_library!r}, got "
            f"normalized absolute arguments {library_arguments!r}"
        )
    return library_arguments


def validate_native_link_fragment_regression():
    fixture_root = os.path.abspath("native link fragment fixture")
    provider = os.path.join(fixture_root, "provider archives", "libprovider.a")
    decoys = [
        os.path.join(fixture_root, "libhipblas-decoy-one.a"),
        os.path.join(fixture_root, "libhipblas-decoy-two.a"),
    ]
    fragments = [
        {
            "fragment": f"{shlex.quote(provider)} {shlex.quote(decoys[0])}",
            "role": "libraries",
        },
        {"fragment": shlex.quote(decoys[1]), "role": "libraries"},
    ]
    observed = validate_expected_link_library(fragments, provider)
    expected = [os.path.normpath(path) for path in (provider, *decoys)]
    if observed != expected:
        raise AssertionError(
            "native-command link-fragment regression: expected "
            f"{expected!r}, got {observed!r}"
        )
    if validate_expected_link_library(fragments, None) != expected:
        raise AssertionError(
            "native-command link-fragment regression changed the no-provider case"
        )

    rejection_cases = [
        (
            "missing",
            fragments,
            os.path.join(fixture_root, "provider archives", "missing-provider.a"),
        ),
        (
            "duplicate",
            [
                *fragments,
                {"fragment": shlex.quote(provider), "role": "libraries"},
            ],
            provider,
        ),
    ]
    for label, rejected_fragments, rejected_provider in rejection_cases:
        try:
            validate_expected_link_library(rejected_fragments, rejected_provider)
        except AssertionError as error:
            if "exactly once" not in str(error):
                raise AssertionError(
                    f"native-command {label} regression failed unexpectedly"
                ) from error
        else:
            raise AssertionError(
                f"native-command link-fragment regression accepted {label} provider"
            )


def parse_receipt(raw):
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as error:
        raise AssertionError(f"invalid build-receipt JSON: {raw!r}") from error
    if list(document) != TOP_LEVEL_KEYS:
        raise AssertionError(
            f"expected ordered receipt keys {TOP_LEVEL_KEYS}, got {list(document)}"
        )
    if document["schema_version"] != 1:
        raise AssertionError(f"unsupported build-receipt schema: {document!r}")
    receipt = document["receipt"]
    if list(receipt) != RECEIPT_KEYS:
        raise AssertionError(
            f"expected ordered payload keys {RECEIPT_KEYS}, got {list(receipt)}"
        )
    expected_digest = hashlib.sha256(canonical_json(receipt).encode()).hexdigest()
    if document["receipt_sha256"] != expected_digest:
        raise AssertionError(
            "build-receipt digest does not bind its canonical payload: "
            f"expected {expected_digest}, got {document['receipt_sha256']}"
        )
    expected_raw = canonical_json(document) + "\n"
    if raw != expected_raw:
        raise AssertionError("build receipt is valid JSON but not canonical")
    return document


def assert_string(value, label, *, nonempty=True):
    if not isinstance(value, str) or (nonempty and not value):
        raise AssertionError(f"{label} must be a string")


def is_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_compile_groups(
    compile_groups,
    *,
    label,
    fingerprint=None,
    expect_invalid_normalization_definition=False,
    expect_poison_gpu_fitness_definition=False,
):
    if not isinstance(compile_groups, list) or not compile_groups:
        raise AssertionError(f"{label} has no effective compile groups")
    compiled_sources = []
    used_languages = set()
    for group in compile_groups:
        if list(group) != [
            "command_fragments",
            "definitions",
            "frameworks",
            "includes",
            "language",
            "language_standard",
            "precompiled_headers",
            "sources",
            "sysroot",
        ]:
            raise AssertionError(f"noncanonical {label} compile group: {group!r}")
        assert_string(group["language"], f"{label} compile language")
        used_languages.add(group["language"])
        if not isinstance(group["command_fragments"], list):
            raise TypeError(f"{label} compile fragments must be a list")
        for fragment in group["command_fragments"]:
            assert_string(
                fragment.get("fragment"),
                f"{label} compile fragment",
                nonempty=False,
            )
            if fragment.get("role") is not None:
                assert_string(fragment["role"], f"{label} compile fragment role")
        if not isinstance(group["definitions"], list):
            raise TypeError(f"{label} definitions must be a list")
        expected_test_definitions = {
            INVALID_NORMALIZATION_DEFINITION: (
                1 if expect_invalid_normalization_definition else 0
            ),
            POISON_GPU_FITNESS_DEFINITION: (
                1 if expect_poison_gpu_fitness_definition else 0
            ),
        }
        for definition, expected_count in expected_test_definitions.items():
            if group["definitions"].count(definition) != expected_count:
                expectation = "exactly once" if expected_count else "never"
                raise AssertionError(
                    f"{label} compile group must contain the test-only "
                    f"definition {definition} {expectation}"
                )
        if fingerprint is not None:
            expected = f"DEAC_BUILD_TOOLCHAIN_FINGERPRINT_SHA256={fingerprint}"
            observed_fingerprints = [
                definition
                for definition in group["definitions"]
                if definition.startswith(
                    "DEAC_BUILD_TOOLCHAIN_FINGERPRINT_SHA256="
                )
            ]
            if observed_fingerprints != [expected]:
                raise AssertionError(
                    f"{label} compile group must bind exactly one expected "
                    "toolchain fingerprint; observed "
                    f"{observed_fingerprints!r}"
                )
        if not isinstance(group["sources"], list) or not group["sources"]:
            raise AssertionError(f"{label} compile group has no sources")
        compiled_sources.extend(group["sources"])
    return compiled_sources, used_languages


def validate_receipt(
    document,
    *,
    exe,
    build_dir,
    source_dir,
    expected_backend,
    expected_target,
    expected_dependency_target,
    expected_link_library,
    expect_invalid_normalization_definition,
    expect_poison_gpu_fitness_definition,
):
    receipt = document["receipt"]
    archive_tools = receipt["archive_tools"]
    if not isinstance(archive_tools, list):
        raise TypeError("archive tools must be a list")
    archive_tool_names = []
    for tool in archive_tools:
        if list(tool) != ["name", "path", "real_path", "sha256"]:
            raise AssertionError(f"noncanonical archive tool: {tool!r}")
        for key in ("name", "path", "real_path", "sha256"):
            assert_string(tool[key], f"archive tool {key}")
        archive_tool_names.append(tool["name"])
        if not is_sha256(tool["sha256"]):
            raise AssertionError("archive tool digest is not lowercase SHA-256")
        if not tool["real_path"].startswith("<"):
            if (
                not tool["path"].startswith("<")
                and Path(tool["path"]).resolve() != Path(tool["real_path"])
            ):
                raise AssertionError("archive tool path and real path disagree")
            if sha256_file(tool["real_path"]) != tool["sha256"]:
                raise AssertionError("archive tool digest disagrees with current bytes")
    if archive_tool_names != ["CMAKE_AR", "CMAKE_RANLIB"]:
        raise AssertionError(
            f"unexpected archive tool identities: {archive_tool_names!r}"
        )

    if receipt["backend"] != expected_backend:
        raise AssertionError(
            f"receipt backend {receipt['backend']!r} != {expected_backend!r}"
        )
    target = receipt["target"]
    if list(target) != [
        "name",
        "name_on_disk",
        "toolchain_fingerprint_sha256",
        "type",
    ]:
        raise AssertionError(f"noncanonical target receipt: {target!r}")
    if (
        target["name"] != expected_target
        or target["name_on_disk"] != exe.name
        or target["type"] != "EXECUTABLE"
    ):
        raise AssertionError(f"unexpected target receipt: {target!r}")
    fingerprint = target["toolchain_fingerprint_sha256"]
    if not is_sha256(fingerprint):
        raise AssertionError("target toolchain fingerprint is not lowercase SHA-256")

    source_identity = receipt["source_identity"]
    if list(source_identity) != SOURCE_IDENTITY_KEYS:
        raise AssertionError(f"noncanonical source identity: {source_identity!r}")
    identity = run([exe, "--build-identity"], exe.parent)
    if identity.stderr:
        raise AssertionError(f"--build-identity wrote to stderr: {identity.stderr!r}")
    if json.loads(identity.stdout) != source_identity:
        raise AssertionError("build receipt and embedded source identity disagree")

    build_system = receipt["build_system"]
    assert_string(build_system.get("configuration"), "configuration")
    generator = build_system.get("generator")
    if not isinstance(generator, dict) or not isinstance(
        generator.get("multi_config"), bool
    ):
        raise TypeError("receipt generator metadata is malformed")
    assert_string(generator.get("name"), "generator name")
    cmake = build_system.get("cmake")
    if not isinstance(cmake, dict):
        raise TypeError("receipt CMake identity is malformed")
    for key in ("path", "real_path", "sha256", "version"):
        assert_string(cmake.get(key), f"CMake {key}")
    if Path(cmake["path"]).resolve() != Path(cmake["real_path"]):
        raise AssertionError("receipt CMake path and real path disagree")
    if sha256_file(cmake["real_path"]) != cmake["sha256"]:
        raise AssertionError("receipt CMake digest disagrees with current bytes")

    cache_entries = receipt["cache_entries"]
    if not isinstance(cache_entries, list) or not cache_entries:
        raise AssertionError("receipt cache entries must be a nonempty list")
    names = [entry.get("name") for entry in cache_entries]
    if names != sorted(set(names)):
        raise AssertionError("receipt cache entries are not unique and sorted")
    for entry in cache_entries:
        if list(entry) != ["name", "type", "value"]:
            raise AssertionError(f"noncanonical cache entry: {entry!r}")
        assert_string(entry["name"], "cache entry name")
        if (entry["type"] is None) != (entry["value"] is None):
            raise AssertionError(f"partial cache entry: {entry!r}")
        if entry["type"] is not None:
            assert_string(entry["type"], "cache entry type")
            assert_string(entry["value"], "cache entry value", nonempty=False)
    backend_entries = [entry for entry in cache_entries if entry["name"] == "GPU_BACKEND"]
    if len(backend_entries) != 1 or backend_entries[0]["value"] != expected_backend:
        raise AssertionError("receipt cache does not bind the selected backend")
    home_entries = [
        entry for entry in cache_entries if entry["name"] == "CMAKE_HOME_DIRECTORY"
    ]
    if len(home_entries) != 1 or home_entries[0]["value"] != "<SOURCE_ROOT>":
        raise AssertionError("CMake source root was not canonicalized in the cache")
    missing_hipblas_keys = HIPBLAS_CACHE_KEYS.difference(names)
    if missing_hipblas_keys:
        raise AssertionError(
            "receipt omits material HIP/hipBLAS cache keys: "
            f"{sorted(missing_hipblas_keys)!r}"
        )

    compiled_sources, used_languages = validate_compile_groups(
        receipt["compile_groups"],
        label="target",
        fingerprint=fingerprint,
        expect_invalid_normalization_definition=(
            expect_invalid_normalization_definition
        ),
        expect_poison_gpu_fitness_definition=(
            expect_poison_gpu_fitness_definition
        ),
    )
    if not any(source.endswith("/deac/src/deac.cpp") for source in compiled_sources):
        raise AssertionError("receipt does not bind the primary solver source")
    if not any(source.endswith("_build_receipt.cpp") for source in compiled_sources):
        raise AssertionError("receipt does not include its embedded receipt object")

    dependencies = receipt["target_dependencies"]
    if not isinstance(dependencies, list):
        raise TypeError("receipt target dependencies must be a list")
    dependency_names = [dependency.get("name") for dependency in dependencies]
    if dependency_names != sorted(set(dependency_names)):
        raise AssertionError("receipt dependencies are not unique and sorted")
    expected_dependencies = {expected_dependency_target: "OBJECT_LIBRARY"}
    if expected_link_library is not None:
        expected_dependencies["deac_hipblas_link_contract"] = "INTERFACE_LIBRARY"
    if set(dependency_names) != set(expected_dependencies):
        raise AssertionError(
            "receipt dependency names changed: expected "
            f"{sorted(expected_dependencies)!r}, got {dependency_names!r}"
        )
    dependency_by_name = {
        dependency["name"]: dependency for dependency in dependencies
    }
    for dependency_name, dependency_type in expected_dependencies.items():
        dependency = dependency_by_name[dependency_name]
        if list(dependency) != [
            "archive",
            "compile_groups",
            "link",
            "name",
            "type",
        ]:
            raise AssertionError(f"noncanonical target dependency: {dependency!r}")
        if dependency["type"] != dependency_type:
            raise AssertionError(f"unexpected dependency target type: {dependency!r}")

    dependency = dependency_by_name[expected_dependency_target]
    if dependency["archive"] is not None or dependency["link"] is not None:
        raise AssertionError("identity object dependency unexpectedly archives or links")
    dependency_sources, dependency_languages = validate_compile_groups(
        dependency["compile_groups"],
        label="dependency",
        fingerprint=fingerprint,
        expect_invalid_normalization_definition=False,
        expect_poison_gpu_fitness_definition=False,
    )
    used_languages.update(dependency_languages)
    if not any(source.endswith("/deac/src/build_identity.cpp") for source in dependency_sources):
        raise AssertionError("receipt does not bind the embedded identity object source")
    if expected_link_library is not None:
        interface_dependency = dependency_by_name["deac_hipblas_link_contract"]
        if (
            interface_dependency["archive"] is not None
            or interface_dependency["compile_groups"] != []
            or interface_dependency["link"] is not None
        ):
            raise AssertionError(
                "hipBLAS interface dependency has nonempty build artifacts: "
                f"{interface_dependency!r}"
            )

    link = receipt["link"]
    if list(link) != ["command_fragments", "language", "lto", "sysroot"]:
        raise AssertionError(f"noncanonical link receipt: {link!r}")
    assert_string(link["language"], "link language")
    used_languages.add(link["language"])
    if not isinstance(link["lto"], bool) or not isinstance(
        link["command_fragments"], list
    ):
        raise TypeError("link receipt has invalid types")
    for fragment in link["command_fragments"]:
        if list(fragment) != ["fragment", "role"]:
            raise AssertionError(f"noncanonical link fragment: {fragment!r}")
        assert_string(fragment["fragment"], "link fragment", nonempty=False)
        if fragment["role"] is not None:
            assert_string(fragment["role"], "link fragment role")
    validate_expected_link_library(
        link["command_fragments"], expected_link_library
    )

    toolchains = receipt["toolchains"]
    if not isinstance(toolchains, list) or not toolchains:
        raise AssertionError("receipt has no bound toolchain")
    toolchain_languages = []
    for toolchain in toolchains:
        if list(toolchain) != ["compiler", "language"]:
            raise AssertionError(f"noncanonical toolchain: {toolchain!r}")
        language = toolchain["language"]
        assert_string(language, "toolchain language")
        toolchain_languages.append(language)
        compiler = toolchain["compiler"]
        for key in ("id", "path", "real_path", "sha256", "version"):
            assert_string(compiler.get(key), f"compiler {key}")
        if not is_sha256(compiler["sha256"]):
            raise AssertionError("compiler digest is not lowercase SHA-256")
        if not compiler["real_path"].startswith("<"):
            if Path(compiler["path"]).resolve() != Path(compiler["real_path"]):
                raise AssertionError("compiler path and real path disagree")
            if sha256_file(compiler["real_path"]) != compiler["sha256"]:
                raise AssertionError("compiler digest disagrees with current bytes")
    if toolchain_languages != sorted(set(toolchain_languages)):
        raise AssertionError("toolchains are not unique and sorted")
    if set(toolchain_languages) != used_languages:
        raise AssertionError("receipt does not bind every effective language toolchain")
    if str(build_dir.resolve()) in canonical_json(document):
        raise AssertionError("receipt leaked its relocatable build-tree pathname")
    if str(source_dir.resolve()) in canonical_json(document):
        raise AssertionError("receipt leaked its relocatable source-tree pathname")


def validate_receipt_endpoint(
    *,
    exe,
    receipt_path,
    build_dir,
    source_dir,
    expected_backend,
    expected_target,
    expected_dependency_target,
    expected_link_library,
    expect_invalid_normalization_definition,
    expect_poison_gpu_fitness_definition,
):
    first = run([exe, "--build-receipt"], exe.parent)
    second = run([exe, "--build-receipt"], exe.parent)
    if first.stderr or second.stderr:
        raise AssertionError("--build-receipt must not write to stderr")
    if first.stdout != second.stdout:
        raise AssertionError("repeated --build-receipt output changed")
    if receipt_path.read_text(encoding="utf-8") != first.stdout:
        raise AssertionError("adjacent and embedded build receipts differ byte-for-byte")
    document = parse_receipt(first.stdout)
    validate_receipt(
        document,
        exe=exe,
        build_dir=build_dir,
        source_dir=source_dir,
        expected_backend=expected_backend,
        expected_target=expected_target,
        expected_dependency_target=expected_dependency_target,
        expected_link_library=expected_link_library,
        expect_invalid_normalization_definition=(
            expect_invalid_normalization_definition
        ),
        expect_poison_gpu_fitness_definition=(
            expect_poison_gpu_fitness_definition
        ),
    )
    return first.stdout


def main():
    validate_native_link_fragment_regression()
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--build-dir", required=True)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--cmake", required=True)
    parser.add_argument("--expected-backend", required=True)
    parser.add_argument("--expected-target", required=True)
    parser.add_argument("--expected-dependency-target", required=True)
    parser.add_argument("--expected-link-library")
    parser.add_argument("--helper-exe")
    parser.add_argument("--helper-receipt")
    parser.add_argument("--expected-helper-target")
    parser.add_argument("--build-config")
    args = parser.parse_args()

    helper_arguments = (
        args.helper_exe,
        args.helper_receipt,
        args.expected_helper_target,
    )
    if any(argument is not None for argument in helper_arguments) and not all(
        argument is not None for argument in helper_arguments
    ):
        parser.error(
            "--helper-exe, --helper-receipt, and --expected-helper-target "
            "must be provided together"
        )

    exe = Path(args.exe).resolve()
    receipt_path = Path(args.receipt).resolve()
    embedded_receipt = validate_receipt_endpoint(
        exe=exe,
        receipt_path=receipt_path,
        build_dir=Path(args.build_dir),
        source_dir=Path(args.source_dir),
        expected_backend=args.expected_backend,
        expected_target=args.expected_target,
        expected_dependency_target=args.expected_dependency_target,
        expected_link_library=args.expected_link_library,
        expect_invalid_normalization_definition=False,
        expect_poison_gpu_fitness_definition=False,
    )
    if args.helper_exe is not None:
        validate_receipt_endpoint(
            exe=Path(args.helper_exe).resolve(),
            receipt_path=Path(args.helper_receipt).resolve(),
            build_dir=Path(args.build_dir),
            source_dir=Path(args.source_dir),
            expected_backend=args.expected_backend,
            expected_target=args.expected_helper_target,
            expected_dependency_target=args.expected_dependency_target,
            expected_link_library=args.expected_link_library,
            expect_invalid_normalization_definition=True,
            expect_poison_gpu_fitness_definition=(args.expected_backend != "none"),
        )

    workdir = Path(args.workdir)
    if workdir.exists():
        shutil.rmtree(workdir)
    install_prefix = workdir / "install"
    command = [args.cmake, "--install", args.build_dir, "--prefix", install_prefix]
    if args.build_config is not None:
        command.extend(["--config", args.build_config])
    run(command, args.build_dir)
    installed = install_prefix / "bin" / exe.name
    installed_receipt = run([installed, "--build-receipt"], installed.parent)
    if installed_receipt.stdout != embedded_receipt or installed_receipt.stderr:
        raise AssertionError("installed executable did not retain its embedded receipt")


if __name__ == "__main__":
    main()
