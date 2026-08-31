import argparse
import hashlib
import json
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


def validate_compile_groups(compile_groups, *, label, fingerprint=None):
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
        if fingerprint is not None:
            expected = f"DEAC_BUILD_TOOLCHAIN_FINGERPRINT_SHA256={fingerprint}"
            if group["definitions"].count(expected) != 1:
                raise AssertionError(
                    f"{label} compile group does not bind the toolchain fingerprint"
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

    compiled_sources, used_languages = validate_compile_groups(
        receipt["compile_groups"], label="target", fingerprint=fingerprint
    )
    if not any(source.endswith("/deac/src/deac.cpp") for source in compiled_sources):
        raise AssertionError("receipt does not bind the primary solver source")
    if not any(source.endswith("_build_receipt.cpp") for source in compiled_sources):
        raise AssertionError("receipt does not include its embedded receipt object")

    dependencies = receipt["target_dependencies"]
    if not isinstance(dependencies, list) or len(dependencies) != 1:
        raise AssertionError("receipt must bind its one direct target dependency")
    dependency = dependencies[0]
    if list(dependency) != ["archive", "compile_groups", "link", "name", "type"]:
        raise AssertionError(f"noncanonical target dependency: {dependency!r}")
    if dependency["name"] != expected_dependency_target:
        raise AssertionError(f"unexpected target dependency: {dependency!r}")
    if dependency["type"] != "OBJECT_LIBRARY":
        raise AssertionError(f"unexpected dependency target type: {dependency!r}")
    if dependency["archive"] is not None or dependency["link"] is not None:
        raise AssertionError("identity object dependency unexpectedly archives or links")
    dependency_sources, dependency_languages = validate_compile_groups(
        dependency["compile_groups"], label="dependency"
    )
    used_languages.update(dependency_languages)
    if not any(source.endswith("/deac/src/build_identity.cpp") for source in dependency_sources):
        raise AssertionError("receipt does not bind the embedded identity object source")

    link = receipt["link"]
    if list(link) != ["command_fragments", "language", "lto", "sysroot"]:
        raise AssertionError(f"noncanonical link receipt: {link!r}")
    assert_string(link["language"], "link language")
    used_languages.add(link["language"])
    if not isinstance(link["lto"], bool) or not isinstance(
        link["command_fragments"], list
    ):
        raise TypeError("link receipt has invalid types")

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


def main():
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
    parser.add_argument("--build-config")
    args = parser.parse_args()

    exe = Path(args.exe).resolve()
    receipt_path = Path(args.receipt).resolve()
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
        build_dir=Path(args.build_dir),
        source_dir=Path(args.source_dir),
        expected_backend=args.expected_backend,
        expected_target=args.expected_target,
        expected_dependency_target=args.expected_dependency_target,
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
    if installed_receipt.stdout != first.stdout or installed_receipt.stderr:
        raise AssertionError("installed executable did not retain its embedded receipt")


if __name__ == "__main__":
    main()
