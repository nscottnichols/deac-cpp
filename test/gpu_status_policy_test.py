import argparse
import re
from pathlib import Path

STATUS_MACRO = re.compile(
    r"^\s*#\s*define\s+(GPU_ASSERT|GPU_BLAS_ASSERT)\s*\(", re.MULTILINE
)
STANDARD_ASSERT = re.compile(r"\bassert\s*\(")
ESSENTIAL_BACKEND_CALL = re.compile(
    r"\b(?:cuda|hip|cublas|hipblas)[A-Za-z0-9_]*\s*\("
    r"|\bdeac_(?:stream_create|stream_destroy|malloc_device|"
    r"memcpy_host_to_device|memcpy_device_to_host|wait|memset|free|"
    r"create_blas_handle|destroy_blas_handle|set_stream)\s*\("
)
SOURCE_SUFFIXES = {".cpp", ".cu", ".cuh", ".h", ".hpp"}


def logical_lines(text):
    pending = ""
    for physical_line in text.splitlines():
        pending += physical_line
        if physical_line.rstrip().endswith("\\"):
            pending += "\n"
            continue
        yield pending
        pending = ""
    if pending:
        yield pending


def check_status_macros(common_gpu):
    common_gpu_text = common_gpu.read_text()
    definitions = [
        line
        for line in logical_lines(common_gpu_text)
        if STATUS_MACRO.match(line)
    ]
    errors = []
    if STANDARD_ASSERT.search(common_gpu_text):
        errors.append(f"{common_gpu}: standard assert is forbidden")
    for macro_name in ("GPU_ASSERT", "GPU_BLAS_ASSERT"):
        matching = [
            definition
            for definition in definitions
            if STATUS_MACRO.match(definition).group(1) == macro_name
        ]
        checked = [
            definition
            for definition in matching
            if "DEAC_GPU_STATUS_CHECK" in definition
        ]
        if len(matching) != 3 or len(checked) != 2:
            errors.append(
                f"{common_gpu}: expected two checked CUDA/HIP and one "
                f"SYCL {macro_name} definitions, got {len(checked)} checked "
                f"of {len(matching)} total"
            )
        expected_category = "runtime" if macro_name == "GPU_ASSERT" else "BLAS"
        for backend in ("CUDA", "HIP"):
            backend_definitions = [
                definition
                for definition in checked
                if f'"{backend}"' in definition
                and f'"{expected_category}"' in definition
            ]
            if len(backend_definitions) != 1:
                errors.append(
                    f"{common_gpu}: expected one checked {backend} "
                    f"{expected_category} {macro_name} definition"
                )
        for definition in matching:
            if STANDARD_ASSERT.search(definition):
                errors.append(
                    f"{common_gpu}: {macro_name} delegates to standard assert"
                )
    return errors


def check_asserted_backend_calls(source_root):
    errors = []
    for search_root in (source_root / "src" / "deac", source_root / "test"):
        for path in sorted(search_root.rglob("*")):
            if not path.is_file() or path.suffix not in SOURCE_SUFFIXES:
                continue
            text = path.read_text()
            for match in STANDARD_ASSERT.finditer(text):
                # An assert expression ends at its statement semicolon.  The
                # bounded fallback also covers a macro with no semicolon.
                statement_end = text.find(";", match.end())
                if statement_end < 0 or statement_end - match.start() > 2000:
                    statement_end = min(len(text), match.start() + 2000)
                expression = text[match.start():statement_end]
                if ESSENTIAL_BACKEND_CALL.search(expression):
                    line = text.count("\n", 0, match.start()) + 1
                    errors.append(
                        f"{path}:{line}: essential backend call is inside "
                        "standard assert"
                    )
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    args = parser.parse_args()

    source_root = Path(args.source_root).resolve()
    common_gpu = source_root / "src" / "deac" / "include" / "common_gpu.hpp"
    gpu_status = source_root / "src" / "deac" / "include" / "gpu_status.hpp"
    errors = check_status_macros(common_gpu)
    if STANDARD_ASSERT.search(gpu_status.read_text()):
        errors.append(f"{gpu_status}: standard assert is forbidden")
    errors.extend(check_asserted_backend_calls(source_root))
    if errors:
        raise SystemExit("\n".join(errors))
    print("CUDA/HIP status-check source policy passed")


if __name__ == "__main__":
    main()
