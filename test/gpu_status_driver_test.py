import argparse
import re
import subprocess

SUCCESS_MARKER = (
    "success checks passed; runtime_evaluations=1; blas_evaluations=1\n"
)
FAILURE_MARKER = (
    "controlled failure checks passed; runtime_evaluations=1; "
    "blas_evaluations=1\n"
)


def run(executable, *arguments):
    return subprocess.run(
        [executable, *arguments],
        capture_output=True,
        text=True,
        check=False,
    )


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", required=True)
    args = parser.parse_args()

    success = run(args.exe)
    require(success.returncode == 0, f"success return code: {success.returncode}")
    require(success.stdout == SUCCESS_MARKER, f"success stdout: {success.stdout!r}")
    require(success.stderr == "", f"success stderr: {success.stderr!r}")

    failure = run(args.exe, "--failure")
    require(failure.returncode == 1, f"failure return code: {failure.returncode}")
    require(failure.stdout == "", f"failure stdout: {failure.stdout!r}")
    require(
        failure.stderr.count(FAILURE_MARKER) == 1,
        f"controlled failure marker count: {failure.stderr!r}",
    )
    require("did not throw" not in failure.stderr, failure.stderr)

    runtime_prefix = (
        "FAKE runtime call failed: counted_status(evaluations, failure); "
        "status=17 (controlled runtime failure); location="
    )
    blas_prefix = (
        "FAKE BLAS call failed: counted_status(evaluations, failure); "
        "status=29; location="
    )
    require(failure.stderr.count(runtime_prefix) == 1, failure.stderr)
    require(failure.stderr.count(blas_prefix) == 1, failure.stderr)
    locations = re.findall(r"gpu_status_test\.cpp:[0-9]+", failure.stderr)
    require(len(locations) == 2, failure.stderr)
    require(failure.stderr.endswith(FAILURE_MARKER), failure.stderr)
    print("Debug/NDEBUG GPU status-check driver passed")


if __name__ == "__main__":
    main()
