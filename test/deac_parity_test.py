#!/usr/bin/env python3
import argparse
import math
import re
import shutil
import struct
import subprocess
from pathlib import Path


def write_doubles(path, values):
    path.write_bytes(struct.pack("<" + "d" * len(values), *values))


def read_doubles(path):
    data = path.read_bytes()
    if len(data) % 8 != 0:
        raise AssertionError(f"{path} does not contain a whole number of doubles")
    return struct.unpack("<" + "d" * (len(data) // 8), data)


def run_command(command, workdir):
    result = subprocess.run(
        command,
        cwd=workdir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"expected command to succeed, got {result.returncode}\n"
            f"command: {' '.join(map(str, command))}\n"
            f"output:\n{result.stdout}"
        )
    if "minimum_fitness:" not in result.stdout:
        raise AssertionError(
            f"expected command output to contain minimum_fitness\n"
            f"command: {' '.join(map(str, command))}\n"
            f"output:\n{result.stdout}"
        )
    return result


def run_deac(exe, fixture, frequency_file, workdir, seed):
    workdir.mkdir(parents=True)
    save_dir = workdir / "results"
    command = [
        exe,
        "-T",
        "1.0",
        "-N",
        "2",
        "-P",
        "8",
        "-M",
        "4",
        "--frequency_file",
        str(frequency_file),
        "--save_directory",
        str(save_dir),
        "--seed",
        seed,
        "--uuid",
        seed,
        "--stop_minimum_fitness",
        # Stop before mutation; CPU and GPU use different parallel RNG streams after initialization.
        "1e300",
        "--track_stats",
        str(fixture),
    ]
    run_command(command, workdir)
    return save_dir


def assert_close_sequence(name, reference, candidate, absolute_tolerance, relative_tolerance):
    if len(reference) != len(candidate):
        raise AssertionError(
            f"{name} length mismatch: reference has {len(reference)}, "
            f"candidate has {len(candidate)}"
        )

    max_abs_diff = 0.0
    max_rel_diff = 0.0
    for idx, (expected, actual) in enumerate(zip(reference, candidate)):
        if not math.isfinite(expected) or not math.isfinite(actual):
            raise AssertionError(
                f"{name}[{idx}] is not finite: reference={expected}, candidate={actual}"
            )
        abs_diff = abs(expected - actual)
        rel_diff = abs_diff / max(abs(expected), abs(actual), 1.0)
        max_abs_diff = max(max_abs_diff, abs_diff)
        max_rel_diff = max(max_rel_diff, rel_diff)
        if abs_diff > absolute_tolerance and rel_diff > relative_tolerance:
            raise AssertionError(
                f"{name}[{idx}] mismatch: reference={expected}, candidate={actual}, "
                f"abs_diff={abs_diff}, rel_diff={rel_diff}, "
                f"max_abs_diff={max_abs_diff}, max_rel_diff={max_rel_diff}"
            )


def read_log_value(log_path, key):
    pattern = re.compile(rf"^{re.escape(key)}: (.+)$", re.MULTILINE)
    match = pattern.search(log_path.read_text())
    if not match:
        raise AssertionError(f"expected {log_path} to contain {key}")
    return match.group(1)


def assert_outputs_match(reference_dir, candidate_dir, seed, absolute_tolerance, relative_tolerance):
    prefix = "deac-spfsf"
    for suffix in [
        "frequency",
        "dsf",
        "stats_fitness-mean",
        "stats_fitness-minimum",
        "stats_fitness-squared-mean",
    ]:
        reference = read_doubles(reference_dir / f"{prefix}_{suffix}_{seed}.bin")
        candidate = read_doubles(candidate_dir / f"{prefix}_{suffix}_{seed}.bin")
        assert_close_sequence(
            suffix,
            reference,
            candidate,
            absolute_tolerance,
            relative_tolerance,
        )

    reference_log = reference_dir / f"{prefix}_log_{seed}.dat"
    candidate_log = candidate_dir / f"{prefix}_log_{seed}.dat"
    reference_generation = read_log_value(reference_log, "generation")
    candidate_generation = read_log_value(candidate_log, "generation")
    if reference_generation != candidate_generation:
        raise AssertionError(
            f"generation mismatch: reference={reference_generation}, "
            f"candidate={candidate_generation}"
        )

    reference_fitness = float(read_log_value(reference_log, "minimum_fitness"))
    candidate_fitness = float(read_log_value(candidate_log, "minimum_fitness"))
    assert_close_sequence(
        "minimum_fitness",
        [reference_fitness],
        [candidate_fitness],
        absolute_tolerance,
        relative_tolerance,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-exe", required=True)
    parser.add_argument("--candidate-exe", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-9)
    parser.add_argument("--relative-tolerance", type=float, default=1e-9)
    args = parser.parse_args()

    reference_exe = str(Path(args.reference_exe))
    candidate_exe = str(Path(args.candidate_exe))
    for exe in [reference_exe, candidate_exe]:
        if not Path(exe).is_file():
            raise AssertionError(f"expected executable {exe} to exist")

    workdir = Path(args.workdir)
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)

    fixture = workdir / "tiny-isf.bin"
    write_doubles(
        fixture,
        [0.0, 0.2, 0.4, 0.6]
        + [-1.0, -0.85, -0.72, -0.61]
        + [0.05, 0.05, 0.05, 0.05],
    )
    frequency_file = workdir / "frequency.bin"
    write_doubles(frequency_file, [0.0, 0.7, 1.8, 3.0])

    seed = "17"
    reference_dir = run_deac(reference_exe, fixture, frequency_file, workdir / "reference", seed)
    candidate_dir = run_deac(candidate_exe, fixture, frequency_file, workdir / "candidate", seed)
    assert_outputs_match(
        reference_dir,
        candidate_dir,
        seed,
        args.absolute_tolerance,
        args.relative_tolerance,
    )


if __name__ == "__main__":
    main()
