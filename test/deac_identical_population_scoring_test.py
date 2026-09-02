#!/usr/bin/env python3
import argparse
import shutil
import struct
import subprocess
from pathlib import Path

MARKER = "test_identical_population_scoring: exact 4"
EVOLVED_MARKERS = (
    "test_forced_invalid_normalization_trials:",
    "test_invalid_normalization_fitness:",
    "test_poisoned_gpu_fitness_evolved:",
)


def write_doubles(path, values):
    path.write_bytes(struct.pack("<" + "d" * len(values), *values))


def run_solver(
    exe,
    workdir,
    fixture,
    save_dir,
    uuid,
    detailed_balance,
    zero_temperature,
):
    command = [
        exe,
        "--temperature",
        "0" if zero_temperature else "1",
        "--number_of_generations",
        "2",
        "--population_size",
        "4",
        "--genome_size",
        "8",
        "--omega_max",
        "4",
        "--normalize",
        "--first_moment",
        "0.5",
        "--first_moment_error",
        "0.25",
        "--third_moment",
        "0.5",
        "--third_moment_error",
        "0.1",
        "--crossover_probability",
        "0.9",
        "--self_adapting_crossover_probability",
        "0.1",
        "--differential_weight",
        "0.9",
        "--self_adapting_differential_weight_probability",
        "0.1",
        "--stop_minimum_fitness",
        "1e300",
        "--save_directory",
        str(save_dir),
        "--seed",
        "17",
        "--uuid",
        uuid,
    ]
    if detailed_balance:
        command.append("--use_negative_first_moment")
    elif not zero_temperature:
        command.extend(["--spectra_type", "bfull"])
    command.append(str(fixture))
    result = subprocess.run(
        command,
        cwd=workdir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"solver failed with {result.returncode}\n"
            f"command: {' '.join(map(str, command))}\n"
            f"output:\n{result.stdout}"
        )
    return result.stdout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-exe", required=True)
    parser.add_argument("--seam-exe", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--detailed-balance", action="store_true")
    parser.add_argument("--zero-temperature", action="store_true")
    args = parser.parse_args()

    workdir = Path(args.workdir)
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)

    fixture = workdir / "positive-isf.bin"
    write_doubles(
        fixture,
        [0.0, 0.2, 0.4, 0.6]
        + [1.0, 0.85, 0.72, 0.61]
        + [0.05, 0.05, 0.05, 0.05],
    )

    reference_output = run_solver(
        args.reference_exe,
        workdir,
        fixture,
        workdir / "reference",
        "reference",
        args.detailed_balance,
        args.zero_temperature,
    )
    seam_output = run_solver(
        args.seam_exe,
        workdir,
        fixture,
        workdir / "seam",
        "seam",
        args.detailed_balance,
        args.zero_temperature,
    )

    if MARKER in reference_output:
        raise AssertionError(
            "production solver unexpectedly enabled the identical-population "
            f"seam:\n{reference_output}"
        )
    if seam_output.count(MARKER) != 1:
        raise AssertionError(
            "test helper did not prove exact initial/evolved scoring once:\n"
            f"{seam_output}"
        )
    reached_evolved_loop = [
        marker for marker in EVOLVED_MARKERS if marker in seam_output
    ]
    if reached_evolved_loop:
        raise AssertionError(
            "identical-population seam did not stop before mutation/evolution: "
            f"markers={reached_evolved_loop}\n{seam_output}"
        )


if __name__ == "__main__":
    main()
