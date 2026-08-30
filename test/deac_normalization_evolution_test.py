#!/usr/bin/env python3
import argparse
import math
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


def trapezoidal_weights(coordinates):
    weights = [0.0] * len(coordinates)
    weights[0] = 0.5 * (coordinates[1] - coordinates[0])
    for index in range(1, len(coordinates) - 1):
        weights[index] = 0.5 * (coordinates[index + 1] - coordinates[index - 1])
    weights[-1] = 0.5 * (coordinates[-1] - coordinates[-2])
    return weights


def run_normalized_evolution(
    exe,
    workdir,
    fixture,
    frequency_file,
    frequency,
    observed_at_zero,
    seed,
    detailed_balance,
    zero_temperature,
):
    save_dir = workdir / f"results-{seed}"
    command = [
        exe,
        "--temperature",
        "0.0" if zero_temperature else "1.0",
        "--number_of_generations",
        "2",
        "--population_size",
        "4",
        "--frequency_file",
        str(frequency_file),
        "--normalize",
        "--crossover_probability",
        "1",
        "--self_adapting_crossover_probability",
        "0",
        "--differential_weight",
        "2",
        "--self_adapting_differential_weight_probability",
        "0",
        "--stop_minimum_fitness",
        "-1",
        "--track_stats",
        "--save_directory",
        str(save_dir),
        "--seed",
        seed,
        "--uuid",
        seed,
    ]
    if not detailed_balance and not zero_temperature:
        command.extend(["--spectra_type", "bfull"])
    command.append(str(fixture))

    result = subprocess.run(
        command,
        cwd=workdir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"normalization evolution run failed with {result.returncode}\n"
            f"command: {' '.join(map(str, command))}\noutput:\n{result.stdout}"
        )
    if "generation: 1" not in result.stdout:
        raise AssertionError(
            f"seed {seed} did not complete the requested evolution step:\n"
            f"{result.stdout}"
        )

    if zero_temperature:
        prefix = "deac-zT"
    elif detailed_balance:
        prefix = "deac-bdsf"
    else:
        prefix = "deac-bfull"
    output_frequency = read_doubles(save_dir / f"{prefix}_frequency_{seed}.bin")
    output_spectrum = read_doubles(save_dir / f"{prefix}_dsf_{seed}.bin")
    fitness_minimum = read_doubles(
        save_dir / f"{prefix}_stats_fitness-minimum_{seed}.bin"
    )
    if len(output_frequency) != len(output_spectrum):
        raise AssertionError(
            f"frequency/spectrum size mismatch for seed {seed}: "
            f"{len(output_frequency)} != {len(output_spectrum)}"
        )
    expected_frequency = (
        tuple(frequency)
        if detailed_balance or zero_temperature
        else (-frequency[-1], -frequency[0], frequency[-1])
    )
    if output_frequency != expected_frequency:
        raise AssertionError(
            f"seed {seed} output frequency mismatch: "
            f"expected {expected_frequency}, got {output_frequency}"
        )
    if any(not math.isfinite(value) for value in output_spectrum):
        raise AssertionError(f"seed {seed} produced an invalid spectrum")
    if len(fitness_minimum) != 2 or any(
        not math.isfinite(value) for value in fitness_minimum
    ):
        raise AssertionError(
            f"seed {seed} produced invalid fitness minima: {fitness_minimum}"
        )

    weights = trapezoidal_weights(frequency)
    if zero_temperature:
        normalization = sum(
            weight * spectrum
            for weight, spectrum in zip(weights, output_spectrum)
        )
    elif detailed_balance:
        beta = 1.0
        normalization = sum(
            weight * spectrum * (1.0 + math.exp(-beta * omega))
            for weight, omega, spectrum in zip(
                weights, output_frequency, output_spectrum
            )
        )
    else:
        genome_size = len(frequency)
        normalization = 0.0
        for index, weight in enumerate(weights):
            positive_index = genome_size + index - 1
            negative_index = genome_size - index - 1
            normalization += weight * (
                output_spectrum[positive_index]
                + output_spectrum[negative_index]
            )

    if not math.isclose(
        normalization, observed_at_zero, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise AssertionError(
            f"seed {seed} evolved spectrum violated --normalize: "
            f"expected {observed_at_zero}, got {normalization}"
        )
    return fitness_minimum[1] < fitness_minimum[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--detailed-balance", action="store_true")
    parser.add_argument("--zero-temperature", action="store_true")
    parser.add_argument("--seeds", nargs="+", default=("2", "3", "4", "34"))
    args = parser.parse_args()

    exe = str(Path(args.exe))
    if not Path(exe).is_file():
        raise AssertionError(f"expected executable {exe} to exist")

    workdir = Path(args.workdir)
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)

    tau = [0.0, 0.2, 0.4, 0.6]
    observed = [1.0, 0.85, 0.72, 0.61]
    errors = [0.05, 0.05, 0.05, 0.05]
    frequency = [0.0, 3.0]
    fixture = workdir / "positive-isf.bin"
    frequency_file = workdir / "frequency.bin"
    write_doubles(fixture, tau + observed + errors)
    write_doubles(frequency_file, frequency)

    # Different model/backend RNG streams accept different trials. Across this
    # fixed seed set at least one final best member comes from the evolution step.
    accepted_improved_trial = False
    for seed in args.seeds:
        accepted_improved_trial |= run_normalized_evolution(
            exe,
            workdir,
            fixture,
            frequency_file,
            frequency,
            observed[0],
            seed,
            args.detailed_balance,
            args.zero_temperature,
        )
    if not accepted_improved_trial:
        raise AssertionError(
            "normalization regression did not accept an improved evolved trial"
        )


if __name__ == "__main__":
    main()
