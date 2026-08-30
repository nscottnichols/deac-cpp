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
    if len(data) % 8:
        raise AssertionError(f"{path} does not contain whole doubles")
    return struct.unpack("<" + "d" * (len(data) // 8), data)


def run_solver(
    exe,
    workdir,
    fixture,
    frequency_file,
    save_dir,
    uuid,
    stop_minimum_fitness,
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
        stop_minimum_fitness,
        "--track_stats",
        "--save_directory",
        str(save_dir),
        "--seed",
        "17",
        "--uuid",
        uuid,
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
    if result.returncode:
        raise AssertionError(
            f"solver failed with {result.returncode}\n"
            f"command: {' '.join(command)}\noutput:\n{result.stdout}"
        )
    return result


def prefix(detailed_balance, zero_temperature):
    if zero_temperature:
        return "deac-zT"
    return "deac-bdsf" if detailed_balance else "deac-bfull"


def assert_normalized(spectrum, detailed_balance, zero_temperature):
    weights = (1.5, 1.5)
    if zero_temperature:
        moment = sum(weight * value for weight, value in zip(weights, spectrum))
    elif detailed_balance:
        frequency = (0.0, 3.0)
        moment = sum(
            weight * value * (1.0 + math.exp(-omega))
            for weight, value, omega in zip(weights, spectrum, frequency)
        )
    else:
        # Two-sided bfull output is ordered [-3, 0, 3].  The zero-frequency
        # value participates in both population halves in the solver's rule.
        moment = weights[0] * (spectrum[1] + spectrum[1])
        moment += weights[1] * (spectrum[2] + spectrum[0])
    if not math.isclose(moment, 1.0, rel_tol=1e-12, abs_tol=1e-12):
        raise AssertionError(f"incumbent lost normalization: {moment}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-exe", required=True)
    parser.add_argument("--forced-exe", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--detailed-balance", action="store_true")
    parser.add_argument("--zero-temperature", action="store_true")
    args = parser.parse_args()

    workdir = Path(args.workdir)
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    fixture = workdir / "positive-isf.bin"
    frequency_file = workdir / "frequency.bin"
    write_doubles(
        fixture,
        [0.0, 0.2, 0.4, 0.6]
        + [1.0, 0.85, 0.72, 0.61]
        + [0.05, 0.05, 0.05, 0.05],
    )
    write_doubles(frequency_file, [0.0, 3.0])

    initial_dir = workdir / "initial"
    forced_dir = workdir / "forced"
    run_solver(
        args.reference_exe,
        workdir,
        fixture,
        frequency_file,
        initial_dir,
        "initial",
        "1e300",
        args.detailed_balance,
        args.zero_temperature,
    )
    forced = run_solver(
        args.forced_exe,
        workdir,
        fixture,
        frequency_file,
        forced_dir,
        "forced",
        "-1",
        args.detailed_balance,
        args.zero_temperature,
    )
    if "generation: 1" not in forced.stdout:
        raise AssertionError(f"forced run did not complete evolution:\n{forced.stdout}")
    if "test_forced_invalid_normalization_trials: 4" not in forced.stdout:
        raise AssertionError(
            f"forced run did not reach the evolved normalization path:\n{forced.stdout}"
        )
    if "test_invalid_normalization_fitness: DBL_MAX" not in forced.stdout:
        raise AssertionError(
            f"forced rows did not receive DBL_MAX fitness:\n{forced.stdout}"
        )

    output_prefix = prefix(args.detailed_balance, args.zero_temperature)
    initial_spectrum_path = initial_dir / f"{output_prefix}_dsf_initial.bin"
    forced_spectrum_path = forced_dir / f"{output_prefix}_dsf_forced.bin"
    if initial_spectrum_path.read_bytes() != forced_spectrum_path.read_bytes():
        raise AssertionError("invalid evolved trials changed the incumbent spectrum")
    initial_frequency_path = initial_dir / f"{output_prefix}_frequency_initial.bin"
    forced_frequency_path = forced_dir / f"{output_prefix}_frequency_forced.bin"
    if initial_frequency_path.read_bytes() != forced_frequency_path.read_bytes():
        raise AssertionError("invalid evolved trials changed the output grid")

    spectrum = read_doubles(forced_spectrum_path)
    if any(not math.isfinite(value) for value in spectrum):
        raise AssertionError(
            f"invalid evolved rows contaminated the spectrum: {spectrum}"
        )
    assert_normalized(spectrum, args.detailed_balance, args.zero_temperature)

    for stat_name in ("fitness-mean", "fitness-minimum", "fitness-squared-mean"):
        initial_stats = read_doubles(
            initial_dir
            / f"{output_prefix}_stats_{stat_name}_initial.bin"
        )
        forced_stats = read_doubles(
            forced_dir
            / f"{output_prefix}_stats_{stat_name}_forced.bin"
        )
        if len(initial_stats) != 2 or len(forced_stats) != 2:
            raise AssertionError(
                f"unexpected {stat_name} lengths: {initial_stats}, {forced_stats}"
            )
        if initial_stats != (initial_stats[0], initial_stats[0]):
            raise AssertionError(
                f"stop-before-evolution reference changed {stat_name}: "
                f"{initial_stats}"
            )
        if forced_stats != initial_stats:
            raise AssertionError(
                f"invalid trials contaminated {stat_name}: "
                f"initial={initial_stats}, forced={forced_stats}"
            )
        if any(not math.isfinite(value) for value in forced_stats):
            raise AssertionError(f"non-finite {stat_name}: {forced_stats}")


if __name__ == "__main__":
    main()
