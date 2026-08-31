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


def run_zero_target(
    exe,
    workdir,
    fixture,
    seed,
    detailed_balance,
    zero_temperature,
    run_label=None,
    first_moment_error=None,
):
    save_dir = workdir / f"results-{run_label or seed}"
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
        "--first_moment",
        "0",
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
    if first_moment_error is not None:
        command.extend(["--first_moment_error", first_moment_error])
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
            f"zero first-moment run failed with {result.returncode}\n"
            f"command: {' '.join(map(str, command))}\noutput:\n{result.stdout}"
        )
    if "generation: 1" not in result.stdout:
        raise AssertionError(
            f"seed {seed} did not complete the evolved-population step:\n"
            f"{result.stdout}"
        )
    prefix = (
        "deac-zT"
        if zero_temperature
        else "deac-bdsf"
        if detailed_balance
        else "deac-spfsf"
    )
    minima = read_doubles(save_dir / f"{prefix}_stats_fitness-minimum_{seed}.bin")
    if len(minima) != 2 or any(not math.isfinite(value) for value in minima):
        raise AssertionError(
            f"seed {seed} produced non-finite initial/evolved fitness: {minima}"
        )
    log_text = (save_dir / f"{prefix}_log_{seed}.dat").read_text()
    if "first_moment: 0\n" not in log_text:
        raise AssertionError("zero first moment was not recorded as active")
    effective_error = 1.0 if first_moment_error is None else float(first_moment_error)
    error_lines = [
        line
        for line in log_text.splitlines()
        if line.startswith("first_moment_error: ")
    ]
    if len(error_lines) != 1 or float(error_lines[0].split(": ", 1)[1]) != effective_error:
        raise AssertionError(
            "effective first-moment uncertainty was not recorded exactly once: "
            f"expected={effective_error}, lines={error_lines}"
        )
    binary_artifacts = {
        path.name: path.read_bytes() for path in sorted(save_dir.glob("*.bin"))
    }
    return minima[1] < minima[0], minima, binary_artifacts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--detailed-balance", action="store_true")
    parser.add_argument("--zero-temperature", action="store_true")
    parser.add_argument(
        "--seeds",
        nargs="+",
        default=("3", "6", "14", "17", "23", "47"),
    )
    args = parser.parse_args()

    exe = str(Path(args.exe))
    if not Path(exe).is_file():
        raise AssertionError(f"expected executable {exe} to exist")
    workdir = Path(args.workdir)
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)

    tau = [0.0, 0.2, 0.4, 0.6]
    observed = (
        [1.0, 0.85, 0.72, 0.61]
        if args.zero_temperature
        else [-1.0, -0.85, -0.72, -0.61]
    )
    fixture = workdir / "tiny-isf.bin"
    write_doubles(fixture, tau + observed + [0.05, 0.05, 0.05, 0.05])

    accepted_improved_trial = False
    first_minima = None
    first_artifacts = None
    unit_runs = {}
    for seed in args.seeds:
        improved, minima, artifacts = run_zero_target(
            exe,
            workdir,
            fixture,
            seed,
            args.detailed_balance,
            args.zero_temperature,
        )
        accepted_improved_trial |= improved
        unit_runs[seed] = (minima, artifacts)
        if first_minima is None:
            first_minima = minima
            first_artifacts = artifacts
    if not accepted_improved_trial:
        raise AssertionError(
            "zero first-moment regression did not accept an improved evolved trial"
        )

    _, repeated_minima, repeated_artifacts = run_zero_target(
        exe,
        workdir,
        fixture,
        args.seeds[0],
        args.detailed_balance,
        args.zero_temperature,
        run_label=f"{args.seeds[0]}-repeat",
    )
    if repeated_minima != first_minima or repeated_artifacts != first_artifacts:
        raise AssertionError(
            "zero first-moment fitness was not deterministic for a fixed seed: "
            f"first={first_minima}, repeated={repeated_minima}"
        )

    _, explicit_unit_minima, explicit_unit_artifacts = run_zero_target(
        exe,
        workdir,
        fixture,
        args.seeds[0],
        args.detailed_balance,
        args.zero_temperature,
        run_label=f"{args.seeds[0]}-explicit-unit",
        first_moment_error="1.0",
    )
    if (
        explicit_unit_minima != first_minima
        or explicit_unit_artifacts != first_artifacts
    ):
        raise AssertionError(
            "omitting --first_moment_error did not preserve the unit-error "
            "fitness and binary artifacts"
        )

    scaled_improved_trial = False
    for seed in args.seeds:
        improved, scaled_minima, _ = run_zero_target(
            exe,
            workdir,
            fixture,
            seed,
            args.detailed_balance,
            args.zero_temperature,
            run_label=f"{seed}-scaled",
            first_moment_error="0.25",
        )
        if not improved:
            continue
        scaled_improved_trial = True
        unit_minima, _ = unit_runs[seed]
        if scaled_minima[0] == unit_minima[0]:
            raise AssertionError(
                "custom first-moment uncertainty did not affect initial fitness"
            )
        if scaled_minima[1] == unit_minima[1]:
            raise AssertionError(
                "custom first-moment uncertainty did not affect accepted evolved fitness"
            )
    if not scaled_improved_trial:
        raise AssertionError(
            "custom first-moment uncertainty regression did not accept an "
            "improved evolved trial"
        )


if __name__ == "__main__":
    main()
