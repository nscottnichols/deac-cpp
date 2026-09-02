#!/usr/bin/env python3
import argparse
import math
import shutil
import struct
import subprocess
from pathlib import Path

NUMBER_OF_GENERATIONS = 2
POPULATION_SIZE = 4


def write_doubles(path, values):
    path.write_bytes(struct.pack("<" + "d" * len(values), *values))


def read_doubles(path):
    data = path.read_bytes()
    if len(data) % 8 != 0:
        raise AssertionError(f"{path} does not contain a whole number of doubles")
    return struct.unpack("<" + "d" * (len(data) // 8), data)


def reduction_slack(*values):
    return 64 * math.ulp(max([1.0, *(abs(value) for value in values)]))


def materially_less(value, reference):
    return value < reference - reduction_slack(value, reference)


def read_fitness_statistics(save_dir, prefix, seed):
    statistics = {
        "minimum": read_doubles(
            save_dir / f"{prefix}_stats_fitness-minimum_{seed}.bin"
        ),
        "mean": read_doubles(save_dir / f"{prefix}_stats_fitness-mean_{seed}.bin"),
        "squared_mean": read_doubles(
            save_dir / f"{prefix}_stats_fitness-squared-mean_{seed}.bin"
        ),
    }
    for name, values in statistics.items():
        if len(values) != NUMBER_OF_GENERATIONS:
            raise AssertionError(
                f"{name} must contain initial and evolved values: {values}"
            )
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise AssertionError(
                f"{name} contains a non-finite or negative fitness: {values}"
            )
    return statistics


def assert_power_of_two_scaling(unit_statistics, scaled_statistics):
    scales = {"minimum": 16.0, "mean": 16.0, "squared_mean": 256.0}
    for name, scale in scales.items():
        for generation, (unit_value, scaled_value) in enumerate(
            zip(unit_statistics[name], scaled_statistics[name])
        ):
            expected = unit_value * scale
            if abs(scaled_value - expected) > reduction_slack(
                scaled_value, expected
            ):
                raise AssertionError(
                    "first-moment uncertainty did not scale fitness exactly: "
                    f"statistic={name}, generation={generation}, "
                    f"unit={unit_value}, scaled={scaled_value}, "
                    f"expected={expected}"
                )


def published_spectrum_artifacts(artifacts):
    selected = {
        name: data
        for name, data in artifacts.items()
        if "_dsf_" in name or "_frequency_" in name
    }
    if len(selected) != 2:
        raise AssertionError(
            "expected exactly one DSF and frequency artifact: "
            f"{sorted(selected)}"
        )
    return selected


def run_zero_target(
    exe,
    workdir,
    fixture,
    seed,
    detailed_balance,
    zero_temperature,
    run_label=None,
    first_moment_error=None,
    use_first_moment=True,
):
    save_dir = workdir / f"results-{run_label or seed}"
    command = [
        exe,
        "--temperature",
        "0" if zero_temperature else "1",
        "--number_of_generations",
        str(NUMBER_OF_GENERATIONS),
        "--population_size",
        str(POPULATION_SIZE),
        "--genome_size",
        "8",
        "--omega_max",
        "4",
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
    if use_first_moment:
        command.extend(["--first_moment", "0"])
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
    statistics = read_fitness_statistics(save_dir, prefix, seed)
    log_text = (save_dir / f"{prefix}_log_{seed}.dat").read_text()
    if use_first_moment and "first_moment: 0\n" not in log_text:
        raise AssertionError("zero first moment was not recorded as active")
    effective_error = 1.0 if first_moment_error is None else float(first_moment_error)
    error_lines = [
        line
        for line in log_text.splitlines()
        if line.startswith("first_moment_error: ")
    ]
    if use_first_moment and (
        len(error_lines) != 1
        or float(error_lines[0].split(": ", 1)[1]) != effective_error
    ):
        raise AssertionError(
            "effective first-moment uncertainty was not recorded exactly once: "
            f"expected={effective_error}, lines={error_lines}"
        )
    if not use_first_moment and error_lines:
        raise AssertionError(
            "inactive first-moment control recorded an uncertainty: "
            f"{error_lines}"
        )
    binary_artifacts = {
        path.name: path.read_bytes() for path in sorted(save_dir.glob("*.bin"))
    }
    return statistics, binary_artifacts


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

    first_seed = args.seeds[0]
    first_statistics, first_artifacts = run_zero_target(
        exe,
        workdir,
        fixture,
        first_seed,
        args.detailed_balance,
        args.zero_temperature,
    )

    repeated_statistics, repeated_artifacts = run_zero_target(
        exe,
        workdir,
        fixture,
        first_seed,
        args.detailed_balance,
        args.zero_temperature,
        run_label=f"{first_seed}-repeat",
    )
    if (
        repeated_statistics != first_statistics
        or repeated_artifacts != first_artifacts
    ):
        raise AssertionError(
            "zero first-moment fitness was not deterministic for a fixed seed: "
            f"first={first_statistics}, repeated={repeated_statistics}"
        )

    explicit_unit_statistics, explicit_unit_artifacts = run_zero_target(
        exe,
        workdir,
        fixture,
        first_seed,
        args.detailed_balance,
        args.zero_temperature,
        run_label=f"{first_seed}-explicit-unit",
        first_moment_error="1.0",
    )
    if (
        explicit_unit_statistics != first_statistics
        or explicit_unit_artifacts != first_artifacts
    ):
        raise AssertionError(
            "omitting --first_moment_error did not preserve the unit-error "
            "fitness and binary artifacts"
        )

    # With this exact power-of-two finite data error, every data-residual
    # contribution underflows to positive zero. An explicit data-only control
    # below verifies that isolation before the moment-weight assertions.
    # Fitness is therefore only the first-moment penalty.
    # Changing its error from 1.0 to 0.25 multiplies every candidate fitness by
    # exactly 16 (and squared-fitness statistics by 256), so selection and the
    # published best-spectrum artifacts must remain identical. This gives a
    # deterministic initial/evolved-path check without relying on a particular
    # row becoming the global minimum.
    penalty_fixture = workdir / "first-moment-only.bin"
    write_doubles(
        penalty_fixture,
        tau + observed + [math.ldexp(1.0, 1023)] * len(tau),
    )

    accepted_evolved_trial = False
    for seed in args.seeds:
        control_statistics, _ = run_zero_target(
            exe,
            workdir,
            penalty_fixture,
            seed,
            args.detailed_balance,
            args.zero_temperature,
            run_label=f"{seed}-data-only-control",
            use_first_moment=False,
        )
        if any(
            value != 0.0
            for values in control_statistics.values()
            for value in values
        ):
            raise AssertionError(
                "data-only isolation control produced nonzero fitness: "
                f"seed={seed}, statistics={control_statistics}"
            )
        unit_statistics, unit_artifacts = run_zero_target(
            exe,
            workdir,
            penalty_fixture,
            seed,
            args.detailed_balance,
            args.zero_temperature,
            run_label=f"{seed}-penalty-unit",
            first_moment_error="1.0",
        )
        scaled_statistics, scaled_artifacts = run_zero_target(
            exe,
            workdir,
            penalty_fixture,
            seed,
            args.detailed_balance,
            args.zero_temperature,
            run_label=f"{seed}-penalty-scaled",
            first_moment_error="0.25",
        )
        assert_power_of_two_scaling(unit_statistics, scaled_statistics)
        unit_spectrum = published_spectrum_artifacts(unit_artifacts)
        scaled_spectrum = published_spectrum_artifacts(scaled_artifacts)
        if scaled_spectrum != unit_spectrum:
            raise AssertionError(
                "uniform first-moment fitness scaling changed published "
                f"best-spectrum artifacts for seed {seed}"
            )
        accepted_evolved_trial |= (
            materially_less(
                unit_statistics["mean"][1], unit_statistics["mean"][0]
            )
            and materially_less(
                unit_statistics["squared_mean"][1],
                unit_statistics["squared_mean"][0],
            )
        )
    if not accepted_evolved_trial:
        raise AssertionError(
            "first-moment-only regression did not accept an evolved population row"
        )


if __name__ == "__main__":
    main()
