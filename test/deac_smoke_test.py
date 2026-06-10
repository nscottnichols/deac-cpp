#!/usr/bin/env python3
import argparse
import shutil
import struct
import subprocess
from pathlib import Path


SMOKE_CASES = [
    "help",
    "bad_spectra",
    "default",
    "normalize",
    "first_moment",
    "track_stats",
]

VALIDATION_CASES = [
    "bad_isf_byte_length",
    "empty_isf",
    "uneven_isf_arrays",
    "too_few_timeslices",
    "nonfinite_isf",
    "positive_isf_single_particle",
    "bad_third_moment_error",
    "too_few_generations",
    "too_small_population",
    "too_small_genome",
    "bad_omega_max",
    "short_frequency_file",
    "nonfinite_frequency",
    "negative_frequency",
    "unsorted_frequency",
]


def write_doubles(path, values):
    path.write_bytes(struct.pack("<" + "d" * len(values), *values))


def write_fixture(path, tau=None, isf=None, error=None):
    if tau is None:
        tau = [0.0, 0.2, 0.4, 0.6]
    if isf is None:
        isf = [-1.0, -0.85, -0.72, -0.61]
    if error is None:
        error = [0.05, 0.05, 0.05, 0.05]
    if not (len(tau) == len(isf) == len(error)):
        raise ValueError("fixture arrays must have equal length")
    values = tau + isf + error
    write_doubles(path, values)


def write_positive_fixture(path):
    tau = [0.0, 0.2, 0.4, 0.6]
    isf = [1.0, 0.85, 0.72, 0.61]
    error = [0.05, 0.05, 0.05, 0.05]
    write_fixture(path, tau=tau, isf=isf, error=error)


def run_command(command, workdir, expected_returncode=0, expected_output=None):
    result = subprocess.run(
        command,
        cwd=workdir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode != expected_returncode:
        raise AssertionError(
            f"expected return code {expected_returncode}, got {result.returncode}\n"
            f"command: {' '.join(map(str, command))}\n"
            f"output:\n{result.stdout}"
        )
    if expected_output is not None and expected_output not in result.stdout:
        raise AssertionError(
            f"expected output to contain {expected_output!r}\n"
            f"command: {' '.join(map(str, command))}\n"
            f"output:\n{result.stdout}"
        )
    return result


def deac_command(
    exe,
    workdir,
    fixture,
    number_of_generations="2",
    population_size="8",
    genome_size="8",
    omega_max="4.0",
    seed="7",
    extra_args=None,
):
    command = [
        exe,
        "-T",
        "1.0",
        "-N",
        number_of_generations,
        "-P",
        population_size,
        "-M",
        genome_size,
        "--omega_max",
        omega_max,
        "--save_directory",
        str(workdir / "results"),
        "--seed",
        seed,
    ]
    if extra_args:
        command.extend(extra_args)
    command.append(str(fixture))
    return command


def assert_file_size(path, expected_size):
    if not path.exists():
        raise AssertionError(f"expected output file {path} to exist")
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise AssertionError(
            f"expected {path} to be {expected_size} bytes, got {actual_size}"
        )


def run_deac_case(exe, workdir, case_name):
    fixture = workdir / "tiny-isf.bin"
    write_fixture(fixture)
    save_dir = workdir / "results"
    seed = {
        "default": "1",
        "normalize": "2",
        "first_moment": "3",
        "track_stats": "4",
    }[case_name]

    extra_args = []
    number_of_generations = "2"
    if case_name == "normalize":
        extra_args.append("--normalize")
    elif case_name == "first_moment":
        extra_args.extend(["--first_moment", "0.5"])
    elif case_name == "track_stats":
        number_of_generations = "3"
        extra_args.append("--track_stats")
    command = deac_command(
        exe,
        workdir,
        fixture,
        number_of_generations=number_of_generations,
        seed=seed,
        extra_args=extra_args,
    )

    run_command(command, workdir, expected_output="minimum_fitness:")

    expected_spectrum_bytes = (2 * 8 - 1) * 8
    prefix = f"deac-spfsf"
    assert_file_size(save_dir / f"{prefix}_dsf_{seed}.bin", expected_spectrum_bytes)
    assert_file_size(save_dir / f"{prefix}_frequency_{seed}.bin", expected_spectrum_bytes)

    log_path = save_dir / f"{prefix}_log_{seed}.dat"
    if not log_path.exists():
        raise AssertionError(f"expected log file {log_path} to exist")
    log_text = log_path.read_text()
    if "minimum_fitness:" not in log_text:
        raise AssertionError(f"expected {log_path} to contain minimum_fitness")

    if case_name == "track_stats":
        expected_stats_bytes = 3 * 8
        assert_file_size(save_dir / f"{prefix}_stats_fitness-mean_{seed}.bin", expected_stats_bytes)
        assert_file_size(save_dir / f"{prefix}_stats_fitness-minimum_{seed}.bin", expected_stats_bytes)
        assert_file_size(save_dir / f"{prefix}_stats_fitness-squared-mean_{seed}.bin", expected_stats_bytes)
        if 'fitness_mean_filename: ""' in log_text:
            raise AssertionError("tracked-stat filenames were not written to the log")


def run_validation_case(exe, workdir, case_name):
    fixture = workdir / "invalid-isf.bin"
    frequency_file = workdir / "frequency.bin"
    command_options = {}
    extra_args = None

    if case_name == "bad_isf_byte_length":
        fixture.write_bytes(b"not-a-double")
        expected_output = "does not contain a whole number of doubles"
    elif case_name == "empty_isf":
        fixture.write_bytes(b"")
        expected_output = "is empty"
    elif case_name == "uneven_isf_arrays":
        write_doubles(fixture, [0.0, 0.2, -1.0, 0.05])
        expected_output = "ISF input file must contain tau, isf, and error arrays of equal length"
    elif case_name == "too_few_timeslices":
        write_fixture(fixture, tau=[0.0], isf=[-1.0], error=[0.05])
        expected_output = "ISF input file must contain at least two timeslices"
    elif case_name == "nonfinite_isf":
        write_fixture(fixture, isf=[-1.0, float("nan"), -0.72, -0.61])
        expected_output = "ISF input file contains non-finite values"
    elif case_name == "positive_isf_single_particle":
        write_positive_fixture(fixture)
        expected_output = "positive ISF values are not supported for single-particle spectra"
    elif case_name == "bad_third_moment_error":
        write_fixture(fixture)
        extra_args = ["--third_moment", "1.0", "--third_moment_error", "0.0"]
        expected_output = "third_moment_error must be positive when third_moment is used"
    elif case_name == "too_few_generations":
        write_fixture(fixture)
        command_options["number_of_generations"] = "1"
        expected_output = "number_of_generations must be at least 2"
    elif case_name == "too_small_population":
        write_fixture(fixture)
        command_options["population_size"] = "3"
        expected_output = "population_size must be at least 4"
    elif case_name == "too_small_genome":
        write_fixture(fixture)
        command_options["genome_size"] = "1"
        expected_output = "genome_size must be at least 2"
    elif case_name == "bad_omega_max":
        write_fixture(fixture)
        command_options["omega_max"] = "0.0"
        expected_output = "omega_max must be positive"
    elif case_name == "short_frequency_file":
        write_fixture(fixture)
        write_doubles(frequency_file, [0.0])
        extra_args = ["--frequency_file", str(frequency_file)]
        expected_output = "frequency_file must contain at least two frequencies"
    elif case_name == "nonfinite_frequency":
        write_fixture(fixture)
        write_doubles(frequency_file, [0.0, float("inf")])
        extra_args = ["--frequency_file", str(frequency_file)]
        expected_output = "frequencies must be finite and non-negative"
    elif case_name == "negative_frequency":
        write_fixture(fixture)
        write_doubles(frequency_file, [0.0, -1.0])
        extra_args = ["--frequency_file", str(frequency_file)]
        expected_output = "frequencies must be finite and non-negative"
    elif case_name == "unsorted_frequency":
        write_fixture(fixture)
        write_doubles(frequency_file, [0.0, 2.0, 1.0])
        extra_args = ["--frequency_file", str(frequency_file)]
        expected_output = "frequencies must be sorted in non-decreasing order"
    else:
        raise AssertionError(f"unknown validation case {case_name}")

    command = deac_command(
        exe,
        workdir,
        fixture,
        extra_args=extra_args,
        **command_options,
    )
    run_command(
        command,
        workdir,
        expected_returncode=1,
        expected_output=expected_output,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument(
        "--case",
        required=True,
        choices=SMOKE_CASES + VALIDATION_CASES,
    )
    args = parser.parse_args()

    workdir = Path(args.workdir)
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)

    exe = str(Path(args.exe))
    if args.case == "help":
        run_command([exe, "--help"], workdir, expected_output="Usage: deac-cpp")
    elif args.case == "bad_spectra":
        fixture = workdir / "tiny-isf.bin"
        write_fixture(fixture)
        run_command(
            [
                exe,
                "-T",
                "1.0",
                "--spectra_type",
                "invalid",
                str(fixture),
            ],
            workdir,
            expected_returncode=1,
            expected_output="Please choose spectra_type",
        )
    elif args.case in VALIDATION_CASES:
        run_validation_case(exe, workdir, args.case)
    else:
        run_deac_case(exe, workdir, args.case)


if __name__ == "__main__":
    main()
