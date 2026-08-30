#!/usr/bin/env python3
import argparse
import math
import os
import shutil
import stat
import struct
import subprocess
from pathlib import Path


SMOKE_CASES = [
    "help",
    "version",
    "bad_spectra",
    "default",
    "evolution_control_lower_boundaries",
    "evolution_control_upper_boundaries",
    "normalize",
    "normalize_large_population",
    "first_moment",
    "third_moment",
    "negative_first_moment",
    "track_stats",
    "valid_nonuniform_frequency",
    "nested_output",
]

VALIDATION_CASES = [
    "bad_isf_byte_length",
    "empty_isf",
    "uneven_isf_arrays",
    "too_few_timeslices",
    "nonfinite_isf",
    "normalize_zero_target",
    "normalize_negative_target",
    "normalize_subnormal_target",
    "normalize_nonfinite_target",
    "normalize_unrepresentable_scale",
    "normalize_signed_weights",
    "missing_isf",
    "positive_isf_single_particle",
    "bad_third_moment_error",
    "bad_crossover_probability",
    "bad_self_adapting_crossover_probability",
    "bad_differential_weight",
    "bad_self_adapting_differential_weight_probability",
    "bad_stop_minimum_fitness",
    "inactive_options_rejected",
    "too_few_generations",
    "too_small_population",
    "too_small_genome",
    "bad_omega_max",
    "missing_frequency",
    "short_frequency_file",
    "nonfinite_frequency",
    "negative_frequency",
    "duplicate_frequency",
    "all_equal_frequency",
    "unsorted_frequency",
    "save_directory_is_file",
    "unwritable_save_directory",
    "log_destination_is_directory",
    "result_destination_is_directory",
    "unsupported_negative_first_moment",
]


def write_doubles(path, values):
    path.write_bytes(struct.pack("<" + "d" * len(values), *values))


def read_doubles(path):
    data = path.read_bytes()
    if len(data) % 8 != 0:
        raise AssertionError(f"{path} does not contain a whole number of doubles")
    return struct.unpack("<" + "d" * (len(data) // 8), data)


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
    save_directory=None,
    uuid=None,
    zero_temperature=False,
):
    if save_directory is None:
        save_directory = workdir / "results"
    command = [
        exe,
        "-T",
        "0.0" if zero_temperature else "1.0",
        "-N",
        number_of_generations,
        "-P",
        population_size,
        "-M",
        genome_size,
        "--omega_max",
        omega_max,
        "--save_directory",
        str(save_directory),
        "--seed",
        seed,
    ]
    if uuid is not None:
        command.extend(["--uuid", uuid])
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


def run_deac_case(
    exe, workdir, case_name, detailed_balance=False, zero_temperature=False
):
    fixture = workdir / "tiny-isf.bin"
    if zero_temperature or case_name in (
        "normalize",
        "normalize_large_population",
    ):
        write_positive_fixture(fixture)
    else:
        write_fixture(fixture)
    save_dir = (
        workdir / "nested" / "result" / "directory"
        if case_name == "nested_output"
        else workdir / "results"
    )
    seed = {
        "default": "1",
        "evolution_control_lower_boundaries": "8",
        "evolution_control_upper_boundaries": "9",
        "normalize": "2",
        "normalize_large_population": "6",
        "first_moment": "3",
        "third_moment": "11",
        "negative_first_moment": "5",
        "track_stats": "4",
        "valid_nonuniform_frequency": "12",
        "nested_output": "10",
    }[case_name]

    extra_args = []
    number_of_generations = "2"
    population_size = "8"
    if case_name == "valid_nonuniform_frequency":
        frequency_file = workdir / "frequency.bin"
        write_doubles(
            frequency_file,
            [0.25, 0.5, 1.0, 1.75, 2.0, 3.5, 4.25, 6.0],
        )
        extra_args.extend(["--frequency_file", str(frequency_file)])
    elif case_name in ("normalize", "normalize_large_population"):
        extra_args.append("--normalize")
        if not detailed_balance and not zero_temperature:
            extra_args.extend(["--spectra_type", "bfull"])
        if case_name == "normalize_large_population":
            population_size = "1028"
    elif case_name == "first_moment":
        extra_args.extend(["--first_moment", "0.5"])
    elif case_name == "third_moment":
        extra_args.extend(
            ["--third_moment", "0.5", "--third_moment_error", "0.1"]
        )
    elif case_name == "negative_first_moment":
        extra_args.append("--use_negative_first_moment")
    elif case_name == "track_stats":
        number_of_generations = "3"
        extra_args.append("--track_stats")
    elif case_name in (
        "evolution_control_lower_boundaries",
        "evolution_control_upper_boundaries",
    ):
        if case_name == "evolution_control_lower_boundaries":
            probability = "0"
            differential_weight = "0"
            stop_minimum_fitness = "-1"
        else:
            probability = "1"
            differential_weight = "2"
            stop_minimum_fitness = "1"
        extra_args.extend(
            [
                "--crossover_probability",
                probability,
                "--self_adapting_crossover_probability",
                probability,
                "--differential_weight",
                differential_weight,
                "--self_adapting_differential_weight_probability",
                probability,
                "--stop_minimum_fitness",
                stop_minimum_fitness,
            ]
        )
    command = deac_command(
        exe,
        workdir,
        fixture,
        number_of_generations=number_of_generations,
        population_size=population_size,
        seed=seed,
        extra_args=extra_args,
        save_directory=save_dir,
        zero_temperature=zero_temperature,
    )

    run_command(command, workdir, expected_output="minimum_fitness:")

    expected_spectrum_bytes = (
        8 if detailed_balance or zero_temperature else 2 * 8 - 1
    ) * 8
    if zero_temperature:
        prefix = "deac-zT"
    elif (
        case_name in ("normalize", "normalize_large_population")
        and not detailed_balance
    ):
        prefix = "deac-bfull"
    else:
        prefix = "deac-bdsf" if detailed_balance else "deac-spfsf"
    assert_file_size(save_dir / f"{prefix}_dsf_{seed}.bin", expected_spectrum_bytes)
    assert_file_size(save_dir / f"{prefix}_frequency_{seed}.bin", expected_spectrum_bytes)

    if zero_temperature:
        frequencies = read_doubles(save_dir / f"{prefix}_frequency_{seed}.bin")
        if len(frequencies) != 8 or any(frequency < 0.0 for frequency in frequencies):
            raise AssertionError(
                "expected ZeroT output to contain only the eight non-negative "
                f"input-grid frequencies, got {frequencies}"
            )

    log_path = save_dir / f"{prefix}_log_{seed}.dat"
    if not log_path.exists():
        raise AssertionError(f"expected log file {log_path} to exist")
    log_text = log_path.read_text()
    if "minimum_fitness:" not in log_text:
        raise AssertionError(f"expected {log_path} to contain minimum_fitness")
    if zero_temperature:
        if "kernel: zero-temperature-positive-laplace" not in log_text:
            raise AssertionError(
                f"expected {log_path} to identify the ZeroT Laplace kernel"
            )
        if "temperature: 0" not in log_text:
            raise AssertionError(f"expected {log_path} to record zero temperature")

    if detailed_balance and case_name == "negative_first_moment":
        error_line = next(
            (
                line
                for line in log_text.splitlines()
                if line.startswith("negative_first_moment_error: ")
            ),
            None,
        )
        if error_line is None:
            raise AssertionError(
                f"expected {log_path} to contain negative_first_moment_error"
            )
        actual_error = float(error_line.split(": ", 1)[1])
        # Trapezoid weights for tau=[0.0, 0.2, 0.4, 0.6] are
        # [0.1, 0.2, 0.2, 0.1].  Every sample has sigma=0.05.
        expected_error = 0.05 * math.sqrt(0.1**2 + 0.2**2 + 0.2**2 + 0.1**2)
        if not math.isclose(actual_error, expected_error, rel_tol=1e-5):
            raise AssertionError(
                f"expected accumulated negative first-moment error "
                f"{expected_error}, got {actual_error}"
            )

    if case_name == "track_stats":
        expected_stats_bytes = 3 * 8
        assert_file_size(save_dir / f"{prefix}_stats_fitness-mean_{seed}.bin", expected_stats_bytes)
        assert_file_size(save_dir / f"{prefix}_stats_fitness-minimum_{seed}.bin", expected_stats_bytes)
        assert_file_size(save_dir / f"{prefix}_stats_fitness-squared-mean_{seed}.bin", expected_stats_bytes)
        if 'fitness_mean_filename: ""' in log_text:
            raise AssertionError("tracked-stat filenames were not written to the log")


def run_validation_case(
    exe, workdir, case_name, detailed_balance=False, zero_temperature=False
):
    fixture = workdir / "invalid-isf.bin"
    frequency_file = workdir / "frequency.bin"
    command_options = {}
    extra_args = None
    save_directory = None
    uuid = None
    permission_restore = None
    expected_returncode = 1

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
    elif case_name in {
        "normalize_zero_target",
        "normalize_negative_target",
        "normalize_subnormal_target",
        "normalize_nonfinite_target",
    }:
        target = {
            "normalize_zero_target": 0.0,
            "normalize_negative_target": -1.0,
            "normalize_subnormal_target": math.ulp(0.0),
            "normalize_nonfinite_target": float("nan"),
        }[case_name]
        write_fixture(fixture, isf=[target, 0.85, 0.72, 0.61])
        extra_args = ["--normalize"]
        if not detailed_balance and not zero_temperature:
            extra_args.extend(["--spectra_type", "bfull"])
        expected_output = (
            "--normalize requires the first ISF value (zeroth moment) to be "
            "finite, positive, and at least the smallest normal double"
        )
    elif case_name == "normalize_unrepresentable_scale":
        target = float.fromhex("0x1.0p-1022")
        write_fixture(fixture, isf=[target, target, target, target])
        extra_args = ["--normalize"]
        if not detailed_balance and not zero_temperature:
            extra_args.extend(["--spectra_type", "bfull"])
        expected_output = (
            "--normalize target is outside the representable normal range "
            "for this model and frequency grid"
        )
    elif case_name == "normalize_signed_weights":
        write_positive_fixture(fixture)
        extra_args = ["--normalize"]
        if not detailed_balance and not zero_temperature:
            extra_args.extend(["--spectra_type", "bfull"])
        expected_output = (
            "--normalize is incompatible with negative spectral weights"
        )
    elif case_name == "missing_isf":
        expected_output = "could not open binary input"
    elif case_name == "positive_isf_single_particle":
        write_positive_fixture(fixture)
        if detailed_balance or zero_temperature:
            expected_returncode = 0
            expected_output = "minimum_fitness:"
        else:
            expected_output = "positive ISF values are not supported for single-particle spectra"
    elif case_name == "bad_third_moment_error":
        write_fixture(fixture)
        extra_args = ["--third_moment", "1.0", "--third_moment_error", "0.0"]
        expected_output = "third_moment_error must be positive when third_moment is used"
    elif case_name == "bad_crossover_probability":
        write_fixture(fixture)
        extra_args = ["--crossover_probability", "1.01"]
        expected_output = "crossover_probability must be finite and in [0, 1]"
    elif case_name == "bad_self_adapting_crossover_probability":
        write_fixture(fixture)
        extra_args = ["--self_adapting_crossover_probability", "-0.01"]
        expected_output = (
            "self_adapting_crossover_probability must be finite and in "
            "[0, 1]"
        )
    elif case_name == "bad_differential_weight":
        write_fixture(fixture)
        extra_args = ["--differential_weight", "2.01"]
        expected_output = "differential_weight must be finite and in [0, 2]"
    elif case_name == "bad_self_adapting_differential_weight_probability":
        write_fixture(fixture)
        extra_args = ["--self_adapting_differential_weight_probability", "nan"]
        expected_output = (
            "self_adapting_differential_weight_probability must be finite and in "
            "[0, 1]"
        )
    elif case_name == "bad_stop_minimum_fitness":
        write_fixture(fixture)
        extra_args = ["--stop_minimum_fitness", "inf"]
        expected_output = "stop_minimum_fitness must be finite"
    elif case_name == "inactive_options_rejected":
        write_fixture(fixture)
        save_dir = workdir / "results"
        for inactive_option in (
            ["--save_state"],
            ["-l", "0.1"],
            ["--self_adapting_differential_weight_shift", "0.1"],
            ["-m", "0.9"],
            ["--self_adapting_differential_weight", "0.9"],
        ):
            command = deac_command(
                exe,
                workdir,
                fixture,
                extra_args=inactive_option,
                zero_temperature=zero_temperature,
            )
            run_command(
                command,
                workdir,
                expected_returncode=1,
                expected_output=inactive_option[0],
            )
            if save_dir.exists():
                raise AssertionError(
                    f"inactive option {inactive_option[0]} created {save_dir}"
                )
        return
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
    elif case_name == "missing_frequency":
        write_fixture(fixture)
        extra_args = ["--frequency_file", str(frequency_file)]
        expected_output = "could not open binary input"
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
    elif case_name == "duplicate_frequency":
        write_fixture(fixture)
        write_doubles(frequency_file, [0.0, 0.75, 0.75, 2.0])
        extra_args = ["--frequency_file", str(frequency_file)]
        expected_output = "frequencies must be strictly increasing"
    elif case_name == "all_equal_frequency":
        write_fixture(fixture)
        write_doubles(frequency_file, [1.0, 1.0, 1.0])
        extra_args = ["--frequency_file", str(frequency_file)]
        expected_output = "frequencies must be strictly increasing"
    elif case_name == "unsorted_frequency":
        write_fixture(fixture)
        write_doubles(frequency_file, [0.0, 2.0, 1.0])
        extra_args = ["--frequency_file", str(frequency_file)]
        expected_output = "frequencies must be strictly increasing"
    elif case_name == "save_directory_is_file":
        write_fixture(fixture)
        save_directory = workdir / "result-path-is-a-file"
        save_directory.write_text("not a directory")
        expected_output = "exists but is not a directory"
    elif case_name == "unwritable_save_directory":
        write_fixture(fixture)
        if hasattr(os, "geteuid") and os.geteuid() == 0 and Path("/proc").is_dir():
            save_directory = Path("/proc") / "deac-cpp-result-io-test" / "nested"
        elif os.name == "posix":
            read_only_parent = workdir / "read-only"
            read_only_parent.mkdir()
            read_only_parent.chmod(stat.S_IRUSR | stat.S_IXUSR)
            permission_restore = read_only_parent
            save_directory = read_only_parent / "nested"
        else:
            blocked_parent = workdir / "blocked-parent"
            blocked_parent.write_text("not a directory")
            save_directory = blocked_parent / "nested"
        expected_output = "could not create result directory"
    elif case_name == "log_destination_is_directory":
        write_fixture(fixture)
        uuid = "log-destination"
        save_directory = workdir / "results"
        save_directory.mkdir()
        if zero_temperature:
            prefix = "deac-zT"
        else:
            prefix = "deac-bdsf" if detailed_balance else "deac-spfsf"
        (save_directory / f"{prefix}_log_{uuid}.dat").mkdir()
        expected_output = "could not open log file"
    elif case_name == "result_destination_is_directory":
        write_fixture(fixture)
        uuid = "result-destination"
        save_directory = workdir / "results"
        save_directory.mkdir()
        if zero_temperature:
            prefix = "deac-zT"
        else:
            prefix = "deac-bdsf" if detailed_balance else "deac-spfsf"
        (save_directory / f"{prefix}_dsf_{uuid}.bin").mkdir()
        expected_output = "could not open binary output"
    elif case_name == "unsupported_negative_first_moment":
        write_fixture(fixture)
        extra_args = ["--use_negative_first_moment"]
        expected_output = (
            "use_negative_first_moment requires a finite-temperature "
            "bosonic detailed-balance build"
        )
    else:
        raise AssertionError(f"unknown validation case {case_name}")

    command = deac_command(
        exe,
        workdir,
        fixture,
        extra_args=extra_args,
        save_directory=save_directory,
        uuid=uuid,
        zero_temperature=zero_temperature,
        **command_options,
    )
    save_dir = save_directory if save_directory is not None else workdir / "results"
    if case_name == "bad_stop_minimum_fitness":
        # Exercise the no-log guarantee independently of the no-directory
        # guarantee covered by the other invalid evolution controls.
        save_dir.mkdir()
    try:
        result = run_command(
            command,
            workdir,
            expected_returncode=expected_returncode,
            expected_output=expected_output,
        )
    finally:
        if permission_restore is not None:
            permission_restore.chmod(stat.S_IRWXU)

    if case_name == "log_destination_is_directory":
        if "minimum_fitness:" in result.stdout:
            raise AssertionError("evolution started despite an invalid initial log destination")
        if zero_temperature:
            prefix = "deac-zT"
        else:
            prefix = "deac-bdsf" if detailed_balance else "deac-spfsf"
        if (save_dir / f"{prefix}_dsf_{uuid}.bin").exists():
            raise AssertionError("invalid initial log destination produced a spectrum")
    elif case_name == "result_destination_is_directory":
        if zero_temperature:
            prefix = "deac-zT"
        else:
            prefix = "deac-bdsf" if detailed_balance else "deac-spfsf"
        if (save_dir / f"{prefix}_frequency_{uuid}.bin").exists():
            raise AssertionError("solver continued writing after the first result I/O failure")
    elif case_name in {
        "normalize_zero_target",
        "normalize_negative_target",
        "normalize_subnormal_target",
        "normalize_nonfinite_target",
        "normalize_unrepresentable_scale",
        "normalize_signed_weights",
    }:
        if "uuid:" in result.stdout:
            raise AssertionError(
                "invalid normalization target emitted a run identifier"
            )
        if "minimum_fitness:" in result.stdout:
            raise AssertionError("evolution started despite an invalid normalization target")
        if save_dir.exists():
            raise AssertionError(
                f"invalid normalization target created output directory {save_dir}"
            )
    elif case_name in {
        "nonfinite_frequency",
        "negative_frequency",
        "duplicate_frequency",
        "all_equal_frequency",
        "unsorted_frequency",
    }:
        if "minimum_fitness:" in result.stdout:
            raise AssertionError("evolution started despite an invalid frequency grid")
        if save_dir.exists():
            raise AssertionError(
                f"invalid frequency grid created output directory {save_dir}"
            )
    if case_name.startswith("bad_") and case_name in {
        "bad_crossover_probability",
        "bad_self_adapting_crossover_probability",
        "bad_differential_weight",
        "bad_self_adapting_differential_weight_probability",
        "bad_stop_minimum_fitness",
    }:
        if case_name == "bad_stop_minimum_fitness":
            if any(save_dir.iterdir()):
                raise AssertionError(
                    f"invalid evolution controls wrote output under {save_dir}"
                )
        elif save_dir.exists():
            raise AssertionError(
                f"invalid evolution controls created output directory {save_dir}"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--detailed-balance", action="store_true")
    parser.add_argument("--zero-temperature", action="store_true")
    parser.add_argument("--expected-version", required=True)
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
        result = run_command(
            [exe, "--help"], workdir, expected_output="Usage: deac-cpp"
        )
        for expected_output in (
            "--build-identity",
            "--self_adapting_differential_weight_probability",
            "Must be finite and in [0, 2].",
            "Negative values are allowed.",
        ):
            if expected_output not in result.stdout:
                raise AssertionError(
                    f"expected --help to contain {expected_output!r}\n"
                    f"output:\n{result.stdout}"
                )
        for inactive_option in (
            "--save_state",
            "--self_adapting_differential_weight_shift",
        ):
            if inactive_option in result.stdout:
                raise AssertionError(
                    f"expected --help to omit inactive option {inactive_option!r}\n"
                    f"output:\n{result.stdout}"
                )
        help_without_probability = result.stdout.replace(
            "--self_adapting_differential_weight_probability", ""
        )
        if "--self_adapting_differential_weight" in help_without_probability:
            raise AssertionError(
                "expected --help to omit inactive differential-weight range option\n"
                f"output:\n{result.stdout}"
            )
        if args.zero_temperature:
            for expected_output in (
                "temperature is fixed to zero",
                "one-sided positive-frequency Laplace kernel",
            ):
                if expected_output not in result.stdout:
                    raise AssertionError(
                        f"expected ZeroT --help to contain {expected_output!r}\n"
                        f"output:\n{result.stdout}"
                    )
    elif args.case == "version":
        result = run_command([exe, "-v"], workdir)
        if result.stdout.strip() != args.expected_version:
            raise AssertionError(
                "expected -v to print only configured version "
                f"{args.expected_version!r}, got {result.stdout!r}"
            )
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
        run_validation_case(
            exe,
            workdir,
            args.case,
            args.detailed_balance,
            args.zero_temperature,
        )
    else:
        run_deac_case(
            exe,
            workdir,
            args.case,
            args.detailed_balance,
            args.zero_temperature,
        )


if __name__ == "__main__":
    main()
