#!/usr/bin/env python3
import argparse
import shutil
import struct
import subprocess
from pathlib import Path


def write_fixture(path):
    tau = [0.0, 0.2, 0.4, 0.6]
    isf = [1.0, 0.85, 0.72, 0.61]
    error = [0.05, 0.05, 0.05, 0.05]
    values = tau + isf + error
    path.write_bytes(struct.pack("<" + "d" * len(values), *values))


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

    command = [
        exe,
        "-T",
        "1.0",
        "-N",
        "2",
        "-P",
        "8",
        "-M",
        "8",
        "--omega_max",
        "4.0",
        "--save_directory",
        str(save_dir),
        "--seed",
        seed,
    ]
    if case_name == "normalize":
        command.append("--normalize")
    elif case_name == "first_moment":
        command.extend(["--first_moment", "0.5"])
    elif case_name == "track_stats":
        command[command.index("-N") + 1] = "3"
        command.append("--track_stats")
    command.append(str(fixture))

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument(
        "--case",
        required=True,
        choices=["help", "bad_spectra", "default", "normalize", "first_moment", "track_stats"],
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
    else:
        run_deac_case(exe, workdir, args.case)


if __name__ == "__main__":
    main()
