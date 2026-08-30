#include "result_io.hpp"

#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

class temporary_directory {
public:
    temporary_directory() {
        const auto suffix = std::chrono::high_resolution_clock::now()
                                    .time_since_epoch()
                                    .count();
        path_ = fs::temp_directory_path() /
                ("deac-result-io-test-" + std::to_string(suffix));
        fs::create_directories(path_);
    }

    ~temporary_directory() {
        try {
            fs::remove_all(path_);
        } catch (...) {
        }
    }

    const fs::path& path() const { return path_; }

private:
    fs::path path_;
};

template <typename Callable>
void expect_error(Callable&& callable, const std::string& expected_text) {
    try {
        callable();
    } catch (const std::runtime_error& error) {
        if (std::string(error.what()).find(expected_text) == std::string::npos) {
            throw std::runtime_error(
                    "expected error containing '" + expected_text +
                    "', got '" + error.what() + "'");
        }
        return;
    }
    throw std::runtime_error(
            "expected runtime_error containing '" + expected_text + "'");
}

void write_bytes(const fs::path& path, const std::string& bytes) {
    std::ofstream output(path.string(), std::ios::binary | std::ios::trunc);
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    output.close();
    if (!output) {
        throw std::runtime_error("test fixture write failed");
    }
}

} // namespace

int main() {
    try {
        temporary_directory temporary;
        const fs::path root = temporary.path();

        const std::vector<double> expected{0.0, -1.25, 4.5};
        const fs::path binary_path = root / "values.bin";
        deac_io::write_binary_doubles(binary_path, expected);
        const std::vector<double> actual = deac_io::read_binary_doubles(binary_path);
        if (actual.size() != expected.size() ||
            std::memcmp(actual.data(), expected.data(), expected.size() * sizeof(double)) != 0) {
            throw std::runtime_error("binary-double round trip changed bytes");
        }

        expect_error(
                [&] { deac_io::read_binary_doubles(root / "missing.bin"); },
                "could not open binary input");

        const fs::path empty_path = root / "empty.bin";
        write_bytes(empty_path, "");
        expect_error(
                [&] { deac_io::read_binary_doubles(empty_path); }, "is empty");

        const fs::path partial_path = root / "partial.bin";
        write_bytes(partial_path, "partial");
        expect_error(
                [&] { deac_io::read_binary_doubles(partial_path); },
                "does not contain a whole number of doubles");

        const fs::path nested_directory = root / "nested" / "result" / "directory";
        deac_io::ensure_result_directory(nested_directory);
        if (!fs::is_directory(nested_directory)) {
            throw std::runtime_error("nested result directory was not created");
        }

        const fs::path file_instead_of_directory = root / "ordinary-file";
        write_bytes(file_instead_of_directory, "content");
        expect_error(
                [&] { deac_io::ensure_result_directory(file_instead_of_directory); },
                "exists but is not a directory");

        expect_error(
                [&] {
                    deac_io::write_binary_doubles(
                            root / "missing-parent" / "output.bin", expected);
                },
                "could not open binary output");

        const fs::path log_path = root / "run.log";
        deac_io::append_text(log_path, "first\n");
        deac_io::append_text(log_path, "second\n");
        std::ifstream log_input(log_path.string());
        const std::string log_text(
                (std::istreambuf_iterator<char>(log_input)),
                std::istreambuf_iterator<char>());
        if (log_text != "first\nsecond\n") {
            throw std::runtime_error("checked log append changed text");
        }

        const fs::path log_directory = root / "directory.log";
        fs::create_directory(log_directory);
        expect_error(
                [&] { deac_io::append_text(log_directory, "text\n"); },
                "could not open log file");

        const fs::path full_device("/dev/full");
        if (fs::exists(full_device)) {
            expect_error(
                    [&] { deac_io::write_binary_doubles(full_device, expected); },
                    "binary output");
            expect_error(
                    [&] { deac_io::append_text(full_device, "text\n"); },
                    "log file");
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
