#ifndef DEAC_RESULT_IO_HPP
#define DEAC_RESULT_IO_HPP

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <fstream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include <fs.h>

namespace deac_io {

inline std::string quoted_path(const fs::path& path) {
    return "'" + path.string() + "'";
}

inline std::string errno_detail(int error_number) {
    if (error_number == 0) {
        return "";
    }
    return ": " + std::error_code(error_number, std::generic_category()).message();
}

inline void ensure_result_directory(const fs::path& directory) {
    if (directory.empty()) {
        throw std::runtime_error("result directory path is empty");
    }

    try {
        if (fs::exists(directory)) {
            if (!fs::is_directory(directory)) {
                throw std::runtime_error(
                        "result path " + quoted_path(directory) +
                        " exists but is not a directory");
            }
            return;
        }
        if (!fs::create_directories(directory) && !fs::is_directory(directory)) {
            throw std::runtime_error(
                    "could not create result directory " + quoted_path(directory));
        }
    } catch (const fs::filesystem_error& error) {
        throw std::runtime_error(
                "could not create result directory " + quoted_path(directory) +
                ": " + error.code().message());
    }
}

inline std::vector<double> read_binary_doubles(const fs::path& path) {
    errno = 0;
    std::ifstream input(path.string(), std::ios::binary | std::ios::ate);
    if (!input.is_open()) {
        const int open_error = errno;
        throw std::runtime_error(
                "could not open binary input " + quoted_path(path) +
                " for reading" + errno_detail(open_error));
    }

    const std::streamoff end_position = input.tellg();
    if (end_position < 0) {
        input.clear();
        input.close();
        throw std::runtime_error(
                "could not determine byte length of binary input " +
                quoted_path(path));
    }

    const auto byte_count = static_cast<unsigned long long>(end_position);
    if (byte_count % sizeof(double) != 0) {
        input.close();
        if (input.fail()) {
            throw std::runtime_error(
                    "failed to close binary input " + quoted_path(path));
        }
        throw std::runtime_error(
                "binary input " + quoted_path(path) +
                " does not contain a whole number of doubles");
    }
    if (byte_count == 0) {
        input.close();
        if (input.fail()) {
            throw std::runtime_error(
                    "failed to close binary input " + quoted_path(path));
        }
        throw std::runtime_error("binary input " + quoted_path(path) + " is empty");
    }
    const auto element_count = byte_count / sizeof(double);
    if (byte_count > static_cast<unsigned long long>(
                             std::numeric_limits<std::size_t>::max())) {
        input.close();
        if (input.fail()) {
            throw std::runtime_error(
                    "failed to close binary input " + quoted_path(path));
        }
        throw std::runtime_error(
                "binary input " + quoted_path(path) +
                " is too large for this process");
    }

    std::vector<double> values;
    if (element_count > values.max_size()) {
        input.close();
        if (input.fail()) {
            throw std::runtime_error(
                    "failed to close binary input " + quoted_path(path));
        }
        throw std::runtime_error(
                "binary input " + quoted_path(path) +
                " is too large for a double array");
    }
    values.resize(static_cast<std::size_t>(element_count));
    input.seekg(0, std::ios::beg);
    if (!input) {
        input.clear();
        input.close();
        throw std::runtime_error(
                "could not seek to the start of binary input " + quoted_path(path));
    }

    std::size_t offset = 0;
    const std::size_t total_bytes = static_cast<std::size_t>(byte_count);
    const auto maximum_chunk = static_cast<std::size_t>(
            std::numeric_limits<std::streamsize>::max());
    auto* destination = reinterpret_cast<char*>(values.data());
    while (offset < total_bytes) {
        const std::size_t chunk = std::min(total_bytes - offset, maximum_chunk);
        input.read(destination + offset, static_cast<std::streamsize>(chunk));
        if (input.gcount() != static_cast<std::streamsize>(chunk)) {
            const std::streamsize bytes_read = input.gcount();
            input.clear();
            input.close();
            throw std::runtime_error(
                    "short read from binary input " + quoted_path(path) +
                    ": expected " + std::to_string(chunk) + " bytes, got " +
                    std::to_string(bytes_read));
        }
        offset += chunk;
    }

    input.close();
    if (input.fail()) {
        throw std::runtime_error(
                "failed to close binary input " + quoted_path(path));
    }
    return values;
}

inline void write_binary_doubles(
        const fs::path& path, std::span<const double> values) {
    errno = 0;
    std::ofstream output(path.string(), std::ios::binary | std::ios::trunc);
    if (!output.is_open()) {
        const int open_error = errno;
        throw std::runtime_error(
                "could not open binary output " + quoted_path(path) +
                " for writing" + errno_detail(open_error));
    }

    const auto* source = reinterpret_cast<const char*>(values.data());
    const std::size_t total_bytes = values.size_bytes();
    const auto maximum_chunk = static_cast<std::size_t>(
            std::numeric_limits<std::streamsize>::max());
    std::size_t offset = 0;
    while (offset < total_bytes) {
        const std::size_t chunk = std::min(total_bytes - offset, maximum_chunk);
        errno = 0;
        output.write(source + offset, static_cast<std::streamsize>(chunk));
        if (!output) {
            const int write_error = errno;
            output.clear();
            output.close();
            throw std::runtime_error(
                    "failed to write binary output " + quoted_path(path) +
                    errno_detail(write_error));
        }
        offset += chunk;
    }

    errno = 0;
    output.flush();
    if (!output) {
        const int flush_error = errno;
        output.clear();
        output.close();
        throw std::runtime_error(
                "failed to flush binary output " + quoted_path(path) +
                errno_detail(flush_error));
    }
    errno = 0;
    output.close();
    if (output.fail()) {
        const int close_error = errno;
        throw std::runtime_error(
                "failed to close binary output " + quoted_path(path) +
                errno_detail(close_error));
    }
}

inline void append_text(const fs::path& path, std::string_view text) {
    errno = 0;
    std::ofstream output(path.string(), std::ios::out | std::ios::app);
    if (!output.is_open()) {
        const int open_error = errno;
        throw std::runtime_error(
                "could not open log file " + quoted_path(path) +
                " for appending" + errno_detail(open_error));
    }

    errno = 0;
    output.write(text.data(), static_cast<std::streamsize>(text.size()));
    if (!output) {
        const int write_error = errno;
        output.clear();
        output.close();
        throw std::runtime_error(
                "failed to append log file " + quoted_path(path) +
                errno_detail(write_error));
    }
    errno = 0;
    output.flush();
    if (!output) {
        const int flush_error = errno;
        output.clear();
        output.close();
        throw std::runtime_error(
                "failed to flush log file " + quoted_path(path) +
                errno_detail(flush_error));
    }
    errno = 0;
    output.close();
    if (output.fail()) {
        const int close_error = errno;
        throw std::runtime_error(
                "failed to close log file " + quoted_path(path) +
                errno_detail(close_error));
    }
}

} // namespace deac_io

#endif
