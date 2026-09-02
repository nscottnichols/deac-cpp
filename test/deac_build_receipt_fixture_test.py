import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path


def run(command, cwd, *, check=True, environment=None):
    result = subprocess.run(
        [str(part) for part in command],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    if check and result.returncode != 0:
        raise AssertionError(
            f"command failed with exit {result.returncode}: {command}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def write(path, contents, *, executable=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")
    if executable:
        path.chmod(0o755)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_fixture_modules(solver_source, fixture_source):
    module_names = [
        "DeacBuildIdentity.cmake",
        "DeacBuildReceipt.cmake",
        "DeacHipblas.cmake",
        "GenerateDeacBuildReceipt.cmake",
        "VerifyDeacBuildReceiptTools.cmake",
        "deac_build_receipt_data.cpp.in",
    ]
    module_directory = fixture_source / "src" / "cmake"
    module_directory.mkdir(parents=True)
    for name in module_names:
        shutil.copy2(solver_source / "cmake" / name, module_directory / name)
    support_directory = fixture_source / "src" / "deac" / "src"
    support_directory.mkdir(parents=True)
    shutil.copy2(
        solver_source / "deac" / "src" / "build_identity.hpp",
        support_directory / "build_identity.hpp",
    )


def create_fixture_source(solver_source, fixture_source):
    fixture_source.mkdir(parents=True)
    copy_fixture_modules(solver_source, fixture_source)
    write(fixture_source / "VERSION", "1.2.3\n")
    write(
        fixture_source / "src" / "dependency.cpp",
        "int receipt_dependency() { return 17; }\n",
    )
    write(
        fixture_source / "src" / "probe.cpp",
        """#include "build_identity.hpp"
#include <iostream>

int receipt_dependency();

int main() {
    if (receipt_dependency() != 17) {
        return 2;
    }
    std::cout << deac_build_identity::build_receipt_json() << '\\n';
    return 0;
}
""",
    )
    write(
        fixture_source / "src" / "dependency_two.cpp",
        "int receipt_dependency_two() { return 23; }\n",
    )
    write(
        fixture_source / "src" / "probe_two.cpp",
        """#include "build_identity.hpp"
#include <iostream>

int receipt_dependency_two();

int main() {
    if (receipt_dependency_two() != 23) {
        return 2;
    }
    std::cout << deac_build_identity::build_receipt_json() << '\\n';
    return 0;
}
""",
    )
    write(
        fixture_source / "src" / "CMakeLists.txt",
        """cmake_minimum_required(VERSION 3.27)
project(deac_build_receipt_fixture LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(GPU_BACKEND none CACHE STRING "fixture backend")
set(CMAKE_DISABLE_FIND_PACKAGE_hipblas FALSE CACHE BOOL
    "disable fixture hipBLAS package discovery")
set(DEAC_FIXTURE_ARCHIVER "" CACHE FILEPATH "effective fixture CMAKE_AR")
set(DEAC_FIXTURE_RULE_ATTACK "" CACHE STRING "fixture rule attack")
set(DEAC_FIXTURE_EFFECTIVE_CXX_FLAGS_ATTACK "" CACHE STRING
    "effective CMAKE_CXX_FLAGS shell mutation")
set(DEAC_FIXTURE_LATE_ATTACK "" CACHE STRING
    "post-registration build-receipt mutation")
set(DEAC_FIXTURE_DEPENDENCY_FINGERPRINT_ATTACK "" CACHE STRING
    "post-registration dependency fingerprint mutation")
set(DEAC_FIXTURE_RECEIPT_ATTACK "" CACHE STRING
    "build-receipt output expression mutation")
set(DEAC_FIXTURE_CONTROL_CACHE "" CACHE STRING
    "build-receipt JSON control-character input")
set(DEAC_FIXTURE_NATIVE_LANGUAGE_SHAPE "" CACHE STRING "native language shape")
set(DEAC_FIXTURE_HIPBLAS_MODE "" CACHE STRING "mock hipBLAS receipt mode")
set(DEAC_FIXTURE_HIPBLAS_CONTRACT_ATTACK "" CACHE STRING
    "mock hipBLAS contract mutation")
set(DEAC_FIXTURE_LINK_LIBRARY_ATTACK "" CACHE STRING
    "build-receipt link-library input mutation")
set(DEAC_FIXTURE_CONTROL_LINK_ARTIFACT "" CACHE STRING
    "required link artifact with a control-byte filename")
set(DEAC_FIXTURE_SUBDIRECTORY_DEPENDENCY_MODE "" CACHE STRING
    "subdirectory dependency attribution mode")
set(DEAC_FIXTURE_HIPBLAS_INCLUDE_DIR "" CACHE PATH "mock hipBLAS include root")
set(DEAC_FIXTURE_HIPBLAS_LIBRARY "" CACHE FILEPATH "mock hipBLAS library")
set(DEAC_FIXTURE_HIPBLAS_PACKAGE_DIR "" CACHE PATH "mock hipBLAS package")
set(DEAC_FIXTURE_EXPECTED_HIPBLAS_ARTIFACT "" CACHE FILEPATH
    "expected resolved mock hipBLAS artifact")
option(DEAC_FIXTURE_PARALLEL_CONSUMERS "add a second receipt consumer" OFF)
option(DEAC_FIXTURE_SHARED_DEPENDENCY
    "make parallel receipt consumers share one compiled dependency" OFF)
set(DEAC_FIXTURE_SHARED_FINGERPRINT_ATTACK "" CACHE STRING
    "between-registration shared-target fingerprint mutation")
option(DEAC_FIXTURE_DEPENDENCY_IPO_RELEASE "enable dependency Release IPO" OFF)

# CMake's persisted compiler-information file restores CMAKE_AR as a normal
# variable on every reconfigure, shadowing a changed -DCMAKE_AR cache entry.
# Rebind both scopes so the fixture can exercise a genuine effective archiver
# transition in one build tree.
if(NOT DEAC_FIXTURE_ARCHIVER STREQUAL "")
    set(CMAKE_AR "${DEAC_FIXTURE_ARCHIVER}" CACHE FILEPATH
        "effective fixture CMAKE_AR" FORCE)
    set(CMAKE_AR "${DEAC_FIXTURE_ARCHIVER}")
endif()

if(DEAC_FIXTURE_EFFECTIVE_CXX_FLAGS_ATTACK STREQUAL "dollar-variable")
    set(CMAKE_CXX_FLAGS "$DEAC_UNATTESTED_FLAGS" CACHE STRING
        "fixture effective CXX flags" FORCE)
elseif(DEAC_FIXTURE_EFFECTIVE_CXX_FLAGS_ATTACK STREQUAL "braced-variable")
    string(CONCAT DEAC_FIXTURE_ATTACK_FLAGS "$" "{DEAC_UNATTESTED_FLAGS}")
    set(CMAKE_CXX_FLAGS "${DEAC_FIXTURE_ATTACK_FLAGS}" CACHE STRING
        "fixture effective CXX flags" FORCE)
elseif(DEAC_FIXTURE_EFFECTIVE_CXX_FLAGS_ATTACK STREQUAL "glob")
    set(CMAKE_CXX_FLAGS "-include *.hpp" CACHE STRING
        "fixture effective CXX flags" FORCE)
elseif(DEAC_FIXTURE_EFFECTIVE_CXX_FLAGS_ATTACK STREQUAL "redirection")
    set(CMAKE_CXX_FLAGS "-DDEAC_FLAG_PROBE=1 > deac-flags.log" CACHE STRING
        "fixture effective CXX flags" FORCE)
elseif(DEAC_FIXTURE_EFFECTIVE_CXX_FLAGS_ATTACK STREQUAL "comment")
    set(CMAKE_CXX_FLAGS "-DDEAC_FLAG_PROBE=1 # ignored" CACHE STRING
        "fixture effective CXX flags" FORCE)
elseif(DEAC_FIXTURE_EFFECTIVE_CXX_FLAGS_ATTACK STREQUAL "command-substitution")
    set(CMAKE_CXX_FLAGS "$(deac_receipt_attack)" CACHE STRING
        "fixture effective CXX flags" FORCE)
elseif(DEAC_FIXTURE_EFFECTIVE_CXX_FLAGS_ATTACK STREQUAL "backtick-substitution")
    set(CMAKE_CXX_FLAGS "`deac_receipt_attack`" CACHE STRING
        "fixture effective CXX flags" FORCE)
elseif(DEAC_FIXTURE_EFFECTIVE_CXX_FLAGS_ATTACK STREQUAL "process-substitution")
    set(CMAKE_CXX_FLAGS "<(deac_receipt_attack)" CACHE STRING
        "fixture effective CXX flags" FORCE)
elseif(DEAC_FIXTURE_EFFECTIVE_CXX_FLAGS_ATTACK STREQUAL "quoted-dollar")
    set(CMAKE_CXX_FLAGS "-DDEAC_UNSAFE_DOLLAR='literal$argument'"
        CACHE STRING "fixture effective CXX flags" FORCE)
elseif(DEAC_FIXTURE_EFFECTIVE_CXX_FLAGS_ATTACK STREQUAL "escaped-dollar")
    set(CMAKE_CXX_FLAGS "-DDEAC_UNSAFE_DOLLAR=literal\\\\$argument"
        CACHE STRING "fixture effective CXX flags" FORCE)
elseif(DEAC_FIXTURE_EFFECTIVE_CXX_FLAGS_ATTACK STREQUAL "safe-quoted")
    set(CMAKE_CXX_FLAGS "-DDEAC_SAFE_GLOB='literal*argument'"
        CACHE STRING "fixture effective CXX flags" FORCE)
elseif(DEAC_FIXTURE_EFFECTIVE_CXX_FLAGS_ATTACK STREQUAL "safe-escaped")
    set(CMAKE_CXX_FLAGS "-DDEAC_SAFE_GLOB=literal\\\\*argument"
        CACHE STRING "fixture effective CXX flags" FORCE)
elseif(NOT DEAC_FIXTURE_EFFECTIVE_CXX_FLAGS_ATTACK STREQUAL "")
    message(FATAL_ERROR "unknown effective CMAKE_CXX_FLAGS shell mutation")
endif()

if(DEAC_FIXTURE_SUBDIRECTORY_DEPENDENCY_MODE STREQUAL "")
    add_library(receipt_dependency STATIC dependency.cpp)
elseif(DEAC_FIXTURE_SUBDIRECTORY_DEPENDENCY_MODE MATCHES
        "^(normal|launcher|rule|compiler)$")
    add_subdirectory(subdirectory_dependency)
else()
    message(FATAL_ERROR "unknown subdirectory dependency attribution mode")
endif()
add_executable(receipt_probe probe.cpp)
target_link_libraries(receipt_probe PRIVATE receipt_dependency)
if(DEAC_FIXTURE_DEPENDENCY_IPO_RELEASE)
    set_property(TARGET receipt_dependency PROPERTY
        INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
endif()
if(DEAC_FIXTURE_PARALLEL_CONSUMERS)
    if(DEAC_FIXTURE_SHARED_DEPENDENCY)
        add_executable(receipt_probe_two probe.cpp)
        target_link_libraries(receipt_probe_two PRIVATE receipt_dependency)
    else()
        add_library(receipt_dependency_two STATIC dependency_two.cpp)
        add_executable(receipt_probe_two probe_two.cpp)
        target_link_libraries(receipt_probe_two PRIVATE receipt_dependency_two)
    endif()
endif()

if(DEAC_FIXTURE_HIPBLAS_MODE STREQUAL "imported_target")
    set(GPU_BACKEND hip CACHE STRING "fixture backend" FORCE)
    set(USE_BLAS ON CACHE BOOL "fixture BLAS mode" FORCE)
    set(CMAKE_DISABLE_FIND_PACKAGE_hipblas FALSE CACHE BOOL
        "disable fixture hipBLAS package discovery" FORCE)
    set(hipblas_DIR "${DEAC_FIXTURE_HIPBLAS_PACKAGE_DIR}"
        CACHE PATH "mock hipBLAS package" FORCE)
elseif(DEAC_FIXTURE_HIPBLAS_MODE STREQUAL "compatibility_library")
    set(GPU_BACKEND hip CACHE STRING "fixture backend" FORCE)
    set(USE_BLAS ON CACHE BOOL "fixture BLAS mode" FORCE)
    set(CMAKE_DISABLE_FIND_PACKAGE_hipblas TRUE CACHE BOOL
        "disable fixture hipBLAS package discovery" FORCE)
    set(DEAC_HIPBLAS_INCLUDE_DIR "${DEAC_FIXTURE_HIPBLAS_INCLUDE_DIR}"
        CACHE PATH "mock hipBLAS include root" FORCE)
    set(DEAC_HIPBLAS_LIBRARY "${DEAC_FIXTURE_HIPBLAS_LIBRARY}"
        CACHE FILEPATH "mock hipBLAS library" FORCE)
elseif(DEAC_FIXTURE_HIPBLAS_MODE STREQUAL "blas_off")
    set(GPU_BACKEND hip CACHE STRING "fixture backend" FORCE)
    set(USE_BLAS OFF CACHE BOOL "fixture BLAS mode" FORCE)
    set(CMAKE_DISABLE_FIND_PACKAGE_hipblas TRUE CACHE BOOL
        "disable fixture hipBLAS package discovery" FORCE)
elseif(DEAC_FIXTURE_HIPBLAS_MODE STREQUAL "non_hip")
    set(GPU_BACKEND sycl CACHE STRING "fixture backend" FORCE)
    set(USE_BLAS ON CACHE BOOL "fixture BLAS mode" FORCE)
    set(CMAKE_DISABLE_FIND_PACKAGE_hipblas TRUE CACHE BOOL
        "disable fixture hipBLAS package discovery" FORCE)
elseif(NOT DEAC_FIXTURE_HIPBLAS_MODE STREQUAL "")
    message(FATAL_ERROR "unknown mock hipBLAS receipt mode")
endif()

# Exercise a backend transition through ordinary CXX compilation without
# requiring a SYCL compiler.  These definitions are the effective File API
# evidence that the regenerated graph followed the changed backend.
if(GPU_BACKEND STREQUAL "sycl")
    target_compile_definitions(receipt_probe PRIVATE USE_GPU=1 USE_SYCL=1)
    if(DEAC_FIXTURE_PARALLEL_CONSUMERS)
        target_compile_definitions(
            receipt_probe_two PRIVATE USE_GPU=1 USE_SYCL=1)
    endif()
endif()

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/DeacHipblas.cmake")
set(DEAC_FIXTURE_HIPBLAS_CONSUMERS receipt_probe)
if(DEAC_FIXTURE_PARALLEL_CONSUMERS)
    list(APPEND DEAC_FIXTURE_HIPBLAS_CONSUMERS receipt_probe_two)
endif()
deac_target_link_hipblas(${DEAC_FIXTURE_HIPBLAS_CONSUMERS})
if(DEAC_FIXTURE_HIPBLAS_CONTRACT_ATTACK STREQUAL "duplicate-provider")
    get_target_property(DEAC_FIXTURE_ATTACK_PROVIDER
        deac_hipblas_link_contract DEAC_HIPBLAS_PROVIDER_TARGET)
    set_property(TARGET deac_hipblas_link_contract PROPERTY
        INTERFACE_LINK_LIBRARIES
        "${DEAC_FIXTURE_ATTACK_PROVIDER};${DEAC_FIXTURE_ATTACK_PROVIDER}")
elseif(NOT DEAC_FIXTURE_HIPBLAS_CONTRACT_ATTACK STREQUAL "")
    message(FATAL_ERROR "unknown mock hipBLAS contract mutation")
endif()

set(DEAC_FIXTURE_RECEIPT_DEPENDENCIES receipt_dependency)
set(DEAC_FIXTURE_REQUIRED_LINK_LIBRARY_NAMES)
set(DEAC_FIXTURE_REQUIRED_LINK_LIBRARY_ARTIFACTS)
if(NOT DEAC_FIXTURE_CONTROL_LINK_ARTIFACT STREQUAL "")
    list(APPEND DEAC_FIXTURE_REQUIRED_LINK_LIBRARY_NAMES control-provider)
    list(APPEND DEAC_FIXTURE_REQUIRED_LINK_LIBRARY_ARTIFACTS
        "${DEAC_FIXTURE_CONTROL_LINK_ARTIFACT}")
endif()
if(USE_BLAS AND GPU_BACKEND STREQUAL "hip")
    if(NOT TARGET deac_hipblas_link_contract)
        message(FATAL_ERROR "mock HIP+BLAS contract is absent")
    endif()
    _deac_hipblas_validate_link_contract(deac_hipblas_link_contract)
    list(APPEND DEAC_FIXTURE_RECEIPT_DEPENDENCIES
        deac_hipblas_link_contract)
    get_target_property(DEAC_FIXTURE_HIPBLAS_PROVIDER_TARGET
        deac_hipblas_link_contract DEAC_HIPBLAS_PROVIDER_TARGET)
    if(NOT TARGET "${DEAC_FIXTURE_HIPBLAS_PROVIDER_TARGET}")
        message(FATAL_ERROR "mock HIP+BLAS provider target is absent")
    endif()
    if(DEAC_FIXTURE_HIPBLAS_MODE STREQUAL "imported_target" AND
            NOT DEAC_FIXTURE_HIPBLAS_PROVIDER_TARGET STREQUAL
                "deac_fixture_hipblas_provider")
        message(FATAL_ERROR
            "mock imported provider alias was not canonicalized")
    endif()
    get_target_property(DEAC_FIXTURE_HIPBLAS_PROVIDER_ARTIFACT
        deac_hipblas_link_contract DEAC_HIPBLAS_PROVIDER_ARTIFACT)
    if(NOT DEAC_FIXTURE_HIPBLAS_PROVIDER_ARTIFACT OR
            DEAC_FIXTURE_HIPBLAS_PROVIDER_ARTIFACT MATCHES "-NOTFOUND$")
        message(FATAL_ERROR "mock HIP+BLAS provider artifact is absent")
    endif()
    if(NOT DEAC_FIXTURE_EXPECTED_HIPBLAS_ARTIFACT STREQUAL "" AND
            NOT DEAC_FIXTURE_HIPBLAS_PROVIDER_ARTIFACT STREQUAL
                DEAC_FIXTURE_EXPECTED_HIPBLAS_ARTIFACT)
        message(FATAL_ERROR
            "mock HIP+BLAS receipt resolver selected the wrong artifact: "
            "${DEAC_FIXTURE_HIPBLAS_PROVIDER_ARTIFACT}")
    endif()
    list(APPEND DEAC_FIXTURE_REQUIRED_LINK_LIBRARY_NAMES
        "${DEAC_FIXTURE_HIPBLAS_PROVIDER_TARGET}")
    list(APPEND DEAC_FIXTURE_REQUIRED_LINK_LIBRARY_ARTIFACTS
        "${DEAC_FIXTURE_HIPBLAS_PROVIDER_ARTIFACT}")
elseif(TARGET deac_hipblas_link_contract)
    message(FATAL_ERROR "mock no-op mode created a hipBLAS contract")
endif()

if(DEAC_FIXTURE_LINK_LIBRARY_ATTACK STREQUAL "unequal-lists")
    list(APPEND DEAC_FIXTURE_REQUIRED_LINK_LIBRARY_NAMES extra-provider)
elseif(DEAC_FIXTURE_LINK_LIBRARY_ATTACK STREQUAL "duplicate-name")
    list(APPEND DEAC_FIXTURE_REQUIRED_LINK_LIBRARY_NAMES
        "${DEAC_FIXTURE_HIPBLAS_PROVIDER_TARGET}")
    list(APPEND DEAC_FIXTURE_REQUIRED_LINK_LIBRARY_ARTIFACTS
        "${CMAKE_COMMAND}")
elseif(DEAC_FIXTURE_LINK_LIBRARY_ATTACK STREQUAL "duplicate-artifact")
    list(APPEND DEAC_FIXTURE_REQUIRED_LINK_LIBRARY_NAMES extra-provider)
    list(APPEND DEAC_FIXTURE_REQUIRED_LINK_LIBRARY_ARTIFACTS
        "${DEAC_FIXTURE_HIPBLAS_PROVIDER_ARTIFACT}")
elseif(DEAC_FIXTURE_LINK_LIBRARY_ATTACK STREQUAL "missing-artifact")
    set(DEAC_FIXTURE_REQUIRED_LINK_LIBRARY_ARTIFACTS
        "${CMAKE_CURRENT_BINARY_DIR}/missing-hipblas-provider.a")
elseif(NOT DEAC_FIXTURE_LINK_LIBRARY_ATTACK STREQUAL "")
    message(FATAL_ERROR "unknown build-receipt link-library input mutation")
endif()

if(DEAC_FIXTURE_RULE_ATTACK STREQUAL "compile-and")
    string(APPEND CMAKE_CXX_COMPILE_OBJECT
        " && \\\"${CMAKE_COMMAND}\\\" -E false")
elseif(DEAC_FIXTURE_RULE_ATTACK STREQUAL "link-or")
    string(APPEND CMAKE_CXX_LINK_EXECUTABLE
        " || \\\"${CMAKE_COMMAND}\\\" -E true")
elseif(DEAC_FIXTURE_RULE_ATTACK STREQUAL "archive-pipe")
    string(APPEND CMAKE_CXX_ARCHIVE_CREATE
        " | \\\"${CMAKE_COMMAND}\\\" -E true")
elseif(DEAC_FIXTURE_RULE_ATTACK STREQUAL "quoted-pipe")
    string(APPEND CMAKE_CXX_COMPILE_OBJECT
        " \\\"-DDEAC_RULE_LITERAL=literal|argument\\\"")
elseif(DEAC_FIXTURE_RULE_ATTACK STREQUAL "compile-semicolon")
    string(APPEND CMAKE_CXX_COMPILE_OBJECT " ; deac_receipt_attack")
elseif(DEAC_FIXTURE_RULE_ATTACK STREQUAL "quoted-semicolon")
    string(APPEND CMAKE_CXX_COMPILE_OBJECT
        " \\\"-DDEAC_RULE_LITERAL=literal\\\\;argument\\\"")
elseif(DEAC_FIXTURE_RULE_ATTACK STREQUAL "escaped-semicolon")
    string(APPEND CMAKE_CXX_COMPILE_OBJECT
        " -DDEAC_RULE_LITERAL=literal\\\\;argument")
elseif(DEAC_FIXTURE_RULE_ATTACK STREQUAL "compile-dollar-paren")
    string(APPEND CMAKE_CXX_COMPILE_OBJECT " $(deac_receipt_attack)")
elseif(DEAC_FIXTURE_RULE_ATTACK STREQUAL "compile-backtick")
    string(APPEND CMAKE_CXX_COMPILE_OBJECT " `deac_receipt_attack`")
elseif(DEAC_FIXTURE_RULE_ATTACK STREQUAL "escaped-backtick")
    string(APPEND CMAKE_CXX_COMPILE_OBJECT
        " -DDEAC_RULE_LITERAL=literal\\\\`argument")
elseif(DEAC_FIXTURE_RULE_ATTACK STREQUAL "compile-dollar-variable")
    string(APPEND CMAKE_CXX_COMPILE_OBJECT
        " -DDEAC_RULE_PROBE=$DEAC_RULE_PROBE")
elseif(DEAC_FIXTURE_RULE_ATTACK STREQUAL "compile-glob")
    string(APPEND CMAKE_CXX_COMPILE_OBJECT " -include *.hpp")
elseif(DEAC_FIXTURE_RULE_ATTACK STREQUAL "compile-redirection")
    string(APPEND CMAKE_CXX_COMPILE_OBJECT " > deac_receipt_attack.log")
elseif(DEAC_FIXTURE_RULE_ATTACK STREQUAL "compile-comment")
    string(APPEND CMAKE_CXX_COMPILE_OBJECT " # deac_receipt_attack")
elseif(DEAC_FIXTURE_RULE_ATTACK STREQUAL "compile-process-substitution")
    string(APPEND CMAKE_CXX_COMPILE_OBJECT " <(deac_receipt_attack)")
elseif(DEAC_FIXTURE_RULE_ATTACK STREQUAL "link-dollar-variable")
    string(APPEND CMAKE_CXX_LINK_EXECUTABLE " $DEAC_RULE_LINK_PROBE")
elseif(DEAC_FIXTURE_RULE_ATTACK STREQUAL "archive-glob")
    string(APPEND CMAKE_CXX_ARCHIVE_CREATE " *.deac-archive-input")
elseif(DEAC_FIXTURE_RULE_ATTACK STREQUAL "quoted-dollar")
    string(APPEND CMAKE_CXX_COMPILE_OBJECT
        " '-DDEAC_RULE_LITERAL=literal$argument'")
elseif(DEAC_FIXTURE_RULE_ATTACK STREQUAL "quoted-backtick")
    string(APPEND CMAKE_CXX_COMPILE_OBJECT
        " '-DDEAC_RULE_LITERAL=\\\"literal`argument\\\"'")
elseif(DEAC_FIXTURE_RULE_ATTACK STREQUAL "quoted-glob")
    string(APPEND CMAKE_CXX_COMPILE_OBJECT
        " \\\"-DDEAC_RULE_LITERAL=literal*argument\\\"")
elseif(DEAC_FIXTURE_RULE_ATTACK STREQUAL "escaped-dollar")
    string(APPEND CMAKE_CXX_COMPILE_OBJECT
        " -DDEAC_RULE_LITERAL=literal\\\\$argument")
elseif(DEAC_FIXTURE_RULE_ATTACK STREQUAL "escaped-glob")
    string(APPEND CMAKE_CXX_COMPILE_OBJECT
        " -DDEAC_RULE_LITERAL=literal\\\\*argument")
elseif(NOT DEAC_FIXTURE_RULE_ATTACK STREQUAL "")
    message(FATAL_ERROR "unknown fixture rule attack")
endif()

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/DeacBuildReceipt.cmake")
if(DEAC_FIXTURE_NATIVE_LANGUAGE_SHAPE MATCHES "^(CUDA|HIP)$")
    # Native toolchains are unavailable in this fixture environment.  Shape
    # the internal query, then exercise the normal target integration below.
    function(_deac_build_receipt_query_enabled_languages output)
        set(${output}
            "CXX;${DEAC_FIXTURE_NATIVE_LANGUAGE_SHAPE}" PARENT_SCOPE)
    endfunction()
elseif(NOT DEAC_FIXTURE_NATIVE_LANGUAGE_SHAPE STREQUAL "")
    message(FATAL_ERROR "invalid native language shape")
endif()
set(DEAC_FIXTURE_CACHE_KEYS
    CMAKE_AR
    CMAKE_BUILD_TYPE
    CMAKE_CONFIGURATION_TYPES
    CMAKE_CXX_COMPILER
    CMAKE_CXX_FLAGS
    CMAKE_DISABLE_FIND_PACKAGE_hipblas
    CMAKE_EXE_LINKER_FLAGS
    CMAKE_GENERATOR
    CMAKE_HOME_DIRECTORY
    CMAKE_PREFIX_PATH
    CMAKE_RANLIB
    DEAC_FIXTURE_CONTROL_CACHE
    DEAC_HIPBLAS_INCLUDE_DIR
    DEAC_HIPBLAS_LIBRARY
    GPU_BACKEND)
list(APPEND DEAC_FIXTURE_CACHE_KEYS
    HIP_RUNTIME_INCLUDE_DIR
    USE_BLAS
    hipblas_DIR
    hipblas_ROOT)
set(DEAC_FIXTURE_RECEIPT
    "${CMAKE_CURRENT_BINARY_DIR}/receipt/$<CONFIG>/build-receipt.json")
if(DEAC_FIXTURE_RECEIPT_ATTACK STREQUAL "hidden-config")
    set(DEAC_FIXTURE_RECEIPT
        "${CMAKE_CURRENT_BINARY_DIR}/receipt/$<$<BOOL:0>:$<CONFIG>>/build-receipt.json")
elseif(NOT DEAC_FIXTURE_RECEIPT_ATTACK STREQUAL "")
    message(FATAL_ERROR "unknown build-receipt output expression mutation")
endif()
deac_target_add_build_receipt(receipt_probe
    SOURCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.."
    GENERATED_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated/receipt"
    IDENTITY_NAME fixture
    RECEIPT "${DEAC_FIXTURE_RECEIPT}"
    BACKEND "${GPU_BACKEND}"
    CACHE_KEYS ${DEAC_FIXTURE_CACHE_KEYS}
    DEPENDENCY_TARGETS ${DEAC_FIXTURE_RECEIPT_DEPENDENCIES}
    REQUIRED_LINK_LIBRARY_NAMES
        ${DEAC_FIXTURE_REQUIRED_LINK_LIBRARY_NAMES}
    REQUIRED_LINK_LIBRARY_ARTIFACTS
        ${DEAC_FIXTURE_REQUIRED_LINK_LIBRARY_ARTIFACTS})
add_custom_target(deac_fixture_generate_receipt
    DEPENDS "${DEAC_BUILD_RECEIPT_REFRESH}")
if(DEAC_FIXTURE_SHARED_FINGERPRINT_ATTACK STREQUAL "conflict")
    if(NOT DEAC_FIXTURE_PARALLEL_CONSUMERS OR
            NOT DEAC_FIXTURE_SHARED_DEPENDENCY)
        message(FATAL_ERROR
            "shared fingerprint attack requires two shared consumers")
    endif()
    set_property(TARGET receipt_dependency PROPERTY
        DEAC_BUILD_RECEIPT_INJECTED_TOOLCHAIN_FINGERPRINT_SHA256
        "0000000000000000000000000000000000000000000000000000000000000000")
elseif(NOT DEAC_FIXTURE_SHARED_FINGERPRINT_ATTACK STREQUAL "")
    message(FATAL_ERROR "unknown shared fingerprint attack")
endif()
if(DEAC_FIXTURE_PARALLEL_CONSUMERS)
    if(DEAC_FIXTURE_SHARED_DEPENDENCY)
        set(DEAC_FIXTURE_SECOND_RECEIPT_DEPENDENCIES receipt_dependency)
    else()
        set(DEAC_FIXTURE_SECOND_RECEIPT_DEPENDENCIES receipt_dependency_two)
    endif()
    if(TARGET deac_hipblas_link_contract)
        list(APPEND DEAC_FIXTURE_SECOND_RECEIPT_DEPENDENCIES
            deac_hipblas_link_contract)
    endif()
    deac_target_add_build_receipt(receipt_probe_two
        SOURCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.."
        GENERATED_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated/receipt"
        IDENTITY_NAME fixture
        RECEIPT
            "${CMAKE_CURRENT_BINARY_DIR}/receipt/$<CONFIG>/build-receipt-two.json"
        BACKEND "${GPU_BACKEND}"
        CACHE_KEYS ${DEAC_FIXTURE_CACHE_KEYS}
        DEPENDENCY_TARGETS ${DEAC_FIXTURE_SECOND_RECEIPT_DEPENDENCIES}
        REQUIRED_LINK_LIBRARY_NAMES
            ${DEAC_FIXTURE_REQUIRED_LINK_LIBRARY_NAMES}
        REQUIRED_LINK_LIBRARY_ARTIFACTS
            ${DEAC_FIXTURE_REQUIRED_LINK_LIBRARY_ARTIFACTS})
endif()

# These mutations intentionally happen after every receipt registration.  The
# module's top-level deferred seal must observe final directory/target state,
# not only the snapshot taken inside deac_target_add_build_receipt().
if(DEAC_FIXTURE_LATE_ATTACK STREQUAL "launcher")
    set_property(TARGET receipt_probe PROPERTY CXX_COMPILER_LAUNCHER
        "${CMAKE_COMMAND};-E;env")
elseif(DEAC_FIXTURE_LATE_ATTACK STREQUAL "rule")
    string(APPEND CMAKE_CXX_COMPILE_OBJECT
        " '-DDEAC_RULE_LITERAL=late-mutation'")
elseif(DEAC_FIXTURE_LATE_ATTACK STREQUAL "ipo")
    set_property(TARGET receipt_probe PROPERTY
        INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
elseif(DEAC_FIXTURE_LATE_ATTACK STREQUAL "link-what-you-use")
    set_property(TARGET receipt_probe PROPERTY LINK_WHAT_YOU_USE TRUE)
elseif(DEAC_FIXTURE_LATE_ATTACK STREQUAL "interface-literal")
    add_library(deac_fixture_late_interface INTERFACE)
    target_link_libraries(receipt_probe PRIVATE deac_fixture_late_interface)
elseif(DEAC_FIXTURE_LATE_ATTACK STREQUAL "interface-conditional")
    add_library(deac_fixture_late_interface INTERFACE)
    target_link_libraries(receipt_probe PRIVATE
        "$<$<CONFIG:Release>:deac_fixture_late_interface>")
elseif(DEAC_FIXTURE_LATE_ATTACK STREQUAL "archive-tool")
    set(CMAKE_AR "${CMAKE_COMMAND}")
elseif(DEAC_FIXTURE_LATE_ATTACK STREQUAL "archive-index-tool")
    set(CMAKE_RANLIB "${CMAKE_COMMAND}")
elseif(DEAC_FIXTURE_LATE_ATTACK STREQUAL "configuration-case-collision")
    set(CMAKE_CONFIGURATION_TYPES "Debug;DEBUG")
    set(CMAKE_CONFIGURATION_TYPES "Debug;DEBUG" CACHE STRING
        "late invalid configuration list" FORCE)
elseif(DEAC_FIXTURE_LATE_ATTACK STREQUAL "configuration-list")
    set(CMAKE_CONFIGURATION_TYPES "Release;Debug")
    set(CMAKE_CONFIGURATION_TYPES "Release;Debug" CACHE STRING
        "late configuration list" FORCE)
elseif(DEAC_FIXTURE_LATE_ATTACK STREQUAL "build-type")
    set(CMAKE_BUILD_TYPE "Debug")
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "late build type" FORCE)
elseif(DEAC_FIXTURE_LATE_ATTACK STREQUAL "shadowed-build-type-cache")
    set(CMAKE_BUILD_TYPE "Release")
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "shadowed build type" FORCE)
elseif(DEAC_FIXTURE_LATE_ATTACK STREQUAL "shadowed-configuration-cache")
    set(CMAKE_CONFIGURATION_TYPES "Debug;Release")
    set(CMAKE_CONFIGURATION_TYPES "Release" CACHE STRING
        "shadowed configuration list" FORCE)
elseif(NOT DEAC_FIXTURE_LATE_ATTACK STREQUAL "")
    message(FATAL_ERROR "unknown post-registration build-receipt mutation")
endif()

if(DEAC_FIXTURE_DEPENDENCY_FINGERPRINT_ATTACK STREQUAL "remove")
    set_property(TARGET receipt_dependency PROPERTY COMPILE_DEFINITIONS "")
elseif(DEAC_FIXTURE_DEPENDENCY_FINGERPRINT_ATTACK STREQUAL "duplicate")
    target_compile_definitions(receipt_dependency PRIVATE
        "DEAC_BUILD_TOOLCHAIN_FINGERPRINT_SHA256=${DEAC_BUILD_RECEIPT_TOOLCHAIN_FINGERPRINT}")
elseif(DEAC_FIXTURE_DEPENDENCY_FINGERPRINT_ATTACK STREQUAL "conflict")
    target_compile_definitions(receipt_dependency PRIVATE
        "DEAC_BUILD_TOOLCHAIN_FINGERPRINT_SHA256=0000000000000000000000000000000000000000000000000000000000000000")
elseif(DEAC_FIXTURE_DEPENDENCY_FINGERPRINT_ATTACK STREQUAL "conflict-function")
    target_compile_definitions(receipt_dependency PRIVATE
        "DEAC_BUILD_TOOLCHAIN_FINGERPRINT_SHA256(x)=bad")
elseif(DEAC_FIXTURE_DEPENDENCY_FINGERPRINT_ATTACK STREQUAL "conflict-spaced")
    target_compile_definitions(receipt_dependency PRIVATE
        "DEAC_BUILD_TOOLCHAIN_FINGERPRINT_SHA256 =bad")
elseif(DEAC_FIXTURE_DEPENDENCY_FINGERPRINT_ATTACK STREQUAL "longer-name")
    target_compile_definitions(receipt_dependency PRIVATE
        "DEAC_BUILD_TOOLCHAIN_FINGERPRINT_SHA256_EXTRA=allowed")
elseif(NOT DEAC_FIXTURE_DEPENDENCY_FINGERPRINT_ATTACK STREQUAL "")
    message(FATAL_ERROR "unknown dependency fingerprint mutation")
endif()
""",
    )
    write(
        fixture_source / "src" / "subdirectory_dependency" / "CMakeLists.txt",
        """if(DEAC_FIXTURE_SUBDIRECTORY_DEPENDENCY_MODE STREQUAL "launcher")
    set_property(DIRECTORY PROPERTY RULE_LAUNCH_COMPILE
        "${CMAKE_COMMAND};-E;env")
elseif(DEAC_FIXTURE_SUBDIRECTORY_DEPENDENCY_MODE STREQUAL "rule")
    string(APPEND CMAKE_CXX_COMPILE_OBJECT
        " '-DDEAC_SUBDIRECTORY_RULE_LITERAL=safe'")
elseif(DEAC_FIXTURE_SUBDIRECTORY_DEPENDENCY_MODE STREQUAL "compiler")
    set(CMAKE_CXX_COMPILER "${CMAKE_COMMAND}")
elseif(NOT DEAC_FIXTURE_SUBDIRECTORY_DEPENDENCY_MODE STREQUAL "normal")
    message(FATAL_ERROR "invalid subdirectory dependency attribution mode")
endif()

add_library(receipt_dependency STATIC ../dependency.cpp)
""",
    )


def create_parent_mutation_fixture_source(solver_source, fixture_source):
    child_source = fixture_source / "src" / "child"
    create_fixture_source(solver_source, child_source)
    write(
        fixture_source / "src" / "CMakeLists.txt",
        """cmake_minimum_required(VERSION 3.27)
project(deac_parent_mutation_fixture LANGUAGES NONE)
set(DEAC_FIXTURE_PARENT_MUTATION "launcher" CACHE STRING
    "parent mutation after child receipt registration")

add_subdirectory(child/src child-build)
if(NOT TARGET receipt_probe)
    message(FATAL_ERROR "child receipt target is absent")
endif()

# This parent-directory rule does not govern the child target.  A seal running
# from the top level must query the captured child directory, then separately
# catch the selected mutation made by its parent.
if(DEAC_FIXTURE_PARENT_MUTATION STREQUAL "launcher")
    string(APPEND CMAKE_CXX_COMPILE_OBJECT
        " $DEAC_PARENT_RULE_MUST_NOT_BE_QUERIED")
    set_property(TARGET receipt_probe PROPERTY CXX_COMPILER_LAUNCHER
        "${CMAKE_COMMAND};-E;env")
elseif(DEAC_FIXTURE_PARENT_MUTATION STREQUAL "build-type")
    set(CMAKE_BUILD_TYPE "Debug")
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "late parent build type" FORCE)
elseif(DEAC_FIXTURE_PARENT_MUTATION STREQUAL "configuration-list")
    set(CMAKE_CONFIGURATION_TYPES "Release")
    set(CMAKE_CONFIGURATION_TYPES "Release" CACHE STRING
        "late parent configuration list" FORCE)
else()
    message(FATAL_ERROR "unknown parent mutation")
endif()
""",
    )


def compiler_shim_contents(real_compiler, marker):
    return (
        "#!/bin/sh\n"
        f"# receipt fixture compiler marker: {marker}\n"
        f"exec {shlex.quote(str(real_compiler))} \"$@\"\n"
    )


def archive_shim_contents(real_archiver, marker, invocation_log):
    return (
        "#!/bin/sh\n"
        f"# receipt fixture archiver marker: {marker}\n"
        f"printf '%s\\n' \"$*\" >> {shlex.quote(str(invocation_log))}\n"
        f"exec {shlex.quote(str(real_archiver))} \"$@\"\n"
    )


def configure(cmake, source, build, compiler, *extra, check=True):
    return run(
        [
            cmake,
            "-S",
            source / "src",
            "-B",
            build,
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_CXX_COMPILER={compiler}",
            "-DGPU_BACKEND=none",
            *extra,
        ],
        source,
        check=check,
    )


def build(
    cmake,
    build_directory,
    *,
    config=None,
    parallel=1,
    target=None,
    check=True,
    environment=None,
):
    command = [
        cmake,
        "--build",
        build_directory,
        "--parallel",
        str(parallel),
        "--verbose",
    ]
    if config is not None:
        command.extend(["--config", config])
    if target is not None:
        command.extend(["--target", target])
    return run(
        command,
        build_directory,
        check=check,
        environment=environment,
    )


def parse_receipt(path):
    raw = path.read_text(encoding="utf-8")
    document = json.loads(raw)
    canonical = json.dumps(document, ensure_ascii=False, separators=(",", ":"))
    if raw != canonical + "\n":
        raise AssertionError("fixture receipt is not canonical JSON")
    payload = json.dumps(
        document["receipt"], ensure_ascii=False, separators=(",", ":")
    )
    expected = hashlib.sha256(payload.encode()).hexdigest()
    if document["receipt_sha256"] != expected:
        raise AssertionError("fixture receipt digest does not bind its payload")
    expected_keys = [
        "archive_tools",
        "backend",
        "build_system",
        "cache_entries",
        "compile_groups",
        "link",
        "source_identity",
        "target",
        "target_dependencies",
        "toolchains",
    ]
    if list(document["receipt"]) != expected_keys:
        raise AssertionError("fixture receipt payload keys are not canonical")
    return document


def archive_tool(document, name):
    tools = document["receipt"]["archive_tools"]
    if [tool.get("name") for tool in tools] != ["CMAKE_AR", "CMAKE_RANLIB"]:
        raise AssertionError(f"unexpected archive tools: {tools!r}")
    tool = next(tool for tool in tools if tool["name"] == name)
    if list(tool) != ["name", "path", "real_path", "sha256"]:
        raise AssertionError(f"noncanonical archive tool: {tool!r}")
    if len(tool["sha256"]) != 64 or any(
        character not in "0123456789abcdef" for character in tool["sha256"]
    ):
        raise AssertionError(f"invalid archive tool digest: {tool!r}")
    if not tool["real_path"].startswith("<"):
        if (
            not tool["path"].startswith("<")
            and Path(tool["path"]).resolve() != Path(tool["real_path"])
        ):
            raise AssertionError(f"archive tool paths disagree: {tool!r}")
        if sha256_file(tool["real_path"]) != tool["sha256"]:
            raise AssertionError(f"archive tool digest disagrees: {tool!r}")
    return tool


def assert_no_git_source(document, version):
    expected = {
        "schema_version": 1,
        "semantic_version": version,
        "source_sha": None,
        "source_state": "unavailable",
    }
    if document["receipt"]["source_identity"] != expected:
        raise AssertionError(
            "no-Git fixture acquired unexpected source identity: "
            f"{document['receipt']['source_identity']!r}"
        )


def assert_static_dependency(
    document, fingerprint, expected_name="receipt_dependency"
):
    dependencies = document["receipt"]["target_dependencies"]
    matches = [
        dependency
        for dependency in dependencies
        if dependency.get("name") == expected_name
    ]
    if len(matches) != 1:
        raise AssertionError(
            f"expected one {expected_name} dependency, got {dependencies!r}"
        )
    dependency = matches[0]
    if dependency["type"] != "STATIC_LIBRARY":
        raise AssertionError(f"dependency is not a static library: {dependency!r}")
    if dependency["link"] is not None or not isinstance(dependency["archive"], dict):
        raise AssertionError(f"static dependency archive is missing: {dependency!r}")
    if not isinstance(dependency["archive"].get("command_fragments"), list):
        raise TypeError("static dependency archive fragments are malformed")
    groups = dependency["compile_groups"]
    sources = [source for group in groups for source in group["sources"]]
    expected_source = expected_name.removeprefix("receipt_") + ".cpp"
    if not any(source.endswith("/" + expected_source) for source in sources):
        raise AssertionError(
            f"static dependency source {expected_source!r} is absent from the receipt"
        )
    definition = f"DEAC_BUILD_TOOLCHAIN_FINGERPRINT_SHA256={fingerprint}"
    if any(group["definitions"].count(definition) != 1 for group in groups):
        raise AssertionError("static dependency does not bind the tool fingerprint")


def assert_hipblas_contract(
    document,
    *,
    static_dependency,
    mode,
    provider_library,
    include_directory,
    package_directory,
    provider_root,
):
    active = mode in {"imported_target", "compatibility_library"}
    expected_dependencies = {static_dependency: "STATIC_LIBRARY"}
    if active:
        expected_dependencies["deac_hipblas_link_contract"] = "INTERFACE_LIBRARY"
    dependencies = document["receipt"]["target_dependencies"]
    names = [dependency.get("name") for dependency in dependencies]
    if names != sorted(expected_dependencies):
        raise AssertionError(
            f"{mode} receipt has unexpected dependencies: {dependencies!r}"
        )
    for dependency in dependencies:
        expected_type = expected_dependencies[dependency["name"]]
        if dependency["type"] != expected_type:
            raise AssertionError(
                f"{mode} dependency has wrong type: {dependency!r}"
            )
    if active:
        interface = next(
            dependency
            for dependency in dependencies
            if dependency["name"] == "deac_hipblas_link_contract"
        )
        if (
            interface["archive"] is not None
            or interface["compile_groups"] != []
            or interface["link"] is not None
        ):
            raise AssertionError(
                f"{mode} interface dependency has build artifacts: {interface!r}"
            )

    link_fragments = document["receipt"]["link"]["command_fragments"]
    libraries = [
        fragment["fragment"]
        for fragment in link_fragments
        if fragment["role"] == "libraries"
    ]
    provider_matches = [
        fragment
        for fragment in libraries
        if Path(fragment).resolve() == provider_library.resolve()
    ]
    expected_provider_count = 1 if active else 0
    if len(provider_matches) != expected_provider_count:
        raise AssertionError(
            f"{mode} receipt expected provider {expected_provider_count} time(s), "
            f"got {libraries!r}"
        )
    hipblas_fragments = [
        fragment for fragment in libraries if "hipblas" in fragment.casefold()
    ]
    if len(hipblas_fragments) != expected_provider_count:
        raise AssertionError(
            f"{mode} receipt has unexpected hipBLAS fragments: {libraries!r}"
        )

    expected_backend = "sycl" if mode == "non_hip" else "hip"
    if document["receipt"]["backend"] != expected_backend:
        raise AssertionError(f"{mode} receipt records the wrong backend")
    if cache_value(document, "HIP_RUNTIME_INCLUDE_DIR") != str(include_directory):
        raise AssertionError(f"{mode} receipt omits the HIP runtime include")
    if cache_value(document, "hipblas_ROOT") != str(provider_root):
        raise AssertionError(f"{mode} receipt omits hipblas_ROOT")
    disable_entry = cache_entry(
        document, "CMAKE_DISABLE_FIND_PACKAGE_hipblas"
    )
    expected_disable = "FALSE" if mode == "imported_target" else "TRUE"
    if disable_entry != {
        "name": "CMAKE_DISABLE_FIND_PACKAGE_hipblas",
        "type": "BOOL",
        "value": expected_disable,
    }:
        raise AssertionError(
            f"{mode} receipt has the wrong package-disable cache input: "
            f"{disable_entry!r}"
        )
    for cache_key in (
        "DEAC_HIPBLAS_INCLUDE_DIR",
        "DEAC_HIPBLAS_LIBRARY",
        "hipblas_DIR",
    ):
        cache_value(document, cache_key)
    if mode == "imported_target":
        if cache_value(document, "hipblas_DIR") != str(package_directory):
            raise AssertionError("imported-target receipt omits hipblas_DIR")
    elif mode == "compatibility_library":
        if cache_value(document, "DEAC_HIPBLAS_INCLUDE_DIR") != str(
            include_directory
        ):
            raise AssertionError("fallback receipt omits the hipBLAS include")
        if cache_value(document, "DEAC_HIPBLAS_LIBRARY") != str(
            provider_library
        ):
            raise AssertionError("fallback receipt omits the hipBLAS library")


def assert_mapped_hipblas_receipt(
    document, *, configuration, provider_library, other_provider_library
):
    expected_dependency = {
        "archive": None,
        "compile_groups": [],
        "link": None,
        "name": "deac_hipblas_link_contract",
        "type": "INTERFACE_LIBRARY",
    }
    dependencies = document["receipt"]["target_dependencies"]
    if dependencies != [expected_dependency]:
        raise AssertionError(
            "mapped hipBLAS receipt does not contain only the synthetic "
            f"contract dependency: {dependencies!r}"
        )

    target = document["receipt"]["target"]
    if target["name"] != "receipt_probe":
        raise AssertionError(f"mapped receipt names the wrong target: {target!r}")
    build_system = document["receipt"]["build_system"]
    generator = build_system["generator"]
    if (
        build_system["configuration"] != configuration
        or generator["name"] != "Ninja Multi-Config"
        or not generator["multi_config"]
    ):
        raise AssertionError(
            f"mapped {configuration} receipt has the wrong build shape: "
            f"{build_system!r}"
        )
    if cache_value(document, "CMAKE_CONFIGURATION_TYPES") != "Debug;Release":
        raise AssertionError(
            "mapped receipt does not bind the exact Debug/Release config set"
        )
    if cache_entry(document, "CMAKE_DISABLE_FIND_PACKAGE_hipblas") != {
        "name": "CMAKE_DISABLE_FIND_PACKAGE_hipblas",
        "type": "BOOL",
        "value": "FALSE",
    }:
        raise AssertionError(
            "mapped package receipt does not attest enabled package discovery"
        )

    library_fragments = [
        fragment["fragment"]
        for fragment in document["receipt"]["link"]["command_fragments"]
        if fragment["role"] == "libraries"
    ]
    libraries = []
    for fragment in library_fragments:
        try:
            libraries.extend(shlex.split(fragment))
        except ValueError as error:
            raise AssertionError(
                f"mapped receipt has invalid native-shell link syntax: {fragment!r}"
            ) from error
    expected_library = str(provider_library.resolve())
    other_library = str(other_provider_library.resolve())
    if libraries.count(expected_library) != 1:
        raise AssertionError(
            f"mapped {configuration} provider is not present exactly once: "
            f"fragments={library_fragments!r}, arguments={libraries!r}"
        )
    if other_library in libraries:
        raise AssertionError(
            f"mapped {configuration} receipt contains the other config's "
            f"provider: {libraries!r}"
        )
    if libraries != [expected_library]:
        raise AssertionError(
            f"mapped {configuration} receipt has unexpected link libraries: "
            f"{libraries!r}"
        )


def assert_embedded_matches(executable, receipt_path):
    endpoint = run([executable], executable.parent)
    if endpoint.stderr:
        raise AssertionError(f"fixture endpoint wrote stderr: {endpoint.stderr!r}")
    if endpoint.stdout != receipt_path.read_text(encoding="utf-8"):
        raise AssertionError("fixture embedded and adjacent receipts disagree")


def assert_build_actions(result, required):
    output = result.stdout + result.stderr
    missing = [needle for needle in required if needle not in output]
    if missing:
        raise AssertionError(
            f"build output omitted required actions {missing!r}:\n{output}"
        )


def receipt_fingerprint(document):
    return document["receipt"]["target"]["toolchain_fingerprint_sha256"]


def cache_entry(document, name):
    matches = [
        entry
        for entry in document["receipt"]["cache_entries"]
        if entry["name"] == name
    ]
    if len(matches) != 1:
        raise AssertionError(f"expected one {name} cache entry, got {matches!r}")
    return matches[0]


def cache_value(document, name):
    return cache_entry(document, name)["value"]


def target_definitions(document):
    return [
        definition
        for group in document["receipt"]["compile_groups"]
        for definition in group["definitions"]
    ]


def create_hipblas_provider(workdir):
    provider_root = workdir / "external-hipblas-provider"
    include_directory = provider_root / "include"
    write(include_directory / "hip" / "hip_runtime.h", "#pragma once\n")
    write(include_directory / "hipblas" / "hipblas.h", "#pragma once\n")
    provider_library = provider_root / "lib" / "libhipblas.a"
    provider_library.parent.mkdir(parents=True, exist_ok=True)
    provider_library.write_bytes(b"!<arch>\n")
    package_directory = provider_root / "lib" / "cmake" / "hipblas"
    write(
        package_directory / "hipblas-config.cmake",
        f"""if(NOT TARGET roc::hipblas)
    add_library(deac_fixture_hipblas_provider UNKNOWN IMPORTED)
    add_library(roc::hipblas ALIAS deac_fixture_hipblas_provider)
    set_target_properties(deac_fixture_hipblas_provider PROPERTIES
        IMPORTED_LOCATION {json.dumps(str(provider_library))}
        INTERFACE_INCLUDE_DIRECTORIES {json.dumps(str(include_directory))})
endif()
""",
    )
    return provider_root, include_directory, provider_library, package_directory


def create_configless_mapped_hipblas_provider(workdir):
    provider_root = workdir / "external-configless-mapped-hipblas-provider"
    include_directory = provider_root / "include"
    write(include_directory / "hip" / "hip_runtime.h", "#pragma once\n")
    write(include_directory / "hipblas" / "hipblas.h", "#pragma once\n")
    configless_library = provider_root / "lib" / "libhipblas-configless.a"
    alternate_library = provider_root / "lib" / "libhipblas-alternate.a"
    configless_library.parent.mkdir(parents=True, exist_ok=True)
    configless_library.write_bytes(b"!<arch>\n")
    alternate_library.write_bytes(b"!<arch>\n")
    package_directory = provider_root / "lib" / "cmake" / "hipblas"
    write(
        package_directory / "hipblas-config.cmake",
        f"""if(NOT TARGET roc::hipblas)
    add_library(deac_fixture_hipblas_provider UNKNOWN IMPORTED)
    add_library(roc::hipblas ALIAS deac_fixture_hipblas_provider)
    set_target_properties(deac_fixture_hipblas_provider PROPERTIES
        IMPORTED_CONFIGURATIONS ALTERNATE_ARCHIVE
        IMPORTED_LOCATION {json.dumps(str(configless_library))}
        IMPORTED_LOCATION_ALTERNATE_ARCHIVE {json.dumps(str(alternate_library))}
        MAP_IMPORTED_CONFIG_RELEASE ";ALTERNATE_ARCHIVE"
        INTERFACE_INCLUDE_DIRECTORIES {json.dumps(str(include_directory))})
endif()
""",
    )
    return (
        provider_root,
        include_directory,
        configless_library,
        alternate_library,
        package_directory,
    )


def create_mapped_hipblas_provider(workdir):
    # Keep both configured artifacts behind a path that the native link line
    # must shell-encode.  The receipt must still compare their decoded paths,
    # not the generator-specific spelling in the File API fragment.
    provider_root = workdir / "external mapped hipblas provider"
    include_directory = provider_root / "include"
    write(include_directory / "hip" / "hip_runtime.h", "#pragma once\n")
    write(include_directory / "hipblas" / "hipblas.h", "#pragma once\n")
    debug_library = provider_root / "lib" / "libhipblas-debug.a"
    release_library = provider_root / "lib" / "libhipblas-release.a"
    debug_library.parent.mkdir(parents=True, exist_ok=True)
    debug_library.write_bytes(b"!<arch>\n")
    release_library.write_bytes(b"!<arch>\n")
    package_directory = provider_root / "lib" / "cmake" / "hipblas"
    write(
        package_directory / "hipblas-config.cmake",
        f"""if(NOT TARGET roc::hipblas)
    add_library(deac_fixture_mapped_hipblas_provider UNKNOWN IMPORTED)
    add_library(roc::hipblas ALIAS deac_fixture_mapped_hipblas_provider)
    set_target_properties(deac_fixture_mapped_hipblas_provider PROPERTIES
        IMPORTED_CONFIGURATIONS "DEBUG_ARCHIVE;RELEASE_ARCHIVE"
        IMPORTED_LOCATION_DEBUG_ARCHIVE {json.dumps(str(debug_library))}
        IMPORTED_LOCATION_RELEASE_ARCHIVE {json.dumps(str(release_library))}
        MAP_IMPORTED_CONFIG_DEBUG DEBUG_ARCHIVE
        MAP_IMPORTED_CONFIG_RELEASE RELEASE_ARCHIVE
        INTERFACE_INCLUDE_DIRECTORIES {json.dumps(str(include_directory))})
endif()
""",
    )
    return (
        provider_root,
        include_directory,
        debug_library,
        release_library,
        package_directory,
    )


def create_mapped_hipblas_fixture_source(solver_source, fixture_source):
    fixture_source.mkdir(parents=True)
    copy_fixture_modules(solver_source, fixture_source)
    write(fixture_source / "VERSION", "1.2.3\n")
    write(
        fixture_source / "src" / "probe.cpp",
        """#include "build_identity.hpp"
#include <iostream>

int main() {
    std::cout << deac_build_identity::build_receipt_json() << '\\n';
    return 0;
}
""",
    )
    write(
        fixture_source / "src" / "CMakeLists.txt",
        """cmake_minimum_required(VERSION 3.27)
project(deac_mapped_hipblas_receipt_fixture LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_DISABLE_FIND_PACKAGE_hipblas FALSE CACHE BOOL
    "disable fixture hipBLAS package discovery" FORCE)
option(DEAC_FIXTURE_DUPLICATE_SELECTED_ARTIFACT
    "duplicate one selected-config required provider artifact" OFF)
foreach(DEAC_FIXTURE_REQUIRED_VARIABLE IN ITEMS
        DEAC_FIXTURE_HIPBLAS_DEBUG_LIBRARY
        DEAC_FIXTURE_HIPBLAS_RELEASE_LIBRARY
        DEAC_FIXTURE_HIPBLAS_PACKAGE_DIR
        HIP_RUNTIME_INCLUDE_DIR)
    if(NOT DEFINED ${DEAC_FIXTURE_REQUIRED_VARIABLE} OR
            "${${DEAC_FIXTURE_REQUIRED_VARIABLE}}" STREQUAL "")
        message(FATAL_ERROR
            "${DEAC_FIXTURE_REQUIRED_VARIABLE} is required")
    endif()
endforeach()
if(NOT "${CMAKE_CONFIGURATION_TYPES}" STREQUAL "Debug;Release")
    message(FATAL_ERROR
        "mapped hipBLAS fixture requires exactly Debug;Release")
endif()
set(GPU_BACKEND hip CACHE STRING "fixture backend" FORCE)
set(USE_BLAS ON CACHE BOOL "fixture BLAS mode" FORCE)
set(hipblas_DIR "${DEAC_FIXTURE_HIPBLAS_PACKAGE_DIR}"
    CACHE PATH "mock hipBLAS package" FORCE)

add_executable(receipt_probe probe.cpp)
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/DeacHipblas.cmake")
deac_target_link_hipblas(receipt_probe)
_deac_hipblas_validate_link_contract(deac_hipblas_link_contract)
get_target_property(DEAC_FIXTURE_HIPBLAS_PROVIDER_TARGET
    deac_hipblas_link_contract DEAC_HIPBLAS_PROVIDER_TARGET)
if(NOT DEAC_FIXTURE_HIPBLAS_PROVIDER_TARGET STREQUAL
        "deac_fixture_mapped_hipblas_provider")
    message(FATAL_ERROR
        "mapped hipBLAS alias did not resolve to its canonical provider: "
        "${DEAC_FIXTURE_HIPBLAS_PROVIDER_TARGET}")
endif()
get_target_property(DEAC_FIXTURE_HIPBLAS_PROVIDER_ARTIFACT
    deac_hipblas_link_contract DEAC_HIPBLAS_PROVIDER_ARTIFACT)
string(CONCAT DEAC_FIXTURE_EXPECTED_PROVIDER_ARTIFACT
    "$<$<CONFIG:Debug>:${DEAC_FIXTURE_HIPBLAS_DEBUG_LIBRARY}>"
    "$<$<CONFIG:Release>:${DEAC_FIXTURE_HIPBLAS_RELEASE_LIBRARY}>")
if(NOT "${DEAC_FIXTURE_HIPBLAS_PROVIDER_ARTIFACT}" STREQUAL
        "${DEAC_FIXTURE_EXPECTED_PROVIDER_ARTIFACT}")
    message(FATAL_ERROR
        "mapped hipBLAS provider artifact is not configuration exact: "
        "${DEAC_FIXTURE_HIPBLAS_PROVIDER_ARTIFACT}")
endif()

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/DeacBuildReceipt.cmake")
set(DEAC_FIXTURE_REQUIRED_LINK_LIBRARY_NAMES
    "${DEAC_FIXTURE_HIPBLAS_PROVIDER_TARGET}")
set(DEAC_FIXTURE_REQUIRED_LINK_LIBRARY_ARTIFACTS
    "${DEAC_FIXTURE_HIPBLAS_PROVIDER_ARTIFACT}")
if(DEAC_FIXTURE_DUPLICATE_SELECTED_ARTIFACT)
    string(CONCAT DEAC_FIXTURE_COLLIDING_PROVIDER_ARTIFACT
        "$<$<CONFIG:Debug>:${DEAC_FIXTURE_HIPBLAS_DEBUG_LIBRARY}>"
        "$<$<CONFIG:Release>:${CMAKE_COMMAND}>")
    list(APPEND DEAC_FIXTURE_REQUIRED_LINK_LIBRARY_NAMES
        deac_fixture_colliding_provider)
    list(APPEND DEAC_FIXTURE_REQUIRED_LINK_LIBRARY_ARTIFACTS
        "${DEAC_FIXTURE_COLLIDING_PROVIDER_ARTIFACT}")
endif()
deac_target_add_build_receipt(receipt_probe
    SOURCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.."
    GENERATED_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated/receipt"
    IDENTITY_NAME mapped_fixture
    RECEIPT
        "${CMAKE_CURRENT_BINARY_DIR}/receipt/$<CONFIG>/build-receipt.json"
    BACKEND "${GPU_BACKEND}"
    CACHE_KEYS
        CMAKE_CONFIGURATION_TYPES
        CMAKE_CXX_COMPILER
        CMAKE_CXX_FLAGS
        CMAKE_DISABLE_FIND_PACKAGE_hipblas
        CMAKE_EXE_LINKER_FLAGS
        CMAKE_GENERATOR
        GPU_BACKEND
        HIP_RUNTIME_INCLUDE_DIR
        USE_BLAS
        hipblas_DIR
    DEPENDENCY_TARGETS deac_hipblas_link_contract
    REQUIRED_LINK_LIBRARY_NAMES
        ${DEAC_FIXTURE_REQUIRED_LINK_LIBRARY_NAMES}
    REQUIRED_LINK_LIBRARY_ARTIFACTS
        ${DEAC_FIXTURE_REQUIRED_LINK_LIBRARY_ARTIFACTS})
add_custom_target(deac_fixture_generate_receipt
    DEPENDS "${DEAC_BUILD_RECEIPT_REFRESH}")
""",
    )


def test_single_config_configuration_rejections(
    cmake, source, workdir, real_compiler
):
    assert_configure_rejected(
        cmake,
        source,
        workdir / "rejected-empty-build-type",
        real_compiler,
        ["-DCMAKE_BUILD_TYPE:STRING="],
        "requires at least one configured build type",
    )
    assert_configure_rejected(
        cmake,
        source,
        workdir / "rejected-list-build-type",
        real_compiler,
        ["-DCMAKE_BUILD_TYPE:STRING=Debug;Release"],
        "single-config generator requires exactly one configured build type",
    )


def test_single_config_refresh_and_replacement(
    cmake, source, workdir, compiler_shim, real_compiler
):
    build_directory = workdir / "single-build"
    write(
        compiler_shim,
        compiler_shim_contents(real_compiler, "first"),
        executable=True,
    )
    configure(cmake, source, build_directory, compiler_shim)
    receipt_actions = [
        "Embedding effective build receipt",
        (
            "Building CXX object "
            "CMakeFiles/receipt_probe.dir/generated/receipt/Release/"
            "fixture_receipt_probe_build_receipt.cpp.o"
        ),
        "Linking CXX executable receipt_probe",
    ]
    first_build = build(cmake, build_directory)
    assert_build_actions(first_build, receipt_actions)
    receipt_path = build_directory / "receipt" / "Release" / "build-receipt.json"
    first_receipt = parse_receipt(receipt_path)
    first_fingerprint = first_receipt["receipt"]["target"][
        "toolchain_fingerprint_sha256"
    ]
    archive_tool(first_receipt, "CMAKE_AR")
    archive_tool(first_receipt, "CMAKE_RANLIB")
    assert_no_git_source(first_receipt, "1.2.3")
    assert_static_dependency(first_receipt, first_fingerprint)
    executable = build_directory / "receipt_probe"
    assert_embedded_matches(executable, receipt_path)

    second_build = build(cmake, build_directory)
    assert_build_actions(second_build, receipt_actions)
    assert_embedded_matches(executable, receipt_path)

    # Refresh a material build-time input without reconfiguring.  The same
    # generated graph must compile the new receipt object before relinking;
    # comparing only the freely replaceable adjacent JSON would miss a stale
    # embedded receipt.
    write(source / "VERSION", "1.2.4\n")
    changed_build = build(cmake, build_directory)
    assert_build_actions(changed_build, receipt_actions)
    changed_receipt = parse_receipt(receipt_path)
    if changed_receipt["receipt_sha256"] == first_receipt["receipt_sha256"]:
        raise AssertionError("material receipt refresh did not change its digest")
    if changed_receipt["receipt"]["source_identity"]["semantic_version"] != "1.2.4":
        raise AssertionError("material receipt refresh retained the old version")
    assert_no_git_source(changed_receipt, "1.2.4")
    assert_embedded_matches(executable, receipt_path)

    write(
        compiler_shim,
        compiler_shim_contents(real_compiler, "replacement"),
        executable=True,
    )
    rejected = build(cmake, build_directory, check=False)
    if rejected.returncode == 0:
        raise AssertionError("persistent compiler replacement was accepted")
    rejected_output = rejected.stdout + rejected.stderr
    if "compiler changed after configuration" not in rejected_output:
        raise AssertionError(
            "compiler replacement failed for an unexpected reason:\n" + rejected_output
        )

    configure(cmake, source, build_directory, compiler_shim)
    replacement_build = build(cmake, build_directory)
    assert_build_actions(
        replacement_build,
        [
            "dependency.cpp",
            "probe.cpp",
            "build_receipt.cpp",
            "receipt_dependency",
            "receipt_probe",
        ],
    )
    replacement_receipt = parse_receipt(receipt_path)
    replacement_fingerprint = replacement_receipt["receipt"]["target"][
        "toolchain_fingerprint_sha256"
    ]
    if replacement_fingerprint == first_fingerprint:
        raise AssertionError("compiler replacement did not change the fingerprint")
    assert_no_git_source(replacement_receipt, "1.2.4")
    assert_static_dependency(replacement_receipt, replacement_fingerprint)
    assert_embedded_matches(build_directory / "receipt_probe", receipt_path)


def test_single_config_ninja(cmake, ninja, source, workdir, real_compiler):
    build_directory = workdir / "ninja-single-build"
    configure(
        cmake,
        source,
        build_directory,
        real_compiler,
        "-G",
        "Ninja",
        f"-DCMAKE_MAKE_PROGRAM={ninja}",
    )
    required = [
        "GenerateDeacBuildReceipt.cmake",
        "fixture_receipt_probe_build_receipt.cpp.o",
        "-o receipt_probe",
    ]
    first_build = build(cmake, build_directory)
    assert_build_actions(first_build, required)
    second_build = build(cmake, build_directory)
    assert_build_actions(second_build, required)

    receipt_path = build_directory / "receipt" / "Release" / "build-receipt.json"
    document = parse_receipt(receipt_path)
    generator = document["receipt"]["build_system"]["generator"]
    if generator["name"] != "Ninja" or generator["multi_config"]:
        raise AssertionError(f"unexpected single-config Ninja receipt: {generator!r}")
    assert_no_git_source(document, "1.2.4")
    assert_static_dependency(document, receipt_fingerprint(document))
    assert_embedded_matches(build_directory / "receipt_probe", receipt_path)


def test_material_flag_reconfiguration(cmake, source, workdir, real_compiler):
    build_directory = workdir / "material-flag-build"
    configure(cmake, source, build_directory, real_compiler)
    build(cmake, build_directory)
    receipt_path = build_directory / "receipt" / "Release" / "build-receipt.json"
    first_receipt = parse_receipt(receipt_path)

    flag = "-DDEAC_FIXTURE_MATERIAL_FLAG=17"
    configure(
        cmake,
        source,
        build_directory,
        real_compiler,
        f"-DCMAKE_CXX_FLAGS={flag}",
    )
    changed_build = build(cmake, build_directory)
    assert_build_actions(
        changed_build,
        [
            flag,
            "dependency.cpp.o",
            "probe.cpp.o",
            "fixture_receipt_probe_build_receipt.cpp.o",
            "Linking CXX static library libreceipt_dependency.a",
            "Linking CXX executable receipt_probe",
        ],
    )
    changed_receipt = parse_receipt(receipt_path)
    if changed_receipt["receipt_sha256"] == first_receipt["receipt_sha256"]:
        raise AssertionError("material flag reconfigure did not change the receipt")
    if receipt_fingerprint(changed_receipt) != receipt_fingerprint(first_receipt):
        raise AssertionError("material flags unexpectedly changed the tool fingerprint")
    if cache_value(changed_receipt, "CMAKE_CXX_FLAGS") != flag:
        raise AssertionError("material flag is absent from the receipt cache")
    effective_compile_fragments = [
        fragment["fragment"]
        for group in changed_receipt["receipt"]["compile_groups"]
        for fragment in group["command_fragments"]
    ]
    if not any(flag in fragment for fragment in effective_compile_fragments):
        raise AssertionError("material flag is absent from effective command data")
    assert_no_git_source(changed_receipt, "1.2.4")
    assert_embedded_matches(build_directory / "receipt_probe", receipt_path)


def test_backend_reconfiguration(cmake, source, workdir, real_compiler):
    build_directory = workdir / "backend-reconfiguration-build"
    receipt_path = (
        build_directory / "receipt" / "Release" / "build-receipt.json"
    )
    executable = build_directory / "receipt_probe"

    configure(cmake, source, build_directory, real_compiler)
    build(cmake, build_directory)
    none_receipt = parse_receipt(receipt_path)
    if none_receipt["receipt"]["backend"] != "none":
        raise AssertionError("initial backend receipt is not none")
    if cache_value(none_receipt, "GPU_BACKEND") != "none":
        raise AssertionError("initial backend cache entry is not none")
    none_definitions = target_definitions(none_receipt)
    if "USE_GPU=1" in none_definitions or "USE_SYCL=1" in none_definitions:
        raise AssertionError(
            f"none backend retained accelerator definitions: {none_definitions!r}"
        )
    assert_embedded_matches(executable, receipt_path)

    # Reconfigure the same graph to a mocked SYCL backend.  The fixture remains
    # an ordinary CXX project, while its effective definitions model the
    # production backend selection closely enough to prove File API refresh.
    configure(
        cmake,
        source,
        build_directory,
        real_compiler,
        "-DGPU_BACKEND:STRING=sycl",
    )
    changed_build = build(cmake, build_directory)
    assert_build_actions(
        changed_build,
        [
            "probe.cpp.o",
            "fixture_receipt_probe_build_receipt.cpp.o",
            "Linking CXX executable receipt_probe",
        ],
    )
    sycl_receipt = parse_receipt(receipt_path)
    if sycl_receipt["receipt"]["backend"] != "sycl":
        raise AssertionError("reconfigured backend receipt is not sycl")
    if cache_value(sycl_receipt, "GPU_BACKEND") != "sycl":
        raise AssertionError("reconfigured backend cache entry is not sycl")
    sycl_definitions = target_definitions(sycl_receipt)
    for definition in ("USE_GPU=1", "USE_SYCL=1"):
        if sycl_definitions.count(definition) != 1:
            raise AssertionError(
                f"reconfigured backend has wrong {definition} definitions: "
                f"{sycl_definitions!r}"
            )
    if sycl_receipt["receipt_sha256"] == none_receipt["receipt_sha256"]:
        raise AssertionError("backend reconfiguration retained the old digest")
    assert_embedded_matches(executable, receipt_path)


def test_archive_tool_reconfiguration(
    cmake, source, workdir, real_compiler, real_archiver
):
    build_directory = workdir / "archive-tool-build"
    tool_directory = workdir / "archive-tools"
    ar_one = tool_directory / "ar-one"
    ar_two = tool_directory / "ar-two"
    ar_one_log = tool_directory / "ar-one.invocations"
    ar_two_log = tool_directory / "ar-two.invocations"
    write(
        ar_one,
        archive_shim_contents(real_archiver, "one", ar_one_log),
        executable=True,
    )
    write(
        ar_two,
        archive_shim_contents(real_archiver, "two", ar_two_log),
        executable=True,
    )

    configure(
        cmake,
        source,
        build_directory,
        real_compiler,
        f"-DDEAC_FIXTURE_ARCHIVER={ar_one}",
    )
    first_build = build(cmake, build_directory)
    assert_build_actions(
        first_build,
        [
            f"{ar_one} qc libreceipt_dependency.a",
            "Linking CXX static library libreceipt_dependency.a",
        ],
    )
    if not ar_one_log.is_file() or not any(
        invocation.startswith("qc libreceipt_dependency.a ")
        for invocation in ar_one_log.read_text(encoding="utf-8").splitlines()
    ):
        raise AssertionError("ar-one did not create the fixture static archive")
    if ar_two_log.exists():
        raise AssertionError("ar-two ran before the archiver reconfiguration")
    receipt_path = build_directory / "receipt" / "Release" / "build-receipt.json"
    first_receipt = parse_receipt(receipt_path)
    first_ar = archive_tool(first_receipt, "CMAKE_AR")
    if first_ar["path"] != str(ar_one):
        raise AssertionError(f"receipt recorded the wrong first archiver: {first_ar!r}")
    if cache_value(first_receipt, "CMAKE_AR") != str(ar_one):
        raise AssertionError("receipt cache omitted the first effective archiver")
    assert_embedded_matches(build_directory / "receipt_probe", receipt_path)

    write(
        ar_one,
        archive_shim_contents(real_archiver, "tampered", ar_one_log),
        executable=True,
    )
    rejected = build(cmake, build_directory, check=False)
    if rejected.returncode == 0:
        raise AssertionError("persistent CMAKE_AR replacement was accepted")
    rejected_output = rejected.stdout + rejected.stderr
    if "CMAKE_AR executable changed after configuration" not in rejected_output:
        raise AssertionError(
            "CMAKE_AR replacement failed for an unexpected reason:\n"
            + rejected_output
        )

    configure(
        cmake,
        source,
        build_directory,
        real_compiler,
        f"-DDEAC_FIXTURE_ARCHIVER={ar_two}",
    )
    second_build = build(cmake, build_directory)
    assert_build_actions(
        second_build,
        [
            f"{ar_two} qc libreceipt_dependency.a",
            "dependency.cpp.o",
            "Linking CXX static library libreceipt_dependency.a",
            "Linking CXX executable receipt_probe",
        ],
    )
    if not ar_two_log.is_file() or not any(
        invocation.startswith("qc libreceipt_dependency.a ")
        for invocation in ar_two_log.read_text(encoding="utf-8").splitlines()
    ):
        raise AssertionError("ar-two did not recreate the fixture static archive")
    second_receipt = parse_receipt(receipt_path)
    second_ar = archive_tool(second_receipt, "CMAKE_AR")
    if second_ar["path"] != str(ar_two) or second_ar["sha256"] != sha256_file(ar_two):
        raise AssertionError(f"receipt recorded the wrong second archiver: {second_ar!r}")
    if cache_value(second_receipt, "CMAKE_AR") != str(ar_two):
        raise AssertionError("receipt cache omitted the second effective archiver")
    if receipt_fingerprint(second_receipt) == receipt_fingerprint(first_receipt):
        raise AssertionError("archiver reconfigure did not invalidate the fingerprint")
    if second_receipt["receipt_sha256"] == first_receipt["receipt_sha256"]:
        raise AssertionError("archiver reconfigure did not change the receipt identity")
    assert_static_dependency(second_receipt, receipt_fingerprint(second_receipt))
    assert_no_git_source(second_receipt, "1.2.4")
    assert_embedded_matches(build_directory / "receipt_probe", receipt_path)


def graph_configurations(ninja, build_directory, configuration):
    result = run(
        [
            ninja,
            "-C",
            build_directory,
            "-f",
            f"build-{configuration}.ninja",
            "-t",
            "commands",
            "receipt_probe",
        ],
        build_directory,
    )
    marker = "DEAC_BUILD_RECEIPT_CONFIGURATION:STRING="
    configurations = set()
    for line in result.stdout.splitlines():
        if marker in line:
            configurations.add(line.split(marker, 1)[1].split()[0])
    return configurations


def assert_graph_edge_count(graph_text, source, target, expected_count):
    needle = f"// {source} -> {target}"
    count = sum(
        line.strip().endswith(needle) for line in graph_text.splitlines()
    )
    if count != expected_count:
        raise AssertionError(
            f"expected graph edge {source} -> {target} {expected_count} "
            f"time(s), got {count}:\n{graph_text}"
        )


def test_ninja_multi_config(cmake, ninja, source, workdir, compiler_shim):
    assert_configure_rejected(
        cmake,
        source,
        workdir / "rejected-casefolded-configurations",
        compiler_shim,
        [
            "-G",
            "Ninja Multi-Config",
            f"-DCMAKE_MAKE_PROGRAM={ninja}",
            "-DCMAKE_CONFIGURATION_TYPES=Debug;DEBUG",
        ],
        "configurations must be unique ignoring ASCII case",
    )
    for attack, expected in (
        (
            "configuration-case-collision",
            "configurations must be unique ignoring ASCII case",
        ),
        (
            "configuration-list",
            "generator configuration state changed after registration",
        ),
        (
            "shadowed-configuration-cache",
            (
                "CMAKE_CONFIGURATION_TYPES cache value disagrees with "
                "effective directory state"
            ),
        ),
    ):
        assert_configure_rejected(
            cmake,
            source,
            workdir / f"rejected-late-{attack}",
            compiler_shim,
            [
                "-G",
                "Ninja Multi-Config",
                f"-DCMAKE_MAKE_PROGRAM={ninja}",
                "-DCMAKE_CONFIGURATION_TYPES=Debug;Release",
                f"-DDEAC_FIXTURE_LATE_ATTACK={attack}",
            ],
            expected,
        )

    build_directory = workdir / "multi-build"
    run(
        [
            cmake,
            "-S",
            source / "src",
            "-B",
            build_directory,
            "-G",
            "Ninja Multi-Config",
            f"-DCMAKE_MAKE_PROGRAM={ninja}",
            f"-DCMAKE_CXX_COMPILER={compiler_shim}",
            "-DCMAKE_CONFIGURATION_TYPES=Debug;Release;RelWithDebInfo",
            (
                "-DCMAKE_PREFIX_PATH="
                f"{source / 'src' / 'prefix-a'};"
                f"{build_directory / 'prefix-b'};/external"
            ),
            "-DGPU_BACKEND=none",
        ],
        source,
    )
    for configuration in ("Release", "Debug"):
        graph = graph_configurations(ninja, build_directory, configuration)
        if graph != {configuration}:
            raise AssertionError(
                f"{configuration} graph crosses receipt configurations: {graph!r}"
            )

    build(cmake, build_directory, config="Release")
    generated_root = build_directory / "generated" / "receipt"
    receipt_root = build_directory / "receipt"
    release_source = generated_root / "Release" / "fixture_receipt_probe_build_receipt.cpp"
    release_receipt = receipt_root / "Release" / "build-receipt.json"
    if not release_source.is_file() or not release_receipt.is_file():
        raise AssertionError("Release receipt outputs were not generated")
    for other in ("Debug", "RelWithDebInfo"):
        if (generated_root / other).exists() or (receipt_root / other).exists():
            raise AssertionError(f"Release build mutated {other} receipt outputs")
    release_bytes = (release_source.read_bytes(), release_receipt.read_bytes())
    release_times = (release_source.stat().st_mtime_ns, release_receipt.stat().st_mtime_ns)

    build(cmake, build_directory, config="Debug")
    debug_source = generated_root / "Debug" / "fixture_receipt_probe_build_receipt.cpp"
    debug_receipt = receipt_root / "Debug" / "build-receipt.json"
    if not debug_source.is_file() or not debug_receipt.is_file():
        raise AssertionError("Debug receipt outputs were not generated")
    if release_bytes != (release_source.read_bytes(), release_receipt.read_bytes()):
        raise AssertionError("Debug build changed Release receipt bytes")
    if release_times != (
        release_source.stat().st_mtime_ns,
        release_receipt.stat().st_mtime_ns,
    ):
        raise AssertionError("Debug build touched Release receipt outputs")
    release_document = parse_receipt(release_receipt)
    debug_document = parse_receipt(debug_receipt)
    if release_document["receipt"]["build_system"]["configuration"] != "Release":
        raise AssertionError("Release receipt records another configuration")
    if debug_document["receipt"]["build_system"]["configuration"] != "Debug":
        raise AssertionError("Debug receipt records another configuration")
    for document in (release_document, debug_document):
        entries = {
            entry["name"]: entry["value"]
            for entry in document["receipt"]["cache_entries"]
        }
        expected_configurations = "Debug;Release;RelWithDebInfo"
        if entries.get("CMAKE_CONFIGURATION_TYPES") != expected_configurations:
            raise AssertionError(
                "multi-config cache value lost list separators: "
                f"{entries.get('CMAKE_CONFIGURATION_TYPES')!r}"
            )
        expected_prefix_path = (
            "<SOURCE_ROOT>/prefix-a;<BUILD_ROOT>/prefix-b;/external"
        )
        if entries.get("CMAKE_PREFIX_PATH") != expected_prefix_path:
            raise AssertionError(
                "path-list cache value was not normalized elementwise: "
                f"{entries.get('CMAKE_PREFIX_PATH')!r}"
            )
        assert_no_git_source(document, "1.2.4")
        assert_static_dependency(document, receipt_fingerprint(document))
    assert_embedded_matches(
        build_directory / "Release" / "receipt_probe", release_receipt
    )
    assert_embedded_matches(
        build_directory / "Debug" / "receipt_probe", debug_receipt
    )


def test_parallel_receipt_consumers(
    cmake, ninja, solver_source, workdir, real_compiler
):
    source = workdir / "parallel-source"
    create_fixture_source(solver_source, source)
    build_directory = workdir / "parallel-build"
    configure(
        cmake,
        source,
        build_directory,
        real_compiler,
        "-G",
        "Ninja",
        f"-DCMAKE_MAKE_PROGRAM={ninja}",
        "-DDEAC_FIXTURE_PARALLEL_CONSUMERS=ON",
    )
    result = build(cmake, build_directory, parallel=4)
    assert_build_actions(
        result,
        [
            "fixture_receipt_probe_build_receipt.cpp.o",
            "fixture_receipt_probe_two_build_receipt.cpp.o",
            "-o receipt_probe ",
            "-o receipt_probe_two",
        ],
    )

    receipt_root = build_directory / "receipt" / "Release"
    checks = [
        (
            "receipt_probe",
            "receipt_dependency",
            receipt_root / "build-receipt.json",
        ),
        (
            "receipt_probe_two",
            "receipt_dependency_two",
            receipt_root / "build-receipt-two.json",
        ),
    ]
    for executable_name, dependency_name, receipt_path in checks:
        document = parse_receipt(receipt_path)
        if document["receipt"]["target"]["name"] != executable_name:
            raise AssertionError(f"parallel receipt names the wrong target: {document!r}")
        assert_no_git_source(document, "1.2.3")
        assert_static_dependency(
            document,
            receipt_fingerprint(document),
            expected_name=dependency_name,
        )
        assert_embedded_matches(build_directory / executable_name, receipt_path)


def test_shared_dependency_receipt_consumers(
    cmake, ninja, solver_source, workdir, real_compiler
):
    source = workdir / "shared-dependency-source"
    create_fixture_source(solver_source, source)
    build_directory = workdir / "shared-dependency-build"
    configure(
        cmake,
        source,
        build_directory,
        real_compiler,
        "-G",
        "Ninja",
        f"-DCMAKE_MAKE_PROGRAM={ninja}",
        "-DDEAC_FIXTURE_PARALLEL_CONSUMERS=ON",
        "-DDEAC_FIXTURE_SHARED_DEPENDENCY=ON",
    )
    build(cmake, build_directory, parallel=4)

    receipt_root = build_directory / "receipt" / "Release"
    receipts = (
        ("receipt_probe", receipt_root / "build-receipt.json"),
        ("receipt_probe_two", receipt_root / "build-receipt-two.json"),
    )
    fingerprints = set()
    for executable_name, receipt_path in receipts:
        document = parse_receipt(receipt_path)
        fingerprint = receipt_fingerprint(document)
        fingerprints.add(fingerprint)
        assert_static_dependency(
            document,
            fingerprint,
            expected_name="receipt_dependency",
        )
        assert_embedded_matches(build_directory / executable_name, receipt_path)
    if len(fingerprints) != 1:
        raise AssertionError(
            "shared dependency received incompatible receipt fingerprints"
        )

    rejected_build = workdir / "rejected-shared-fingerprint-conflict"
    assert_configure_rejected(
        cmake,
        source,
        rejected_build,
        real_compiler,
        [
            "-G",
            "Ninja",
            f"-DCMAKE_MAKE_PROGRAM={ninja}",
            "-DDEAC_FIXTURE_PARALLEL_CONSUMERS=ON",
            "-DDEAC_FIXTURE_SHARED_DEPENDENCY=ON",
            "-DDEAC_FIXTURE_SHARED_FINGERPRINT_ATTACK=conflict",
        ],
        "shared by incompatible toolchain fingerprints",
    )
    if list(rejected_build.glob("receipt/**/*.json")):
        raise AssertionError("incompatible shared fingerprint produced a receipt")


def test_mapped_hipblas_multi_config(
    cmake, ninja, solver_source, workdir, real_compiler
):
    source = workdir / "mapped-hipblas-source"
    create_mapped_hipblas_fixture_source(solver_source, source)
    (
        provider_root,
        include_directory,
        debug_library,
        release_library,
        package_directory,
    ) = create_mapped_hipblas_provider(workdir)
    if debug_library.resolve() == release_library.resolve():
        raise AssertionError("mapped hipBLAS configurations share one artifact")

    build_directory = workdir / "mapped-hipblas-build"
    graph_path = build_directory / "targets.dot"
    run(
        [
            cmake,
            f"--graphviz={graph_path}",
            "-S",
            source / "src",
            "-B",
            build_directory,
            "-G",
            "Ninja Multi-Config",
            f"-DCMAKE_MAKE_PROGRAM:FILEPATH={ninja}",
            f"-DCMAKE_CXX_COMPILER:FILEPATH={real_compiler}",
            "-DCMAKE_CONFIGURATION_TYPES:STRING=Debug;Release",
            (
                "-DDEAC_FIXTURE_HIPBLAS_DEBUG_LIBRARY:FILEPATH="
                f"{debug_library}"
            ),
            (
                "-DDEAC_FIXTURE_HIPBLAS_RELEASE_LIBRARY:FILEPATH="
                f"{release_library}"
            ),
            f"-DDEAC_FIXTURE_HIPBLAS_PACKAGE_DIR:PATH={package_directory}",
            f"-DHIP_RUNTIME_INCLUDE_DIR:PATH={include_directory}",
            f"-Dhipblas_ROOT:PATH={provider_root}",
        ],
        source,
    )
    if not graph_path.is_file():
        raise AssertionError("mapped hipBLAS configure did not emit Graphviz data")
    graph_text = graph_path.read_text(encoding="utf-8")
    provider_target = "deac_fixture_mapped_hipblas_provider"
    canonical_label = f'{provider_target}\\n(roc::hipblas)'
    if graph_text.count(canonical_label) != 1:
        raise AssertionError(
            "mapped hipBLAS graph does not expose exactly one canonical "
            f"provider plus alias: {graph_text}"
        )
    assert_graph_edge_count(
        graph_text, "receipt_probe", "deac_hipblas_link_contract", 1
    )
    assert_graph_edge_count(
        graph_text, "deac_hipblas_link_contract", provider_target, 1
    )
    assert_graph_edge_count(graph_text, "receipt_probe", provider_target, 0)

    receipts = {}
    configurations = (
        ("Debug", debug_library, release_library),
        ("Release", release_library, debug_library),
    )
    for configuration, provider_library, other_provider_library in configurations:
        build(cmake, build_directory, config=configuration, parallel=4)
        receipt_path = (
            build_directory
            / "receipt"
            / configuration
            / "build-receipt.json"
        )
        document = parse_receipt(receipt_path)
        assert_mapped_hipblas_receipt(
            document,
            configuration=configuration,
            provider_library=provider_library,
            other_provider_library=other_provider_library,
        )
        assert_no_git_source(document, "1.2.3")
        assert_embedded_matches(
            build_directory / configuration / "receipt_probe", receipt_path
        )
        receipts[configuration] = document
    if (
        receipts["Debug"]["receipt_sha256"]
        == receipts["Release"]["receipt_sha256"]
    ):
        raise AssertionError(
            "mapped Debug and Release artifacts produced one receipt identity"
        )


def test_selected_config_provider_collision(
    cmake, ninja, solver_source, workdir, real_compiler
):
    source = workdir / "mapped-collision-source"
    create_mapped_hipblas_fixture_source(solver_source, source)
    (
        provider_root,
        include_directory,
        debug_library,
        release_library,
        package_directory,
    ) = create_mapped_hipblas_provider(workdir)
    build_directory = workdir / "mapped-collision-build"
    result = configure(
        cmake,
        source,
        build_directory,
        real_compiler,
        "-G",
        "Ninja Multi-Config",
        f"-DCMAKE_MAKE_PROGRAM:FILEPATH={ninja}",
        "-DCMAKE_CONFIGURATION_TYPES:STRING=Debug;Release",
        f"-DDEAC_FIXTURE_HIPBLAS_DEBUG_LIBRARY:FILEPATH={debug_library}",
        f"-DDEAC_FIXTURE_HIPBLAS_RELEASE_LIBRARY:FILEPATH={release_library}",
        f"-DDEAC_FIXTURE_HIPBLAS_PACKAGE_DIR:PATH={package_directory}",
        f"-DHIP_RUNTIME_INCLUDE_DIR:PATH={include_directory}",
        f"-Dhipblas_ROOT:PATH={provider_root}",
        "-DDEAC_FIXTURE_DUPLICATE_SELECTED_ARTIFACT:BOOL=ON",
        check=False,
    )
    stage = "configure"
    if result.returncode == 0:
        result = build(
            cmake,
            build_directory,
            config="Debug",
            target="deac_fixture_generate_receipt",
            check=False,
        )
        stage = "Debug build"
    assert_rejected(
        result,
        context=f"{stage} selected-config provider collision",
        expected=(
            "selected configuration contains duplicate required "
            "link-library artifact",
            "required link-library artifacts must be unique",
        ),
    )
    receipt_path = build_directory / "receipt" / "Debug" / "build-receipt.json"
    if receipt_path.exists():
        raise AssertionError("provider collision produced a Debug receipt")


def test_nested_build_root_canonicalization(
    cmake, ninja, solver_source, workdir, real_compiler
):
    source = workdir / "nested-build-source"
    create_fixture_source(solver_source, source)
    # Cover a two-byte ASCII component and a two-byte UTF-8 code point.
    source_prefixes = [
        source / "src" / component / "prefix"
        for component in ("ab", "é")
    ]
    for prefix in source_prefixes:
        prefix.mkdir(parents=True)
    receipts = []
    for suffix in ("a", "b"):
        # Put the build inside CMAKE_SOURCE_DIR so both canonical roots match;
        # BUILD_ROOT must win as the more-specific containing root.
        build_directory = source / "src" / f"nested-build-{suffix}"
        build_prefixes = [
            build_directory / component / "prefix"
            for component in ("xy", "ü")
        ]
        for prefix in build_prefixes:
            prefix.mkdir(parents=True)
        configure(
            cmake,
            source,
            build_directory,
            real_compiler,
            "-G",
            "Ninja",
            f"-DCMAKE_MAKE_PROGRAM={ninja}",
            "-DCMAKE_PREFIX_PATH="
            + ";".join(str(path) for path in source_prefixes + build_prefixes),
        )
        build(cmake, build_directory)
        receipt_path = (
            build_directory / "receipt" / "Release" / "build-receipt.json"
        )
        document = parse_receipt(receipt_path)
        generated_sources = [
            path
            for group in document["receipt"]["compile_groups"]
            for path in group["sources"]
            if path.endswith("fixture_receipt_probe_build_receipt.cpp")
        ]
        if len(generated_sources) != 1 or not generated_sources[0].startswith(
            "<BUILD_ROOT>/generated/receipt/Release/"
        ):
            raise AssertionError(
                "nested generated source did not prefer the most-specific "
                f"build root: {generated_sources!r}"
            )
        cache_entries = {
            entry["name"]: entry["value"]
            for entry in document["receipt"]["cache_entries"]
        }
        expected_prefix_path = (
            "<SOURCE_ROOT>/ab/prefix;<SOURCE_ROOT>/é/prefix;"
            "<BUILD_ROOT>/xy/prefix;<BUILD_ROOT>/ü/prefix"
        )
        if cache_entries.get("CMAKE_PREFIX_PATH") != expected_prefix_path:
            raise AssertionError(
                "two-byte path components were not tokenized against the "
                "most-specific canonical root: "
                f"{cache_entries.get('CMAKE_PREFIX_PATH')!r}"
            )
        receipts.append(receipt_path.read_bytes())
    if receipts[0] != receipts[1]:
        raise AssertionError(
            "equivalent nested build roots produced different canonical receipts"
        )


def test_hidden_receipt_config_rejection(
    cmake, ninja, source, workdir, real_compiler
):
    assert_configure_rejected(
        cmake,
        source,
        workdir / "rejected-hidden-receipt-config",
        real_compiler,
        [
            "-G",
            "Ninja Multi-Config",
            f"-DCMAKE_MAKE_PROGRAM={ninja}",
            "-DCMAKE_CONFIGURATION_TYPES=Debug;Release",
            "-DDEAC_FIXTURE_RECEIPT_ATTACK=hidden-config",
        ],
        "RECEIPT",
    )


def test_output_path_symlink_rejections(
    cmake, source, workdir, real_compiler
):
    target_root = workdir / "output-path-symlink-targets"
    target_root.mkdir()
    attacks = (
        ("generated-existing", Path("generated/receipt"), True),
        ("generated-dangling", Path("generated/receipt"), False),
        ("receipt-existing", Path("receipt"), True),
        ("receipt-dangling", Path("receipt"), False),
    )
    for name, relative_link, target_exists in attacks:
        build_directory = workdir / f"rejected-output-symlink-{name}"
        build_directory.mkdir()
        link = build_directory / relative_link
        link.parent.mkdir(parents=True, exist_ok=True)
        target = target_root / name
        if target_exists:
            target.mkdir()
        link.symlink_to(target, target_is_directory=True)
        if not link.is_symlink():
            raise AssertionError(f"{name} fixture did not create a symlink")

        result = configure(
            cmake,
            source,
            build_directory,
            real_compiler,
            check=False,
        )
        assert_rejected(result, context=name, expected="symlink")
        receipt_path = (
            build_directory / "receipt" / "Release" / "build-receipt.json"
        )
        if receipt_path.exists():
            raise AssertionError(f"{name} symlink produced a receipt")


def test_configless_mapped_hipblas(
    cmake, ninja, source, workdir, real_compiler
):
    (
        provider_root,
        include_directory,
        configless_library,
        alternate_library,
        package_directory,
    ) = create_configless_mapped_hipblas_provider(workdir)
    if configless_library.resolve() == alternate_library.resolve():
        raise AssertionError("configless and alternate providers are identical")

    build_directory = workdir / "configless-mapped-hipblas-build"
    configure(
        cmake,
        source,
        build_directory,
        real_compiler,
        "-G",
        "Ninja",
        f"-DCMAKE_MAKE_PROGRAM:FILEPATH={ninja}",
        "-DDEAC_FIXTURE_HIPBLAS_MODE=imported_target",
        f"-DDEAC_FIXTURE_HIPBLAS_INCLUDE_DIR:PATH={include_directory}",
        f"-DDEAC_FIXTURE_HIPBLAS_LIBRARY:FILEPATH={configless_library}",
        f"-DDEAC_FIXTURE_HIPBLAS_PACKAGE_DIR:PATH={package_directory}",
        (
            "-DDEAC_FIXTURE_EXPECTED_HIPBLAS_ARTIFACT:FILEPATH="
            f"{configless_library}"
        ),
        f"-DHIP_RUNTIME_INCLUDE_DIR:PATH={include_directory}",
        f"-Dhipblas_ROOT:PATH={provider_root}",
    )
    build(cmake, build_directory, parallel=4)
    receipt_path = (
        build_directory / "receipt" / "Release" / "build-receipt.json"
    )
    document = parse_receipt(receipt_path)
    assert_static_dependency(document, receipt_fingerprint(document))
    assert_hipblas_contract(
        document,
        static_dependency="receipt_dependency",
        mode="imported_target",
        provider_library=configless_library,
        include_directory=include_directory,
        package_directory=package_directory,
        provider_root=provider_root,
    )

    library_arguments = []
    for fragment in document["receipt"]["link"]["command_fragments"]:
        if fragment["role"] == "libraries":
            library_arguments.extend(shlex.split(fragment["fragment"]))
    expected_library = str(configless_library.resolve())
    unexpected_library = str(alternate_library.resolve())
    if library_arguments.count(expected_library) != 1:
        raise AssertionError(
            "File API did not select the configless mapped artifact exactly "
            f"once: {library_arguments!r}"
        )
    if unexpected_library in library_arguments:
        raise AssertionError(
            "File API selected the alternate named mapped artifact: "
            f"{library_arguments!r}"
        )
    assert_no_git_source(document, "1.2.4")
    assert_embedded_matches(build_directory / "receipt_probe", receipt_path)


def test_hipblas_receipt_contract(
    cmake, ninja, source, workdir, real_compiler
):
    (
        provider_root,
        include_directory,
        provider_library,
        package_directory,
    ) = create_hipblas_provider(workdir)
    modes = (
        "imported_target",
        "compatibility_library",
        "blas_off",
        "non_hip",
    )
    for generator in ("Ninja", "Unix Makefiles"):
        generator_slug = generator.lower().replace(" ", "-")
        for mode in modes:
            build_directory = workdir / f"hipblas-{generator_slug}-{mode}"
            generator_arguments = ["-G", generator]
            if generator == "Ninja":
                generator_arguments.append(f"-DCMAKE_MAKE_PROGRAM={ninja}")
            configure(
                cmake,
                source,
                build_directory,
                real_compiler,
                *generator_arguments,
                "-DDEAC_FIXTURE_PARALLEL_CONSUMERS=ON",
                f"-DDEAC_FIXTURE_HIPBLAS_MODE={mode}",
                f"-DDEAC_FIXTURE_HIPBLAS_INCLUDE_DIR={include_directory}",
                f"-DDEAC_FIXTURE_HIPBLAS_LIBRARY={provider_library}",
                f"-DDEAC_FIXTURE_HIPBLAS_PACKAGE_DIR={package_directory}",
                f"-DHIP_RUNTIME_INCLUDE_DIR={include_directory}",
                f"-Dhipblas_ROOT={provider_root}",
            )
            build(cmake, build_directory, parallel=4)
            receipt_root = build_directory / "receipt" / "Release"
            checks = (
                (
                    "receipt_probe",
                    "receipt_dependency",
                    receipt_root / "build-receipt.json",
                ),
                (
                    "receipt_probe_two",
                    "receipt_dependency_two",
                    receipt_root / "build-receipt-two.json",
                ),
            )
            for executable_name, dependency_name, receipt_path in checks:
                document = parse_receipt(receipt_path)
                assert_static_dependency(
                    document,
                    receipt_fingerprint(document),
                    expected_name=dependency_name,
                )
                assert_hipblas_contract(
                    document,
                    static_dependency=dependency_name,
                    mode=mode,
                    provider_library=provider_library,
                    include_directory=include_directory,
                    package_directory=package_directory,
                    provider_root=provider_root,
                )
                assert_embedded_matches(
                    build_directory / executable_name, receipt_path
                )


def diagnostic_text(result):
    return result.stdout + result.stderr


def escaped_diagnostic(text):
    return "".join(
        character
        if character in "\n\r\t" or ord(character) >= 0x20
        else f"\\x{ord(character):02x}"
        for character in text
    )


def assert_rejected(result, *, context, expected):
    if result.returncode == 0:
        raise AssertionError(f"unsupported {context} was accepted")
    output = diagnostic_text(result)
    expected_fragments = (expected,) if isinstance(expected, str) else expected
    normalized_output = " ".join(output.split())
    if not any(
        fragment in output
        or " ".join(fragment.split()) in normalized_output
        for fragment in expected_fragments
    ):
        raise AssertionError(
            f"{context} rejection did not mention {expected_fragments!r}:\n"
            f"{escaped_diagnostic(output)}"
        )
    return output


def assert_configure_rejected(cmake, source, build_directory, compiler, arguments, text):
    result = configure(
        cmake,
        source,
        build_directory,
        compiler,
        *arguments,
        check=False,
    )
    assert_rejected(
        result,
        context=f"configure route {arguments!r}",
        expected=text,
    )


def assert_configure_or_build_rejected(
    cmake,
    source,
    build_directory,
    compiler,
    arguments,
    expected,
    *,
    config=None,
    target=None,
    environment=None,
):
    result = configure(
        cmake,
        source,
        build_directory,
        compiler,
        *arguments,
        check=False,
    )
    stage = "configure"
    if result.returncode == 0:
        result = build(
            cmake,
            build_directory,
            config=config,
            target=target,
            check=False,
            environment=environment,
        )
        stage = "build"
    output = assert_rejected(
        result,
        context=f"{stage} route {arguments!r}",
        expected=expected,
    )
    return stage, output


def test_hipblas_receipt_rejections(
    cmake, source, workdir, real_compiler
):
    (
        provider_root,
        include_directory,
        provider_library,
        package_directory,
    ) = create_hipblas_provider(workdir)
    common_arguments = [
        "-DDEAC_FIXTURE_HIPBLAS_MODE=imported_target",
        f"-DDEAC_FIXTURE_HIPBLAS_INCLUDE_DIR={include_directory}",
        f"-DDEAC_FIXTURE_HIPBLAS_LIBRARY={provider_library}",
        f"-DDEAC_FIXTURE_HIPBLAS_PACKAGE_DIR={package_directory}",
        f"-DHIP_RUNTIME_INCLUDE_DIR={include_directory}",
        f"-Dhipblas_ROOT={provider_root}",
    ]
    attacks = (
        (
            "duplicate-contract-provider",
            "-DDEAC_FIXTURE_HIPBLAS_CONTRACT_ATTACK=duplicate-provider",
            "provider once",
        ),
        (
            "unequal-link-lists",
            "-DDEAC_FIXTURE_LINK_LIBRARY_ATTACK=unequal-lists",
            "equal",
        ),
        (
            "duplicate-link-name",
            "-DDEAC_FIXTURE_LINK_LIBRARY_ATTACK=duplicate-name",
            "must be unique:",
        ),
        (
            "duplicate-link-artifact",
            "-DDEAC_FIXTURE_LINK_LIBRARY_ATTACK=duplicate-artifact",
            "required link-library artifacts must be unique",
        ),
        (
            "missing-link-artifact",
            "-DDEAC_FIXTURE_LINK_LIBRARY_ATTACK=missing-artifact",
            "regular file:",
        ),
    )
    for name, attack_argument, expected in attacks:
        assert_configure_rejected(
            cmake,
            source,
            workdir / f"rejected-hipblas-{name}",
            real_compiler,
            [*common_arguments, attack_argument],
            expected,
        )


def test_control_byte_link_artifact_rejections(
    cmake, source, workdir, real_compiler
):
    artifact_root = workdir / "control-byte-link-artifacts"
    artifact_root.mkdir()
    controls = (
        ("tab", "\t"),
        ("escape", chr(0x1B)),
        ("delete", chr(0x7F)),
    )
    for name, control in controls:
        artifact = artifact_root / f"libbefore{control}after.a"
        artifact.write_bytes(b"fixture link artifact\n")
        if not artifact.is_file():
            raise AssertionError(f"{name} control artifact was not created")

        build_directory = workdir / f"rejected-control-artifact-{name}"
        result = configure(
            cmake,
            source,
            build_directory,
            real_compiler,
            f"-DDEAC_FIXTURE_CONTROL_LINK_ARTIFACT:STRING={artifact}",
            check=False,
        )
        output = assert_rejected(
            result,
            context=f"{name} control artifact",
            expected="control byte",
        )
        if control in output:
            raise AssertionError(
                f"{name} artifact control was echoed raw:\n"
                f"{escaped_diagnostic(output)}"
            )
        receipt_path = (
            build_directory / "receipt" / "Release" / "build-receipt.json"
        )
        if receipt_path.exists():
            raise AssertionError(f"{name} control artifact produced a receipt")


def test_effective_shell_input_rejections(
    cmake, source, workdir, real_compiler
):
    dollar_build = workdir / "rejected-effective-flags-dollar-variable"
    configure(
        cmake,
        source,
        dollar_build,
        real_compiler,
        "-DDEAC_FIXTURE_EFFECTIVE_CXX_FLAGS_ATTACK=dollar-variable",
    )
    for value in ("-DDEAC_ENV_PROBE=one", "-DDEAC_ENV_PROBE=two"):
        environment = os.environ.copy()
        environment["DEAC_UNATTESTED_FLAGS"] = value
        result = build(
            cmake,
            dollar_build,
            target="deac_fixture_generate_receipt",
            check=False,
            environment=environment,
        )
        assert_rejected(
            result,
            context=f"effective CMAKE_CXX_FLAGS under {value}",
            expected="contains unsafe POSIX shell syntax",
        )
        receipt_path = (
            dollar_build / "receipt" / "Release" / "build-receipt.json"
        )
        if receipt_path.exists():
            raise AssertionError(
                "changed environment reused an unattested configured identity"
            )

    attacks = (
        "braced-variable",
        "glob",
        "redirection",
        "comment",
        "command-substitution",
        "backtick-substitution",
        "process-substitution",
        "quoted-dollar",
        "escaped-dollar",
    )
    for attack in attacks:
        build_directory = workdir / f"rejected-effective-flags-{attack}"
        assert_configure_or_build_rejected(
            cmake,
            source,
            build_directory,
            real_compiler,
            [f"-DDEAC_FIXTURE_EFFECTIVE_CXX_FLAGS_ATTACK={attack}"],
            "contains unsafe POSIX shell syntax",
            target="deac_fixture_generate_receipt",
        )
        receipt_path = (
            build_directory / "receipt" / "Release" / "build-receipt.json"
        )
        if receipt_path.exists():
            raise AssertionError(f"effective {attack} flags produced a receipt")

    # Globs remain data when their protection survives both CMake's generator
    # and the POSIX shell.  Dollar signs have separate Make/Ninja expansion
    # semantics and are rejected even when shell quoting alone would protect
    # them.
    for accepted in ("safe-quoted", "safe-escaped"):
        accepted_build = workdir / f"accepted-effective-flags-{accepted}"
        configure(
            cmake,
            source,
            accepted_build,
            real_compiler,
            f"-DDEAC_FIXTURE_EFFECTIVE_CXX_FLAGS_ATTACK={accepted}",
        )
        build(cmake, accepted_build)


def test_deferred_seal_rejections(cmake, source, workdir, real_compiler):
    attacks = (
        ("launcher", "CXX_COMPILER_LAUNCHER"),
        ("rule", "CXX rule templates changed after registration"),
        ("ipo", "INTERPROCEDURAL_OPTIMIZATION_RELEASE"),
        ("link-what-you-use", "LINK_WHAT_YOU_USE"),
        ("interface-literal", "interface dependencies"),
        (
            "interface-conditional",
            "has an unsupported generator-expression link item",
        ),
        ("archive-tool", "CMAKE_AR"),
        ("archive-index-tool", "CMAKE_RANLIB"),
        (
            "build-type",
            "generator configuration state changed after registration",
        ),
        (
            "shadowed-build-type-cache",
            (
                "CMAKE_BUILD_TYPE cache value disagrees with effective "
                "directory state"
            ),
        ),
    )
    for attack, expected in attacks:
        assert_configure_rejected(
            cmake,
            source,
            workdir / f"rejected-deferred-{attack}",
            real_compiler,
            [f"-DDEAC_FIXTURE_LATE_ATTACK={attack}"],
            expected,
        )


def test_parent_deferred_seal_rejection(
    cmake, ninja, solver_source, workdir, real_compiler
):
    source = workdir / "parent-mutation-source"
    create_parent_mutation_fixture_source(solver_source, source)
    attacks = (
        ("launcher", [], "CXX_COMPILER_LAUNCHER"),
        (
            "build-type",
            ["-DDEAC_FIXTURE_PARENT_MUTATION=build-type"],
            "generator configuration state changed after registration",
        ),
        (
            "configuration-list",
            [
                "-G",
                "Ninja Multi-Config",
                f"-DCMAKE_MAKE_PROGRAM={ninja}",
                "-DCMAKE_CONFIGURATION_TYPES=Debug;Release",
                "-DDEAC_FIXTURE_PARENT_MUTATION=configuration-list",
            ],
            "generator configuration state changed after registration",
        ),
    )
    for name, arguments, expected in attacks:
        assert_configure_rejected(
            cmake,
            source,
            workdir / f"rejected-parent-mutation-{name}",
            real_compiler,
            arguments,
            expected,
        )


def test_subdirectory_dependency_directory_attribution(
    cmake, source, workdir, real_compiler
):
    normal_build = workdir / "subdirectory-dependency-normal"
    configure(
        cmake,
        source,
        normal_build,
        real_compiler,
        "-G",
        "Unix Makefiles",
        "-DDEAC_FIXTURE_SUBDIRECTORY_DEPENDENCY_MODE=normal",
    )
    # The exported primary refresh output must be directly buildable even when
    # a recorded dependency was declared in another source directory.  This
    # specifically guards the Unix Makefiles portability regression that
    # occurred when the convenience target depended on a generated-source
    # BYPRODUCT owned by the real consumer's build.make.
    build(
        cmake,
        normal_build,
        target="deac_fixture_generate_receipt",
    )
    receipt_path = (
        normal_build / "receipt" / "Release" / "build-receipt.json"
    )
    refresh_document = parse_receipt(receipt_path)
    assert_static_dependency(
        refresh_document, receipt_fingerprint(refresh_document)
    )
    if (normal_build / "receipt_probe").exists():
        raise AssertionError("receipt-only target built the consumer executable")
    refresh_digest = refresh_document["receipt_sha256"]

    # Building the real consumer must embed the same attributed receipt.
    build(cmake, normal_build)
    document = parse_receipt(receipt_path)
    assert_static_dependency(document, receipt_fingerprint(document))
    if document["receipt_sha256"] != refresh_digest:
        raise AssertionError("consumer build changed the refresh-only receipt")
    assert_embedded_matches(normal_build / "receipt_probe", receipt_path)

    attacks = (
        ("launcher", "RULE_LAUNCH_COMPILE"),
        ("rule", "differ from the receipt registration directory"),
        ("compiler", "CMAKE_CXX_COMPILER"),
    )
    for attack, expected in attacks:
        assert_configure_rejected(
            cmake,
            source,
            workdir / f"rejected-subdirectory-dependency-{attack}",
            real_compiler,
            [f"-DDEAC_FIXTURE_SUBDIRECTORY_DEPENDENCY_MODE={attack}"],
            expected,
        )


def test_dependency_fingerprint_rejections(
    cmake, source, workdir, real_compiler
):
    # First exercise the integration route: registration injects the reserved
    # macro, then a late target mutation removes it before File API generation.
    removed_build = workdir / "rejected-dependency-fingerprint-remove"
    assert_configure_or_build_rejected(
        cmake,
        source,
        removed_build,
        real_compiler,
        ["-DDEAC_FIXTURE_DEPENDENCY_FINGERPRINT_ATTACK=remove"],
        "toolchain fingerprint",
        target="deac_fixture_generate_receipt",
    )
    removed_receipt = (
        removed_build / "receipt" / "Release" / "build-receipt.json"
    )
    if removed_receipt.exists():
        raise AssertionError("late fingerprint removal produced a receipt")

    # CMake may de-duplicate identical definitions or discard function-style
    # definitions.  Mutate a fresh reply so every exact-name adversarial shape
    # reaches the receipt generator independent of generator normalization.
    reply_build = workdir / "rejected-dependency-fingerprint-replies"
    configure(
        cmake,
        source,
        reply_build,
        real_compiler,
    )
    target_reply = file_api_target_reply(reply_build, "receipt_dependency")
    original = json.loads(target_reply.read_text(encoding="utf-8"))
    fingerprint_name = "DEAC_BUILD_TOOLCHAIN_FINGERPRINT_SHA256"
    definitions = original["compileGroups"][0]["defines"]
    fingerprints = [
        definition
        for definition in definitions
        if definition["define"].startswith(f"{fingerprint_name}=")
    ]
    if len(fingerprints) != 1:
        raise AssertionError(
            f"fixture dependency has unexpected fingerprints: {definitions!r}"
        )

    mutations = (
        ("duplicate", fingerprints[0]["define"]),
        ("conflict", f"{fingerprint_name}=" + "0" * 64),
        ("function-conflict", f"{fingerprint_name}(x)=bad"),
        ("spaced-conflict", f"{fingerprint_name} =bad"),
    )
    reply_receipt = (
        reply_build / "receipt" / "Release" / "build-receipt.json"
    )
    expected = (
        "exactly one configured toolchain fingerprint without a duplicate "
        "or conflicting macro definition"
    )
    for name, extra_definition in mutations:
        document = json.loads(json.dumps(original))
        document["compileGroups"][0]["defines"].append(
            {"define": extra_definition}
        )
        write(
            target_reply,
            json.dumps(document, ensure_ascii=False, separators=(",", ":"))
            + "\n",
        )
        result = build(
            cmake,
            reply_build,
            target="deac_fixture_generate_receipt",
            check=False,
        )
        assert_rejected(result, context=f"{name} fingerprint", expected=expected)
        if reply_receipt.exists():
            raise AssertionError(f"{name} fingerprint produced a receipt")

    # Definitions are also present in the effective command fragments.  Edit
    # the reply directly so compiler- and generator-specific normalization
    # cannot hide attached, split, or slash-D spellings from this check.
    original_fragments = original["compileGroups"][0].get(
        "compileCommandFragments"
    )
    if not isinstance(original_fragments, list):
        raise TypeError("fixture dependency compile fragments are malformed")
    conflicting_definition = f"{fingerprint_name}=" + "0" * 64
    fragment_mutations = (
        ("fragment-d-duplicate", f" -D{fingerprints[0]['define']}"),
        ("fragment-d-conflict", f" -D{conflicting_definition}"),
        ("fragment-split-d-duplicate", f" -D {fingerprints[0]['define']}"),
        ("fragment-split-d-conflict", f" -D {conflicting_definition}"),
        ("fragment-slash-d-duplicate", f" /D{fingerprints[0]['define']}"),
        ("fragment-split-slash-d-conflict", f" /D {conflicting_definition}"),
    )
    for name, extra_fragment in fragment_mutations:
        document = json.loads(json.dumps(original))
        document["compileGroups"][0]["compileCommandFragments"].append(
            {"fragment": extra_fragment}
        )
        write(
            target_reply,
            json.dumps(document, ensure_ascii=False, separators=(",", ":"))
            + "\n",
        )
        result = build(
            cmake,
            reply_build,
            target="deac_fixture_generate_receipt",
            check=False,
        )
        assert_rejected(
            result,
            context=name,
            expected="reserved fingerprint fragment conflict",
        )
        if reply_receipt.exists():
            raise AssertionError(f"{name} fingerprint produced a receipt")

    # Identifier-prefix matching must not turn a distinct longer macro into a
    # false conflict with the reserved fingerprint name in either source of
    # effective compile definitions.
    longer_document = json.loads(json.dumps(original))
    longer_definition = f"{fingerprint_name}_EXTRA=allowed"
    longer_document["compileGroups"][0]["defines"].append(
        {"define": longer_definition}
    )
    longer_fragments = (
        f" -D{longer_definition}",
        f" -D {longer_definition}",
        f" /D{longer_definition}",
        f" /D {longer_definition}",
    )
    longer_document["compileGroups"][0]["compileCommandFragments"].extend(
        {"fragment": fragment} for fragment in longer_fragments
    )
    write(
        target_reply,
        json.dumps(longer_document, ensure_ascii=False, separators=(",", ":"))
        + "\n",
    )
    build(
        cmake,
        reply_build,
        target="deac_fixture_generate_receipt",
    )
    document = parse_receipt(reply_receipt)
    assert_static_dependency(document, receipt_fingerprint(document))
    dependency = next(
        entry
        for entry in document["receipt"]["target_dependencies"]
        if entry["name"] == "receipt_dependency"
    )
    recorded_definitions = [
        definition
        for group in dependency["compile_groups"]
        for definition in group["definitions"]
    ]
    if longer_definition not in recorded_definitions:
        raise AssertionError("longer fingerprint-like definition was discarded")
    recorded_fragments = [
        fragment["fragment"]
        for group in dependency["compile_groups"]
        for fragment in group["command_fragments"]
    ]
    missing_fragments = [
        fragment
        for fragment in longer_fragments
        if fragment not in recorded_fragments
    ]
    if missing_fragments:
        raise AssertionError(
            "longer fingerprint-like command fragments were discarded: "
            f"{missing_fragments!r}"
        )


def test_json_control_encoding(cmake, source, workdir, real_compiler):
    controls = (
        ("escape", chr(0x1B), b"\\u001b"),
        ("vertical-tab", chr(0x0B), b"\\u000b"),
    )
    for name, control, encoded in controls:
        build_directory = workdir / f"json-control-{name}"
        configure_result = configure(
            cmake,
            source,
            build_directory,
            real_compiler,
            f"-DDEAC_FIXTURE_CONTROL_CACHE:STRING=before{control}after",
        )
        build_result = build(cmake, build_directory)
        output = diagnostic_text(configure_result) + diagnostic_text(build_result)
        if control in output:
            raise AssertionError(
                f"{name} control was copied into a build-receipt diagnostic"
            )
        receipt_path = (
            build_directory / "receipt" / "Release" / "build-receipt.json"
        )
        raw_receipt = receipt_path.read_bytes()
        if control.encode() in raw_receipt:
            raise AssertionError(
                f"{name} control was copied raw into build-receipt JSON"
            )
        if encoded not in raw_receipt:
            raise AssertionError(
                f"{name} control did not use canonical lowercase JSON escaping"
            )
        document = parse_receipt(receipt_path)
        if cache_value(document, "DEAC_FIXTURE_CONTROL_CACHE") != (
            f"before{control}after"
        ):
            raise AssertionError(f"{name} cache control did not round-trip")
        assert_embedded_matches(build_directory / "receipt_probe", receipt_path)


def file_api_target_reply(build_directory, target_name):
    reply_directory = build_directory / ".cmake" / "api" / "v1" / "reply"
    indices = sorted(reply_directory.glob("index-*.json"))
    if not indices:
        raise AssertionError("configured fixture has no File API index")
    index = json.loads(indices[-1].read_text(encoding="utf-8"))
    codemodel_reference = index["reply"]["codemodel-v2"]
    codemodel = json.loads(
        (reply_directory / codemodel_reference["jsonFile"]).read_text(
            encoding="utf-8"
        )
    )
    target_references = [
        target
        for configuration in codemodel["configurations"]
        for target in configuration["targets"]
        if target["name"] == target_name
    ]
    if len(target_references) != 1:
        raise AssertionError(
            f"expected one {target_name} File API target, got "
            f"{target_references!r}"
        )
    return reply_directory / target_references[0]["jsonFile"]


def file_api_toolchains_reply(build_directory):
    reply_directory = build_directory / ".cmake" / "api" / "v1" / "reply"
    indices = sorted(reply_directory.glob("index-*.json"))
    if not indices:
        raise AssertionError("configured fixture has no File API index")
    index = json.loads(indices[-1].read_text(encoding="utf-8"))
    reference = index["reply"].get("toolchains-v1")
    if not isinstance(reference, dict) or not isinstance(
        reference.get("jsonFile"), str
    ):
        raise AssertionError(
            f"fixture toolchains reference is malformed: {reference!r}"
        )
    return reply_directory / reference["jsonFile"]


def cxx_toolchain(document):
    matches = [
        toolchain
        for toolchain in document.get("toolchains", [])
        if toolchain.get("language") == "CXX"
    ]
    if len(matches) != 1 or not isinstance(matches[0].get("compiler"), dict):
        raise AssertionError(f"fixture CXX toolchain is malformed: {matches!r}")
    return matches[0]


def test_file_api_shape_rejections(cmake, source, workdir, real_compiler):
    build_directory = workdir / "rejected-file-api-shapes"
    configure(cmake, source, build_directory, real_compiler)
    target_reply = file_api_target_reply(build_directory, "receipt_probe")
    original = json.loads(target_reply.read_text(encoding="utf-8"))

    malformed_documents = []
    missing_compile_groups = json.loads(json.dumps(original))
    missing_compile_groups.pop("compileGroups", None)
    malformed_documents.append(
        ("missing compileGroups", missing_compile_groups, "has no compile groups")
    )
    wrong_compile_groups = json.loads(json.dumps(original))
    wrong_compile_groups["compileGroups"] = {}
    malformed_documents.append(
        (
            "malformed compileGroups",
            wrong_compile_groups,
            "compile groups must be an array",
        )
    )
    malformed_optional_parent = json.loads(json.dumps(original))
    malformed_optional_parent.setdefault("link", {})["sysroot"] = "not-an-object"
    malformed_documents.append(
        (
            "malformed optional link.sysroot",
            malformed_optional_parent,
            "link sysroot is malformed",
        )
    )

    receipt_path = (
        build_directory / "receipt" / "Release" / "build-receipt.json"
    )
    for name, document, expected in malformed_documents:
        write(
            target_reply,
            json.dumps(document, ensure_ascii=False, separators=(",", ":"))
            + "\n",
        )
        result = build(
            cmake,
            build_directory,
            target="deac_fixture_generate_receipt",
            check=False,
        )
        assert_rejected(
            result,
            context=name,
            expected=expected,
        )
        if receipt_path.exists():
            raise AssertionError(f"{name} produced a build receipt")

    # CMake's JSON decoder can materialize a semantic NUL from a valid escape.
    # Exercise the raw reply scanner directly, with adjacent UTF-8 text proving
    # that this is byte-parity validation rather than an ASCII-only shortcut.
    semantic_nul_document = json.loads(json.dumps(original))
    semantic_nul_document["deacRawJsonProbe"] = "pré\x00后"
    semantic_nul_json = (
        json.dumps(
            semantic_nul_document,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
    )
    if "\\u0000" not in semantic_nul_json or "\\\\u0000" in semantic_nul_json:
        raise AssertionError("semantic NUL fixture does not use one JSON escape")
    write(target_reply, semantic_nul_json)
    result = build(
        cmake,
        build_directory,
        target="deac_fixture_generate_receipt",
        check=False,
    )
    assert_rejected(result, context="semantic JSON NUL", expected="NUL")
    if receipt_path.exists():
        raise AssertionError("semantic JSON NUL produced a build receipt")

    # Two source backslashes encode string data containing the literal
    # characters backslash-u0000.  That is not a semantic NUL and must remain
    # accepted by the raw scanner.
    literal_escape_document = json.loads(json.dumps(original))
    literal_escape_document["deacRawJsonProbe"] = "pré\\u0000后"
    literal_escape_json = (
        json.dumps(
            literal_escape_document,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
    )
    if "\\\\u0000" not in literal_escape_json:
        raise AssertionError(
            "literal backslash-u0000 fixture does not use an even escape"
        )
    write(target_reply, literal_escape_json)
    build(
        cmake,
        build_directory,
        target="deac_fixture_generate_receipt",
    )
    document = parse_receipt(receipt_path)
    assert_static_dependency(document, receipt_fingerprint(document))

    write(
        target_reply,
        json.dumps(original, ensure_ascii=False, separators=(",", ":")) + "\n",
    )


def test_compiler_target_reply_rejections(
    cmake, source, workdir, real_compiler
):
    target_result = run([real_compiler, "-dumpmachine"], source)
    configured_target = target_result.stdout.strip()
    if (
        not configured_target
        or len(target_result.stdout.splitlines()) != 1
        or any(character.isspace() for character in configured_target)
    ):
        raise AssertionError(
            f"compiler returned an unsafe target triple: {target_result.stdout!r}"
        )

    present_build = workdir / "compiler-target-present"
    configure(
        cmake,
        source,
        present_build,
        real_compiler,
        f"-DCMAKE_CXX_COMPILER_TARGET:STRING={configured_target}",
    )
    present_reply = file_api_toolchains_reply(present_build)
    present_original = json.loads(present_reply.read_text(encoding="utf-8"))
    present_compiler = cxx_toolchain(present_original)["compiler"]
    if present_compiler.get("target") != configured_target:
        raise AssertionError("File API omitted the configured compiler target")
    present_receipt = (
        present_build / "receipt" / "Release" / "build-receipt.json"
    )
    for name, replacement in (("missing", None), ("mismatch", "-mismatch")):
        document = json.loads(json.dumps(present_original))
        compiler = cxx_toolchain(document)["compiler"]
        if replacement is None:
            compiler.pop("target")
        else:
            compiler["target"] = configured_target + replacement
        write(
            present_reply,
            json.dumps(document, ensure_ascii=False, separators=(",", ":"))
            + "\n",
        )
        result = build(
            cmake,
            present_build,
            target="deac_fixture_generate_receipt",
            check=False,
        )
        assert_rejected(
            result,
            context=f"{name} configured compiler target",
            expected=(
                "CXX compiler target disagrees with configure-time identity"
            ),
        )
        if present_receipt.exists():
            raise AssertionError(f"{name} compiler target produced a receipt")

    write(
        present_reply,
        json.dumps(present_original, ensure_ascii=False, separators=(",", ":"))
        + "\n",
    )
    build(cmake, present_build, target="deac_fixture_generate_receipt")
    present_document = parse_receipt(present_receipt)
    recorded_target = next(
        toolchain["compiler"]["target"]
        for toolchain in present_document["receipt"]["toolchains"]
        if toolchain["language"] == "CXX"
    )
    if recorded_target != configured_target:
        raise AssertionError("receipt changed the configured compiler target")

    absent_build = workdir / "compiler-target-absent"
    configure(cmake, source, absent_build, real_compiler)
    absent_reply = file_api_toolchains_reply(absent_build)
    absent_document = json.loads(absent_reply.read_text(encoding="utf-8"))
    absent_compiler = cxx_toolchain(absent_document)["compiler"]
    if "target" in absent_compiler:
        raise AssertionError("fixture compiler unexpectedly has a target field")
    absent_compiler["target"] = configured_target
    write(
        absent_reply,
        json.dumps(absent_document, ensure_ascii=False, separators=(",", ":"))
        + "\n",
    )
    result = build(
        cmake,
        absent_build,
        target="deac_fixture_generate_receipt",
        check=False,
    )
    assert_rejected(
        result,
        context="unexpected compiler target",
        expected="CXX compiler target disagrees with configure-time identity",
    )
    absent_receipt = (
        absent_build / "receipt" / "Release" / "build-receipt.json"
    )
    if absent_receipt.exists():
        raise AssertionError("unexpected compiler target produced a receipt")


def test_rule_override_rejections(
    cmake, ninja, source, workdir, real_compiler
):
    compile_rule = (
        "/usr/bin/env <CMAKE_CXX_COMPILER> <DEFINES> <INCLUDES> <FLAGS> "
        "-o <OBJECT> -c <SOURCE>"
    )
    link_rule = (
        "/usr/bin/env <CMAKE_CXX_COMPILER> <FLAGS> <CMAKE_CXX_LINK_FLAGS> "
        "<LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>"
    )
    attacks = [
        ("compile", [f"-DCMAKE_CXX_COMPILE_OBJECT={compile_rule}"], "CMAKE_CXX_COMPILE_OBJECT"),
        ("link", [f"-DCMAKE_CXX_LINK_EXECUTABLE={link_rule}"], "CMAKE_CXX_LINK_EXECUTABLE"),
        (
            "cuda",
            ["-DCMAKE_CUDA_COMPILE_OBJECT=/usr/bin/env <CMAKE_CUDA_COMPILER>"],
            "CMAKE_CUDA_COMPILE_OBJECT",
        ),
        (
            "hip",
            ["-DCMAKE_HIP_LINK_EXECUTABLE=/usr/bin/env <CMAKE_HIP_COMPILER>"],
            "CMAKE_HIP_LINK_EXECUTABLE",
        ),
        (
            "cuda-device-link",
            [
                (
                    "-DCMAKE_CUDA_DEVICE_LINK_EXECUTABLE="
                    "/usr/bin/env <CMAKE_CUDA_COMPILER>"
                )
            ],
            "CMAKE_CUDA_DEVICE_LINK_EXECUTABLE",
        ),
    ]
    for name, arguments, expected in attacks:
        assert_configure_rejected(
            cmake,
            source,
            workdir / f"rejected-{name}",
            real_compiler,
            arguments,
            expected,
        )

    project_rule_attacks = [
        ("compile-and", "CMAKE_CXX_COMPILE_OBJECT"),
        ("link-or", "CMAKE_CXX_LINK_EXECUTABLE"),
        ("archive-pipe", "CMAKE_CXX_ARCHIVE_CREATE"),
        ("compile-semicolon", "CMAKE_CXX_COMPILE_OBJECT"),
        ("escaped-semicolon", "CMAKE_CXX_COMPILE_OBJECT"),
        ("compile-dollar-paren", "CMAKE_CXX_COMPILE_OBJECT"),
        ("compile-backtick", "CMAKE_CXX_COMPILE_OBJECT"),
        ("escaped-backtick", "CMAKE_CXX_COMPILE_OBJECT"),
        ("compile-dollar-variable", "CMAKE_CXX_COMPILE_OBJECT"),
        ("quoted-dollar", "CMAKE_CXX_COMPILE_OBJECT"),
        ("escaped-dollar", "CMAKE_CXX_COMPILE_OBJECT"),
        ("compile-glob", "CMAKE_CXX_COMPILE_OBJECT"),
        ("compile-redirection", "CMAKE_CXX_COMPILE_OBJECT"),
        ("compile-comment", "CMAKE_CXX_COMPILE_OBJECT"),
        ("compile-process-substitution", "CMAKE_CXX_COMPILE_OBJECT"),
        ("link-dollar-variable", "CMAKE_CXX_LINK_EXECUTABLE"),
        ("archive-glob", "CMAKE_CXX_ARCHIVE_CREATE"),
    ]
    for attack, expected in project_rule_attacks:
        assert_configure_rejected(
            cmake,
            source,
            workdir / f"rejected-project-rule-{attack}",
            real_compiler,
            [f"-DDEAC_FIXTURE_RULE_ATTACK={attack}"],
            expected,
        )

    # These protected operators survive both CMake's build-file generation and
    # POSIX shell parsing as one literal argument.
    for accepted in (
        "quoted-pipe",
        "quoted-semicolon",
        "quoted-backtick",
        "quoted-glob",
        "escaped-glob",
    ):
        accepted_build = workdir / f"accepted-{accepted}"
        configure(
            cmake,
            source,
            accepted_build,
            real_compiler,
            f"-DDEAC_FIXTURE_RULE_ATTACK={accepted}",
        )
        build(cmake, accepted_build)

    quoted_backtick_ninja = workdir / "accepted-quoted-backtick-ninja"
    configure(
        cmake,
        source,
        quoted_backtick_ninja,
        real_compiler,
        "-G",
        "Ninja",
        f"-DCMAKE_MAKE_PROGRAM={ninja}",
        "-DDEAC_FIXTURE_RULE_ATTACK=quoted-backtick",
    )
    build(cmake, quoted_backtick_ninja)

    assert_configure_rejected(
        cmake,
        source,
        workdir / "rejected-generic-ipo",
        real_compiler,
        ["-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON"],
        "INTERPROCEDURAL_OPTIMIZATION",
    )
    assert_configure_rejected(
        cmake,
        source,
        workdir / "rejected-release-ipo",
        real_compiler,
        ["-DCMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE=ON"],
        "INTERPROCEDURAL_OPTIMIZATION_RELEASE",
    )
    assert_configure_rejected(
        cmake,
        source,
        workdir / "rejected-dependency-release-ipo",
        real_compiler,
        ["-DDEAC_FIXTURE_DEPENDENCY_IPO_RELEASE=ON"],
        "INTERPROCEDURAL_OPTIMIZATION_RELEASE",
    )
    assert_configure_rejected(
        cmake,
        source,
        workdir / "rejected-multi-release-ipo",
        real_compiler,
        [
            "-G",
            "Ninja Multi-Config",
            f"-DCMAKE_MAKE_PROGRAM={ninja}",
            "-DCMAKE_CONFIGURATION_TYPES=Debug;Release;RelWithDebInfo",
            "-DCMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE=ON",
        ],
        "INTERPROCEDURAL_OPTIMIZATION_RELEASE",
    )

    for language in ("CUDA", "HIP"):
        assert_configure_rejected(
            cmake,
            source,
            workdir / f"rejected-native-{language.lower()}",
            real_compiler,
            [f"-DDEAC_FIXTURE_NATIVE_LANGUAGE_SHAPE={language}"],
            f"native CMake {language} language",
        )

    override = workdir / "user-rules.cmake"
    write(override, "# intentionally empty override route\n")
    assert_configure_rejected(
        cmake,
        source,
        workdir / "rejected-user-rules",
        real_compiler,
        [f"-DCMAKE_USER_MAKE_RULES_OVERRIDE={override}"],
        "CMAKE_USER_MAKE_RULES_OVERRIDE",
    )
    toolchain = workdir / "toolchain.cmake"
    write(toolchain, f"set(CMAKE_CXX_COMPILER {json.dumps(str(real_compiler))})\n")
    assert_configure_rejected(
        cmake,
        source,
        workdir / "rejected-toolchain",
        real_compiler,
        [f"-DCMAKE_TOOLCHAIN_FILE={toolchain}"],
        "CMAKE_TOOLCHAIN_FILE",
    )
    assert_configure_rejected(
        cmake,
        source,
        workdir / "rejected-module-path",
        real_compiler,
        [f"-DCMAKE_MODULE_PATH={workdir}"],
        "CMAKE_MODULE_PATH",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmake", required=True)
    parser.add_argument("--ninja", required=True)
    parser.add_argument("--solver-source-root", required=True)
    parser.add_argument("--cxx-compiler", required=True)
    parser.add_argument("--archiver", required=True)
    parser.add_argument("--workdir", required=True)
    args = parser.parse_args()

    workdir = Path(args.workdir).resolve()
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    source = workdir / "fixture-source"
    solver_source = Path(args.solver_source_root).resolve()
    real_compiler = Path(args.cxx_compiler).resolve()
    real_archiver = Path(args.archiver).resolve()
    ninja = Path(args.ninja).resolve()
    compiler_shim = workdir / "compiler-shim"
    create_fixture_source(solver_source, source)

    test_single_config_configuration_rejections(
        args.cmake, source, workdir, real_compiler
    )
    test_single_config_refresh_and_replacement(
        args.cmake, source, workdir, compiler_shim, real_compiler
    )
    test_single_config_ninja(
        args.cmake, ninja, source, workdir, real_compiler
    )
    test_material_flag_reconfiguration(
        args.cmake, source, workdir, real_compiler
    )
    test_backend_reconfiguration(
        args.cmake, source, workdir, real_compiler
    )
    test_archive_tool_reconfiguration(
        args.cmake, source, workdir, real_compiler, real_archiver
    )
    test_ninja_multi_config(
        args.cmake, ninja, source, workdir, compiler_shim
    )
    test_parallel_receipt_consumers(
        args.cmake, ninja, solver_source, workdir, real_compiler
    )
    test_shared_dependency_receipt_consumers(
        args.cmake, ninja, solver_source, workdir, real_compiler
    )
    test_nested_build_root_canonicalization(
        args.cmake, ninja, solver_source, workdir, real_compiler
    )
    test_mapped_hipblas_multi_config(
        args.cmake, ninja, solver_source, workdir, real_compiler
    )
    test_selected_config_provider_collision(
        args.cmake, ninja, solver_source, workdir, real_compiler
    )
    test_configless_mapped_hipblas(
        args.cmake, ninja, source, workdir, real_compiler
    )
    test_hipblas_receipt_contract(
        args.cmake, ninja, source, workdir, real_compiler
    )
    test_hipblas_receipt_rejections(
        args.cmake, source, workdir, real_compiler
    )
    test_control_byte_link_artifact_rejections(
        args.cmake, source, workdir, real_compiler
    )
    test_effective_shell_input_rejections(
        args.cmake, source, workdir, real_compiler
    )
    test_deferred_seal_rejections(
        args.cmake, source, workdir, real_compiler
    )
    test_parent_deferred_seal_rejection(
        args.cmake, ninja, solver_source, workdir, real_compiler
    )
    test_subdirectory_dependency_directory_attribution(
        args.cmake, source, workdir, real_compiler
    )
    test_dependency_fingerprint_rejections(
        args.cmake, source, workdir, real_compiler
    )
    test_json_control_encoding(
        args.cmake, source, workdir, real_compiler
    )
    test_file_api_shape_rejections(
        args.cmake, source, workdir, real_compiler
    )
    test_compiler_target_reply_rejections(
        args.cmake, source, workdir, real_compiler
    )
    test_hidden_receipt_config_rejection(
        args.cmake, ninja, source, workdir, real_compiler
    )
    test_output_path_symlink_rejections(
        args.cmake, source, workdir, real_compiler
    )
    test_rule_override_rejections(
        args.cmake, ninja, source, workdir, real_compiler
    )


if __name__ == "__main__":
    main()
