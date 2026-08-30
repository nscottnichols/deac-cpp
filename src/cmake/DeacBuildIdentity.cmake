include_guard(GLOBAL)

include(CMakeParseArguments)
find_package(Git QUIET)

# Git honors these variables ahead of -C and repository-local discovery.  A
# build identity must describe SOURCE_ROOT, not an inherited repository chosen
# by the caller's environment.  Keep this list compatible with CMake 3.18's
# `cmake -E env --unset=...` command.
set(_DEAC_GIT_REDIRECTION_ENVIRONMENT
    GIT_DIR
    GIT_WORK_TREE
    GIT_INDEX_FILE
    GIT_COMMON_DIR
    GIT_OBJECT_DIRECTORY
    GIT_ALTERNATE_OBJECT_DIRECTORIES
    GIT_QUARANTINE_PATH
    GIT_NAMESPACE
    GIT_SHALLOW_FILE
    GIT_CEILING_DIRECTORIES
    GIT_DISCOVERY_ACROSS_FILESYSTEM
    GIT_PREFIX
    GIT_EXEC_PATH
    GIT_CONFIG_COUNT
    GIT_CONFIG_PARAMETERS
    GIT_CONFIG_GLOBAL
    GIT_CONFIG_SYSTEM
    GIT_CONFIG_NOSYSTEM
    GIT_REPLACE_REF_BASE
    GIT_LITERAL_PATHSPECS
    GIT_GLOB_PATHSPECS
    GIT_NOGLOB_PATHSPECS
    GIT_ICASE_PATHSPECS)

function(_deac_git_command output_variable source_root)
    set(_command "${CMAKE_COMMAND}" -E env)
    foreach(_environment_name IN LISTS _DEAC_GIT_REDIRECTION_ENVIRONMENT)
        list(APPEND _command "--unset=${_environment_name}")
    endforeach()
    list(APPEND _command "${GIT_EXECUTABLE}" -C "${source_root}")
    set(${output_variable} "${_command}" PARENT_SCOPE)
endfunction()

function(_deac_read_semantic_version source_root output_variable)
    set(_version_file "${source_root}/VERSION")
    if(NOT EXISTS "${_version_file}")
        message(FATAL_ERROR
            "DEAC semantic-version file does not exist: ${_version_file}")
    endif()
    file(READ "${_version_file}" _semantic_version)
    string(STRIP "${_semantic_version}" _semantic_version)
    if(NOT _semantic_version MATCHES
            "^[0-9]+\\.[0-9]+\\.[0-9]+(-[0-9A-Za-z.-]+)?(\\+[0-9A-Za-z.-]+)?$")
        message(FATAL_ERROR
            "VERSION must contain one injection-safe semantic version, got "
            "'${_semantic_version}'")
    endif()
    set(${output_variable} "${_semantic_version}" PARENT_SCOPE)
endfunction()

function(_deac_compute_build_identity source_root)
    get_filename_component(_source_root "${source_root}" REALPATH)
    _deac_read_semantic_version("${_source_root}" _semantic_version)

    # Unavailable is the conservative default.  A SHA is published only after
    # every sanitized probe succeeds and Git's exact top level equals the
    # source root, which prevents archives from borrowing an enclosing repo.
    set(_source_sha "")
    set(_source_state "unavailable")
    set(_source_sha_available false)
    if(Git_FOUND)
        _deac_git_command(_git_command "${_source_root}")
        execute_process(
            COMMAND ${_git_command} rev-parse --show-toplevel
            RESULT_VARIABLE _git_root_result
            OUTPUT_VARIABLE _git_root
            ERROR_QUIET
            OUTPUT_STRIP_TRAILING_WHITESPACE)
        if(_git_root_result EQUAL 0)
            get_filename_component(_git_root "${_git_root}" REALPATH)
        endif()

        if(_git_root_result EQUAL 0 AND
                "${_git_root}" STREQUAL "${_source_root}")
            execute_process(
                COMMAND ${_git_command} rev-parse --verify "HEAD^{commit}"
                RESULT_VARIABLE _git_sha_result
                OUTPUT_VARIABLE _git_sha
                ERROR_QUIET
                OUTPUT_STRIP_TRAILING_WHITESPACE)
            string(LENGTH "${_git_sha}" _git_sha_length)
            if(_git_sha_result EQUAL 0 AND
                    _git_sha_length EQUAL 40 AND
                    _git_sha MATCHES "^[0-9A-Fa-f]+$")
                execute_process(
                    COMMAND ${_git_command}
                        status --porcelain=v1 --untracked-files=normal --
                        VERSION src
                    RESULT_VARIABLE _git_status_result
                    OUTPUT_VARIABLE _git_status
                    ERROR_QUIET
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
                if(_git_status_result EQUAL 0)
                    string(TOLOWER "${_git_sha}" _source_sha)
                    set(_source_sha_available true)
                    if(_git_status STREQUAL "")
                        set(_source_state "clean")
                    else()
                        set(_source_state "dirty")
                    endif()
                endif()
            endif()
        endif()
    endif()

    if(_source_sha_available)
        set(_json_source_sha "\"${_source_sha}\"")
    else()
        set(_json_source_sha "null")
    endif()
    string(CONCAT _canonical_json
        "{\"schema_version\":1,\"semantic_version\":\"${_semantic_version}\","
        "\"source_sha\":${_json_source_sha},"
        "\"source_state\":\"${_source_state}\"}")
    string(SHA256 _canonical_digest "${_canonical_json}")

    set(DEAC_BUILD_IDENTITY_SEMANTIC_VERSION
        "${_semantic_version}" PARENT_SCOPE)
    set(DEAC_BUILD_IDENTITY_SOURCE_SHA "${_source_sha}" PARENT_SCOPE)
    set(DEAC_BUILD_IDENTITY_SOURCE_STATE "${_source_state}" PARENT_SCOPE)
    set(DEAC_BUILD_IDENTITY_SOURCE_SHA_AVAILABLE
        "${_source_sha_available}" PARENT_SCOPE)
    set(DEAC_BUILD_IDENTITY_JSON_SOURCE_SHA
        "${_json_source_sha}" PARENT_SCOPE)
    set(DEAC_BUILD_IDENTITY_CANONICAL_JSON
        "${_canonical_json}" PARENT_SCOPE)
    set(DEAC_BUILD_IDENTITY_CANONICAL_DIGEST
        "${_canonical_digest}" PARENT_SCOPE)
endfunction()

function(_deac_watch_git_path source_root git_path_spec)
    _deac_git_command(_git_command "${source_root}")
    execute_process(
        COMMAND ${_git_command} rev-parse --git-path "${git_path_spec}"
        RESULT_VARIABLE _git_path_result
        OUTPUT_VARIABLE _git_path
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT _git_path_result EQUAL 0)
        return()
    endif()
    get_filename_component(
        _git_path "${_git_path}" ABSOLUTE BASE_DIR "${source_root}")

    # The exact-path glob is deliberately retained even when the loose ref or
    # packed-refs file is absent.  CONFIGURE_DEPENDS then detects its creation
    # by a later metadata-only commit.  Existing files are also direct content
    # dependencies.  Build-time identity generation remains the correctness
    # backstop when filesystem timestamp resolution is coarse.
    file(GLOB _git_path_match
        CONFIGURE_DEPENDS
        LIST_DIRECTORIES false
        "${_git_path}")
    if(EXISTS "${_git_path}")
        set_property(DIRECTORY APPEND PROPERTY
            CMAKE_CONFIGURE_DEPENDS "${_git_path}")
    endif()
endfunction()

function(_deac_watch_build_identity_inputs source_root)
    file(GLOB_RECURSE _source_inputs
        CONFIGURE_DEPENDS
        LIST_DIRECTORIES false
        "${source_root}/src/*")
    list(APPEND _source_inputs "${source_root}/VERSION")
    list(REMOVE_DUPLICATES _source_inputs)
    set_property(DIRECTORY APPEND PROPERTY
        CMAKE_CONFIGURE_DEPENDS ${_source_inputs})

    if(NOT Git_FOUND)
        return()
    endif()
    _deac_git_command(_git_command "${source_root}")
    execute_process(
        COMMAND ${_git_command} rev-parse --show-toplevel
        RESULT_VARIABLE _git_root_result
        OUTPUT_VARIABLE _git_root
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT _git_root_result EQUAL 0)
        return()
    endif()
    get_filename_component(_git_root "${_git_root}" REALPATH)
    if(NOT "${_git_root}" STREQUAL "${source_root}")
        return()
    endif()

    foreach(_git_path_spec HEAD index packed-refs commondir)
        _deac_watch_git_path("${source_root}" "${_git_path_spec}")
    endforeach()
    execute_process(
        COMMAND ${_git_command} symbolic-ref --quiet HEAD
        RESULT_VARIABLE _symbolic_ref_result
        OUTPUT_VARIABLE _symbolic_ref
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(_symbolic_ref_result EQUAL 0 AND NOT _symbolic_ref STREQUAL "")
        _deac_watch_git_path("${source_root}" "${_symbolic_ref}")
    endif()
endfunction()

function(deac_add_build_identity)
    cmake_parse_arguments(
        PARSE_ARGV 0
        DEAC_IDENTITY
        ""
        "SOURCE_ROOT;GENERATED_DIRECTORY;IDENTITY_NAME;RECEIPT"
        "")
    foreach(_required_argument
            SOURCE_ROOT
            GENERATED_DIRECTORY
            IDENTITY_NAME
            RECEIPT)
        if(NOT DEAC_IDENTITY_${_required_argument})
            message(FATAL_ERROR
                "deac_add_build_identity requires ${_required_argument}")
        endif()
    endforeach()

    get_filename_component(
        _source_root "${DEAC_IDENTITY_SOURCE_ROOT}" REALPATH)
    _deac_read_semantic_version("${_source_root}" _semantic_version)
    _deac_watch_build_identity_inputs("${_source_root}")

    set(_generated_header
        "${DEAC_IDENTITY_GENERATED_DIRECTORY}/deac_build_identity_data.hpp")
    set(_receipt "${DEAC_IDENTITY_RECEIPT}")
    string(CONCAT _refresh
        "${DEAC_IDENTITY_GENERATED_DIRECTORY}/"
        "${DEAC_IDENTITY_IDENTITY_NAME}-build-identity.refresh")
    file(MAKE_DIRECTORY "${DEAC_IDENTITY_GENERATED_DIRECTORY}")

    set(_git_argument)
    if(Git_FOUND)
        list(APPEND _git_argument
            "-DGIT_EXECUTABLE:FILEPATH=${GIT_EXECUTABLE}")
    endif()
    # The symbolic primary output is intentionally never created.  Do not list
    # the header or receipt as BYPRODUCTS: Ninja would add `restat`, and an
    # equal coarse timestamp could then suppress the dependent compile/link
    # even though their canonical bytes changed.  The target helper registers
    # those two side effects for cleaning instead.
    add_custom_command(
        OUTPUT "${_refresh}"
        COMMAND
            "${CMAKE_COMMAND}"
            "-DDEAC_BUILD_IDENTITY_SOURCE_ROOT:PATH=${_source_root}"
            "-DDEAC_BUILD_IDENTITY_OUTPUT_HEADER:FILEPATH=${_generated_header}"
            "-DDEAC_BUILD_IDENTITY_OUTPUT_RECEIPT:FILEPATH=${_receipt}"
            ${_git_argument}
            -P "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/GenerateDeacBuildIdentity.cmake"
        DEPENDS
            "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/DeacBuildIdentity.cmake"
            "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/GenerateDeacBuildIdentity.cmake"
            "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/deac_build_identity_data.hpp.in"
            "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/deac_build_identity.json.in"
            "${_source_root}/VERSION"
        COMMENT "Refreshing canonical DEAC build identity"
        VERBATIM)
    set_source_files_properties(
        "${_refresh}"
        PROPERTIES GENERATED TRUE SYMBOLIC TRUE)
    set_source_files_properties(
        "${_generated_header}"
        PROPERTIES GENERATED TRUE)

    get_filename_component(
        _support_directory
        "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../deac/src"
        REALPATH)
    set(DEAC_BUILD_IDENTITY_SEMANTIC_VERSION
        "${_semantic_version}" PARENT_SCOPE)
    set(DEAC_BUILD_IDENTITY_GENERATED_HEADER
        "${_generated_header}" PARENT_SCOPE)
    set(DEAC_BUILD_IDENTITY_GENERATED_INCLUDE_DIRECTORY
        "${DEAC_IDENTITY_GENERATED_DIRECTORY}" PARENT_SCOPE)
    set(DEAC_BUILD_IDENTITY_IMPLEMENTATION_SOURCE
        "${_support_directory}/build_identity.cpp" PARENT_SCOPE)
    set(DEAC_BUILD_IDENTITY_SUPPORT_INCLUDE_DIRECTORY
        "${_support_directory}" PARENT_SCOPE)
    set(DEAC_BUILD_IDENTITY_RECEIPT "${_receipt}" PARENT_SCOPE)
    set(DEAC_BUILD_IDENTITY_REFRESH "${_refresh}" PARENT_SCOPE)
endfunction()

function(deac_target_add_build_identity target_name)
    if(NOT TARGET "${target_name}")
        message(FATAL_ERROR
            "deac_target_add_build_identity requires an existing target")
    endif()
    foreach(_required_variable
            DEAC_BUILD_IDENTITY_IMPLEMENTATION_SOURCE
            DEAC_BUILD_IDENTITY_GENERATED_HEADER
            DEAC_BUILD_IDENTITY_SUPPORT_INCLUDE_DIRECTORY
            DEAC_BUILD_IDENTITY_GENERATED_INCLUDE_DIRECTORY
            DEAC_BUILD_IDENTITY_RECEIPT
            DEAC_BUILD_IDENTITY_REFRESH)
        if(NOT DEFINED ${_required_variable} OR
                "${${_required_variable}}" STREQUAL "")
            message(FATAL_ERROR
                "deac_add_build_identity must run before "
                "deac_target_add_build_identity (${_required_variable} missing)")
        endif()
    endforeach()

    target_sources("${target_name}" PRIVATE
        "${DEAC_BUILD_IDENTITY_IMPLEMENTATION_SOURCE}"
        "${DEAC_BUILD_IDENTITY_GENERATED_HEADER}")
    target_include_directories("${target_name}" PRIVATE
        "${DEAC_BUILD_IDENTITY_SUPPORT_INCLUDE_DIRECTORY}"
        "${DEAC_BUILD_IDENTITY_GENERATED_INCLUDE_DIRECTORY}")

    # A build-time refresh can discover a changed Git identity even when every
    # relevant filesystem timestamp is equal at the platform's resolution.
    # Make both the tiny identity implementation object and the link depend on
    # the symbolic refresh output so those canonical bytes reach the executable
    # during the same ordinary build.
    set_property(SOURCE "${DEAC_BUILD_IDENTITY_IMPLEMENTATION_SOURCE}"
        APPEND PROPERTY OBJECT_DEPENDS "${DEAC_BUILD_IDENTITY_REFRESH}")
    set_property(TARGET "${target_name}" APPEND PROPERTY
        LINK_DEPENDS "${DEAC_BUILD_IDENTITY_REFRESH}")
    set_property(TARGET "${target_name}" APPEND PROPERTY
        ADDITIONAL_CLEAN_FILES
            "${DEAC_BUILD_IDENTITY_GENERATED_HEADER};${DEAC_BUILD_IDENTITY_RECEIPT}")
endfunction()
