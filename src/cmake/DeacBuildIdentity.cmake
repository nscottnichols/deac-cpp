include_guard(GLOBAL)

find_package(Git QUIET)

function(deac_configure_build_identity)
    cmake_parse_arguments(
        PARSE_ARGV 0
        DEAC_IDENTITY
        ""
        "SOURCE_ROOT;OUTPUT_HEADER;OUTPUT_RECEIPT"
        "")

    foreach(required_argument SOURCE_ROOT OUTPUT_HEADER OUTPUT_RECEIPT)
        if(NOT DEAC_IDENTITY_${required_argument})
            message(FATAL_ERROR
                "deac_configure_build_identity requires ${required_argument}")
        endif()
    endforeach()

    get_filename_component(
        DEAC_BUILD_IDENTITY_SOURCE_ROOT
        "${DEAC_IDENTITY_SOURCE_ROOT}"
        REALPATH)

    set(_deac_version_file "${DEAC_BUILD_IDENTITY_SOURCE_ROOT}/VERSION")
    if(NOT EXISTS "${_deac_version_file}")
        message(FATAL_ERROR
            "DEAC semantic-version file does not exist: ${_deac_version_file}")
    endif()
    file(READ "${_deac_version_file}" DEAC_BUILD_IDENTITY_SEMANTIC_VERSION)
    string(STRIP
        "${DEAC_BUILD_IDENTITY_SEMANTIC_VERSION}"
        DEAC_BUILD_IDENTITY_SEMANTIC_VERSION)
    if(NOT DEAC_BUILD_IDENTITY_SEMANTIC_VERSION MATCHES
            "^[0-9]+\\.[0-9]+\\.[0-9]+(-[0-9A-Za-z.-]+)?(\\+[0-9A-Za-z.-]+)?$")
        message(FATAL_ERROR
            "VERSION must contain one injection-safe semantic version, got "
            "'${DEAC_BUILD_IDENTITY_SEMANTIC_VERSION}'")
    endif()

    # Never inherit a revision from an enclosing, unrelated repository.  This
    # exact-root check is what makes extracted release archives report the
    # explicit unavailable fallback even when unpacked below another checkout.
    set(DEAC_BUILD_IDENTITY_SOURCE_SHA "")
    set(DEAC_BUILD_IDENTITY_SOURCE_STATE "unavailable")
    set(DEAC_BUILD_IDENTITY_SOURCE_SHA_AVAILABLE false)
    if(Git_FOUND)
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" -C
                "${DEAC_BUILD_IDENTITY_SOURCE_ROOT}"
                rev-parse --show-toplevel
            RESULT_VARIABLE _deac_git_root_result
            OUTPUT_VARIABLE _deac_git_root
            ERROR_QUIET
            OUTPUT_STRIP_TRAILING_WHITESPACE)
        if(_deac_git_root_result EQUAL 0)
            get_filename_component(_deac_git_root "${_deac_git_root}" REALPATH)
        endif()

        if(_deac_git_root_result EQUAL 0 AND
                "${_deac_git_root}" STREQUAL
                    "${DEAC_BUILD_IDENTITY_SOURCE_ROOT}")
            execute_process(
                COMMAND "${GIT_EXECUTABLE}" -C
                    "${DEAC_BUILD_IDENTITY_SOURCE_ROOT}"
                    rev-parse --verify "HEAD^{commit}"
                RESULT_VARIABLE _deac_git_sha_result
                OUTPUT_VARIABLE _deac_git_sha
                ERROR_QUIET
                OUTPUT_STRIP_TRAILING_WHITESPACE)
            string(LENGTH "${_deac_git_sha}" _deac_git_sha_length)
            if(_deac_git_sha_result EQUAL 0 AND
                    _deac_git_sha_length EQUAL 40 AND
                    _deac_git_sha MATCHES "^[0-9A-Fa-f]+$")
                execute_process(
                    COMMAND "${GIT_EXECUTABLE}" -C
                        "${DEAC_BUILD_IDENTITY_SOURCE_ROOT}"
                        status --porcelain=v1 --untracked-files=normal --
                        VERSION src
                    RESULT_VARIABLE _deac_git_status_result
                    OUTPUT_VARIABLE _deac_git_status
                    ERROR_QUIET
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
                if(_deac_git_status_result EQUAL 0)
                    string(TOLOWER
                        "${_deac_git_sha}"
                        DEAC_BUILD_IDENTITY_SOURCE_SHA)
                    set(DEAC_BUILD_IDENTITY_SOURCE_SHA_AVAILABLE true)
                    if(_deac_git_status STREQUAL "")
                        set(DEAC_BUILD_IDENTITY_SOURCE_STATE "clean")
                    else()
                        set(DEAC_BUILD_IDENTITY_SOURCE_STATE "dirty")
                    endif()
                endif()
            endif()
        endif()
    endif()

    if(DEAC_BUILD_IDENTITY_SOURCE_SHA_AVAILABLE)
        set(DEAC_BUILD_IDENTITY_JSON_SOURCE_SHA
            "\"${DEAC_BUILD_IDENTITY_SOURCE_SHA}\"")
    else()
        set(DEAC_BUILD_IDENTITY_JSON_SOURCE_SHA "null")
    endif()

    # A normal build must regenerate the configured identity after a relevant
    # edit, staging operation, commit, checkout, or ref update.  CONFIGURE_DEPENDS
    # catches additions/removals while CMAKE_CONFIGURE_DEPENDS catches content
    # changes to the files already present.
    file(GLOB_RECURSE _deac_identity_source_inputs
        CONFIGURE_DEPENDS
        LIST_DIRECTORIES false
        "${DEAC_BUILD_IDENTITY_SOURCE_ROOT}/src/*")
    list(APPEND _deac_identity_source_inputs "${_deac_version_file}")

    if(Git_FOUND AND DEAC_BUILD_IDENTITY_SOURCE_SHA_AVAILABLE)
        set(_deac_git_metadata_specs HEAD index packed-refs)
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" -C
                "${DEAC_BUILD_IDENTITY_SOURCE_ROOT}"
                symbolic-ref --quiet HEAD
            RESULT_VARIABLE _deac_symbolic_ref_result
            OUTPUT_VARIABLE _deac_symbolic_ref
            ERROR_QUIET
            OUTPUT_STRIP_TRAILING_WHITESPACE)
        if(_deac_symbolic_ref_result EQUAL 0 AND NOT _deac_symbolic_ref STREQUAL "")
            list(APPEND _deac_git_metadata_specs "${_deac_symbolic_ref}")
        endif()

        foreach(_deac_git_metadata_spec IN LISTS _deac_git_metadata_specs)
            execute_process(
                COMMAND "${GIT_EXECUTABLE}" -C
                    "${DEAC_BUILD_IDENTITY_SOURCE_ROOT}"
                    rev-parse --git-path "${_deac_git_metadata_spec}"
                RESULT_VARIABLE _deac_git_path_result
                OUTPUT_VARIABLE _deac_git_path
                ERROR_QUIET
                OUTPUT_STRIP_TRAILING_WHITESPACE)
            if(_deac_git_path_result EQUAL 0)
                get_filename_component(
                    _deac_git_path
                    "${_deac_git_path}"
                    ABSOLUTE
                    BASE_DIR "${DEAC_BUILD_IDENTITY_SOURCE_ROOT}")
                if(EXISTS "${_deac_git_path}")
                    list(APPEND
                        _deac_identity_source_inputs
                        "${_deac_git_path}")
                endif()
            endif()
        endforeach()
    endif()

    list(REMOVE_DUPLICATES _deac_identity_source_inputs)
    set_property(DIRECTORY APPEND PROPERTY
        CMAKE_CONFIGURE_DEPENDS ${_deac_identity_source_inputs})

    get_filename_component(
        _deac_identity_header_directory
        "${DEAC_IDENTITY_OUTPUT_HEADER}"
        DIRECTORY)
    get_filename_component(
        _deac_identity_receipt_directory
        "${DEAC_IDENTITY_OUTPUT_RECEIPT}"
        DIRECTORY)
    file(MAKE_DIRECTORY
        "${_deac_identity_header_directory}"
        "${_deac_identity_receipt_directory}")
    configure_file(
        "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/deac_build_identity.hpp.in"
        "${DEAC_IDENTITY_OUTPUT_HEADER}"
        @ONLY)
    configure_file(
        "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/deac_build_identity.json.in"
        "${DEAC_IDENTITY_OUTPUT_RECEIPT}"
        @ONLY)

    set(DEAC_BUILD_IDENTITY_SEMANTIC_VERSION
        "${DEAC_BUILD_IDENTITY_SEMANTIC_VERSION}"
        PARENT_SCOPE)
    set(DEAC_BUILD_IDENTITY_SOURCE_SHA
        "${DEAC_BUILD_IDENTITY_SOURCE_SHA}"
        PARENT_SCOPE)
    set(DEAC_BUILD_IDENTITY_SOURCE_STATE
        "${DEAC_BUILD_IDENTITY_SOURCE_STATE}"
        PARENT_SCOPE)
    set(DEAC_BUILD_IDENTITY_SOURCE_SHA_AVAILABLE
        "${DEAC_BUILD_IDENTITY_SOURCE_SHA_AVAILABLE}"
        PARENT_SCOPE)
endfunction()
