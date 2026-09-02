include_guard(GLOBAL)

include(CMakeParseArguments)
find_package(Git QUIET)

function(_deac_build_receipt_require_plain value label)
    if("${value}" MATCHES "[;\r\n]")
        message(FATAL_ERROR
            "${label} contains a list separator or line break and cannot be "
            "represented safely in the build-receipt command")
    endif()
endfunction()

function(_deac_build_receipt_require_safe_path_text value label)
    if("${value}" STREQUAL "" OR "${value}" MATCHES ";")
        message(FATAL_ERROR "build-receipt ${label} is not a safe path")
    endif()
    string(HEX "${value}" _value_hex)
    string(LENGTH "${_value_hex}" _value_hex_length)
    set(_hex_index 0)
    while(_hex_index LESS _value_hex_length)
        string(SUBSTRING "${_value_hex}" ${_hex_index} 2 _byte_hex)
        if(_byte_hex MATCHES "^(0[0-9a-f]|1[0-9a-f]|7f)$")
            message(FATAL_ERROR
                "build-receipt ${label} contains a control byte")
        endif()
        math(EXPR _hex_index "${_hex_index} + 2")
    endwhile()
endfunction()

function(_deac_build_receipt_require_build_tree_descendant
        value build_root label)
    set(_normalized_build_root "${build_root}")
    cmake_path(NORMAL_PATH _normalized_build_root)
    cmake_path(IS_PREFIX _normalized_build_root "${value}" NORMALIZE
        _is_build_tree_path)
    if(NOT _is_build_tree_path OR value STREQUAL _normalized_build_root)
        message(FATAL_ERROR
            "build-receipt ${label} must be a descendant of CMAKE_BINARY_DIR")
    endif()
endfunction()

function(_deac_build_receipt_reject_descendant_symlinks
        value build_root label)
    _deac_build_receipt_require_build_tree_descendant(
        "${value}" "${build_root}" "${label}")
    file(RELATIVE_PATH _relative_path "${build_root}" "${value}")
    string(REPLACE "/" ";" _path_components "${_relative_path}")
    set(_candidate_path "${build_root}")
    foreach(_path_component IN LISTS _path_components)
        if(_path_component STREQUAL "")
            continue()
        endif()
        cmake_path(APPEND _candidate_path "${_path_component}")
        if(IS_SYMLINK "${_candidate_path}")
            message(FATAL_ERROR
                "build-receipt ${label} must not contain a symlink "
                "component below CMAKE_BINARY_DIR")
        endif()
    endforeach()
endfunction()

function(_deac_build_receipt_normalize_generated_directory
        value build_root output)
    _deac_build_receipt_require_safe_path_text(
        "${value}" "GENERATED_DIRECTORY")
    if("${value}" MATCHES "[<>]" OR "${value}" MATCHES "\\$<" OR
            NOT IS_ABSOLUTE "${value}")
        message(FATAL_ERROR
            "build-receipt GENERATED_DIRECTORY must be an absolute "
            "generator-expression-free path")
    endif()
    if("${value}" MATCHES "(^|/)\\.\\.?(/|$)")
        message(FATAL_ERROR
            "build-receipt GENERATED_DIRECTORY must not contain . or .. "
            "path components")
    endif()
    set(_normalized "${value}")
    cmake_path(NORMAL_PATH _normalized)
    if(NOT _normalized STREQUAL "${value}")
        message(FATAL_ERROR
            "build-receipt GENERATED_DIRECTORY must be normalized")
    endif()
    _deac_build_receipt_require_build_tree_descendant(
        "${_normalized}" "${build_root}" "GENERATED_DIRECTORY")
    _deac_build_receipt_reject_descendant_symlinks(
        "${_normalized}" "${build_root}" "GENERATED_DIRECTORY")
    set(${output} "${_normalized}" PARENT_SCOPE)
endfunction()

function(_deac_build_receipt_normalize_receipt_path
        value require_config build_root output)
    _deac_build_receipt_require_safe_path_text("${value}" "RECEIPT")
    if(NOT IS_ABSOLUTE "${value}")
        message(FATAL_ERROR
            "build-receipt RECEIPT must be an absolute path")
    endif()

    string(REGEX MATCHALL "\\$<CONFIG>" _config_tokens "${value}")
    list(LENGTH _config_tokens _config_token_count)
    if(require_config AND NOT _config_token_count EQUAL 1)
        message(FATAL_ERROR
            "build-receipt RECEIPT must contain exactly one literal "
            "$<CONFIG> path component for multi-config generators")
    elseif(_config_token_count GREATER 1)
        message(FATAL_ERROR
            "build-receipt RECEIPT contains more than one literal "
            "$<CONFIG> token")
    endif()
    if(_config_token_count EQUAL 1 AND
            NOT "${value}" MATCHES "(^|/)\\$<CONFIG>(/|$)")
        message(FATAL_ERROR
            "build-receipt RECEIPT requires $<CONFIG> as a complete path "
            "component")
    endif()

    string(REPLACE "$<CONFIG>" "" _without_config "${value}")
    if(_without_config MATCHES "\\$<" OR _without_config MATCHES "[<>]")
        message(FATAL_ERROR
            "build-receipt RECEIPT contains unsupported "
            "generator-expression syntax")
    endif()

    string(REPLACE "$<CONFIG>" "DEAC_CONFIG_COMPONENT"
        _normalization_path "${value}")
    if(_normalization_path MATCHES "(^|/)\\.\\.?(/|$)")
        message(FATAL_ERROR
            "build-receipt RECEIPT must not contain . or .. path components")
    endif()
    set(_normalized_path "${_normalization_path}")
    cmake_path(NORMAL_PATH _normalized_path)
    if(NOT _normalized_path STREQUAL _normalization_path)
        message(FATAL_ERROR "build-receipt RECEIPT must be normalized")
    endif()
    _deac_build_receipt_require_build_tree_descendant(
        "${_normalized_path}" "${build_root}" "RECEIPT")
    set(${output} "${value}" PARENT_SCOPE)
endfunction()

function(_deac_build_receipt_normalize_link_library_artifact
        artifact output)
    set(_selected_artifacts_output "")
    if(ARGC GREATER 2)
        set(_selected_artifacts_output "${ARGV2}")
    endif()
    _deac_build_receipt_require_safe_path_text(
        "${artifact}" "link-library artifact")
    string(FIND "${artifact}" "$<" _generator_expression_position)
    if(_generator_expression_position EQUAL -1)
        if("${artifact}" MATCHES "[<>]" OR
                NOT IS_ABSOLUTE "${artifact}" OR
                NOT EXISTS "${artifact}" OR IS_DIRECTORY "${artifact}")
            message(FATAL_ERROR
                "build-receipt link-library artifact must be an absolute "
                "regular file: ${artifact}")
        endif()
        set(_normalized_artifact "${artifact}")
        cmake_path(NORMAL_PATH _normalized_artifact)
        set(${output} "${_normalized_artifact}" PARENT_SCOPE)
        if(NOT _selected_artifacts_output STREQUAL "")
            set(_selected_artifacts)
            get_property(_artifact_is_multi_config GLOBAL
                PROPERTY GENERATOR_IS_MULTI_CONFIG)
            if(_artifact_is_multi_config)
                foreach(_configuration IN LISTS CMAKE_CONFIGURATION_TYPES)
                    list(APPEND _selected_artifacts "${_normalized_artifact}")
                endforeach()
            else()
                list(APPEND _selected_artifacts "${_normalized_artifact}")
            endif()
            set(${_selected_artifacts_output}
                "${_selected_artifacts}" PARENT_SCOPE)
        endif()
        return()
    endif()

    # The provider is resolved at configure time.  Multi-config builds need
    # only one target-free $<$<CONFIG:name>:absolute-file> segment per config.
    get_property(_artifact_is_multi_config GLOBAL
        PROPERTY GENERATOR_IS_MULTI_CONFIG)
    if(NOT _generator_expression_position EQUAL 0 OR
            NOT _artifact_is_multi_config OR
            "${CMAKE_CONFIGURATION_TYPES}" STREQUAL "")
        message(FATAL_ERROR
            "build-receipt link-library artifact uses an unsupported "
            "generator expression: ${artifact}")
    endif()
    set(_remaining_artifact "${artifact}")
    set(_normalized_artifact "")
    set(_configuration_artifacts)
    foreach(_configuration IN LISTS CMAKE_CONFIGURATION_TYPES)
        if(NOT _configuration MATCHES "^[A-Za-z0-9_.+-]+$" OR
                _configuration STREQUAL "." OR
                _configuration STREQUAL "..")
            message(FATAL_ERROR
                "build-receipt configuration is not path-safe: "
                "${_configuration}")
        endif()
        set(_segment_prefix "$<$<CONFIG:${_configuration}>:")
        string(FIND "${_remaining_artifact}" "${_segment_prefix}"
            _segment_prefix_position)
        if(NOT _segment_prefix_position EQUAL 0)
            message(FATAL_ERROR
                "build-receipt link-library artifact must contain one "
                "ordered config-only segment for ${_configuration}: "
                "${artifact}")
        endif()
        string(LENGTH "${_segment_prefix}" _segment_prefix_length)
        string(SUBSTRING "${_remaining_artifact}"
            ${_segment_prefix_length} -1 _segment_tail)
        string(FIND "${_segment_tail}" ">" _segment_end)
        if(_segment_end EQUAL -1)
            message(FATAL_ERROR
                "build-receipt link-library artifact has an unterminated "
                "config-only segment: ${artifact}")
        endif()
        string(SUBSTRING "${_segment_tail}" 0 ${_segment_end}
            _configuration_artifact)
        if("${_configuration_artifact}" MATCHES "[<>]")
            message(FATAL_ERROR
                "build-receipt config-only link-library artifact contains "
                "nested or unsafe generator-expression syntax: ${artifact}")
        endif()
        _deac_build_receipt_normalize_link_library_artifact(
            "${_configuration_artifact}" _configuration_artifact)
        list(APPEND _configuration_artifacts
            "${_configuration_artifact}")
        string(APPEND _normalized_artifact
            "$<$<CONFIG:${_configuration}>:${_configuration_artifact}>")
        math(EXPR _next_segment "${_segment_end} + 1")
        string(SUBSTRING "${_segment_tail}" ${_next_segment} -1
            _remaining_artifact)
    endforeach()
    if(NOT _remaining_artifact STREQUAL "")
        message(FATAL_ERROR
            "build-receipt link-library artifact has trailing or unsupported "
            "generator-expression content: ${artifact}")
    endif()
    set(_unique_configuration_artifacts ${_configuration_artifacts})
    list(REMOVE_DUPLICATES _unique_configuration_artifacts)
    list(LENGTH _unique_configuration_artifacts
        _unique_configuration_artifact_count)
    if(_unique_configuration_artifact_count LESS 2)
        message(FATAL_ERROR
            "build-receipt config-only link-library artifact is noncanonical; "
            "use its common absolute path directly")
    endif()
    set(${output} "${_normalized_artifact}" PARENT_SCOPE)
    if(NOT _selected_artifacts_output STREQUAL "")
        set(${_selected_artifacts_output}
            "${_configuration_artifacts}" PARENT_SCOPE)
    endif()
endfunction()

function(_deac_build_receipt_require_posix_shell_literal value label)
    # This helper is also called from cmake -P, where no cmake_minimum_required
    # policy initialization is guaranteed.
    if(POLICY CMP0054)
        cmake_policy(SET CMP0054 NEW)
    endif()
    set(_allowed_placeholders ${ARGN})
    foreach(_placeholder IN LISTS _allowed_placeholders)
        if(NOT _placeholder MATCHES "^[A-Z][A-Z0-9_]*$")
            message(FATAL_ERROR
                "build-receipt ${label} has an invalid allowed placeholder")
        endif()
    endforeach()
    list(REMOVE_DUPLICATES _allowed_placeholders)

    # Recipe text is security-sensitive input.  Reject every control byte;
    # rule templates and File API fragments have no need for one, including
    # horizontal tabs and embedded line breaks.
    string(HEX "${value}" _value_hex)
    string(LENGTH "${_value_hex}" _value_hex_length)
    set(_hex_index 0)
    while(_hex_index LESS _value_hex_length)
        string(SUBSTRING "${_value_hex}" ${_hex_index} 2 _byte_hex)
        if(_byte_hex MATCHES "^(0[0-9a-f]|1[0-9a-f]|7f)$")
            message(FATAL_ERROR
                "build-receipt ${label} contains a POSIX shell control byte")
        endif()
        math(EXPR _hex_index "${_hex_index} + 2")
    endwhile()

    string(LENGTH "${value}" _length)
    set(_index 0)
    set(_quote "")
    set(_escaped false)
    while(_index LESS _length)
        string(SUBSTRING "${value}" ${_index} 1 _character)
        math(EXPR _next_index "${_index} + 1")
        set(_next_character "")
        if(_next_index LESS _length)
            string(SUBSTRING "${value}" ${_next_index} 1 _next_character)
        endif()

        # A generator expression is evaluated before the recipe reaches the
        # shell, even when it is written inside POSIX single quotes.
        if(_character STREQUAL "$" AND _next_character STREQUAL "<")
            message(FATAL_ERROR
                "build-receipt ${label} contains unsupported "
                "generator-expression syntax")
        endif()
        # Dollar expansion happens in Make/Ninja before POSIX shell quoting is
        # applied, so even a single-quoted or backslash-protected dollar is not
        # stable across the supported generators.  Backtick command
        # substitution is supported only when POSIX single quotes protect it;
        # a generator-preserved backslash form is outside this schema.  CMake
        # likewise consumes an unquoted backslash-semicolon layer while
        # materializing rule templates.  Reject these shapes instead of
        # attributing recipe text that differs from the command eventually
        # executed.
        if(_character STREQUAL "$")
            message(FATAL_ERROR
                "build-receipt ${label} contains unsafe POSIX shell syntax")
        endif()
        if(_character STREQUAL "`" AND NOT _quote STREQUAL "'")
            message(FATAL_ERROR
                "build-receipt ${label} contains unsafe POSIX shell syntax")
        endif()
        if(_character STREQUAL ";" AND _quote STREQUAL "")
            message(FATAL_ERROR
                "build-receipt ${label} contains unsafe POSIX shell syntax")
        endif()

        if(_escaped)
            set(_escaped false)
        elseif(_character STREQUAL "<")
            string(SUBSTRING "${value}" ${_index} -1 _placeholder_tail)
            string(FIND "${_placeholder_tail}" ">" _placeholder_end)
            set(_recognized_placeholder false)
            if(NOT _placeholder_end EQUAL -1)
                math(EXPR _placeholder_length "${_placeholder_end} + 1")
                math(EXPR _placeholder_name_length
                    "${_placeholder_end} - 1")
                string(SUBSTRING "${_placeholder_tail}" 0
                    ${_placeholder_length} _placeholder_token)
                string(SUBSTRING "${_placeholder_tail}" 1
                    ${_placeholder_name_length} _placeholder_name)
                if(_placeholder_name MATCHES "^[A-Z][A-Z0-9_]*$")
                    list(FIND _allowed_placeholders "${_placeholder_name}"
                        _placeholder_index)
                    if(_placeholder_index EQUAL -1)
                        message(FATAL_ERROR
                            "build-receipt ${label} uses unsupported "
                            "placeholder ${_placeholder_token}")
                    endif()
                    set(_placeholder_has_left_boundary false)
                    if(_index EQUAL 0)
                        set(_placeholder_has_left_boundary true)
                    else()
                        math(EXPR _previous_index "${_index} - 1")
                        string(SUBSTRING "${value}" ${_previous_index} 1
                            _previous_character)
                        if(_previous_character STREQUAL " ")
                            set(_placeholder_has_left_boundary true)
                        endif()
                    endif()
                    math(EXPR _after_placeholder
                        "${_index} + ${_placeholder_length}")
                    set(_placeholder_has_right_boundary false)
                    if(_after_placeholder EQUAL _length)
                        set(_placeholder_has_right_boundary true)
                    elseif(_after_placeholder LESS _length)
                        string(SUBSTRING "${value}" ${_after_placeholder} 1
                            _after_placeholder_character)
                        if(_after_placeholder_character STREQUAL " ")
                            set(_placeholder_has_right_boundary true)
                        endif()
                    endif()
                    if(NOT _quote STREQUAL "" OR
                            NOT _placeholder_has_left_boundary OR
                            NOT _placeholder_has_right_boundary)
                        message(FATAL_ERROR
                            "build-receipt ${label} uses placeholder "
                            "${_placeholder_token} outside a standalone "
                            "unquoted shell word")
                    endif()
                    set(_recognized_placeholder true)
                    math(EXPR _index
                        "${_index} + ${_placeholder_length} - 1")
                endif()
            endif()
            if(NOT _recognized_placeholder AND _quote STREQUAL "")
                message(FATAL_ERROR
                    "build-receipt ${label} contains unsafe POSIX shell "
                    "syntax")
            endif()
        elseif(_quote STREQUAL "'")
            if(_character STREQUAL "'")
                set(_quote "")
            endif()
        elseif(_quote STREQUAL "\"")
            if(_character STREQUAL "\\")
                set(_escaped true)
            elseif(_character STREQUAL "\"")
                set(_quote "")
            elseif(_character STREQUAL "$" OR _character STREQUAL "`")
                message(FATAL_ERROR
                    "build-receipt ${label} contains unsafe POSIX shell "
                    "syntax")
            endif()
        elseif(_character STREQUAL "\\")
            set(_escaped true)
        elseif(_character STREQUAL "'" OR _character STREQUAL "\"")
            set(_quote "${_character}")
        elseif(_character STREQUAL "$" OR _character STREQUAL "`" OR
                _character STREQUAL "*" OR _character STREQUAL "?" OR
                _character STREQUAL "[" OR _character STREQUAL "]" OR
                _character STREQUAL ">" OR _character STREQUAL "~" OR
                _character STREQUAL "#" OR _character STREQUAL ";" OR
                _character STREQUAL "|" OR _character STREQUAL "&" OR
                _character STREQUAL "(" OR _character STREQUAL ")" OR
                _character STREQUAL "{" OR _character STREQUAL "}")
            message(FATAL_ERROR
                "build-receipt ${label} contains unsafe POSIX shell syntax")
        endif()
        math(EXPR _index "${_index} + 1")
    endwhile()
    if(_escaped OR NOT _quote STREQUAL "")
        message(FATAL_ERROR
            "build-receipt ${label} has incomplete POSIX shell quoting")
    endif()
endfunction()

function(_deac_build_receipt_snapshot_named_tool
        variable output_arguments output_fingerprint_material
        output_path output_real_path output_sha256)
    if(NOT DEFINED ${variable} OR "${${variable}}" STREQUAL "")
        message(FATAL_ERROR "build receipt requires ${variable}")
    endif()
    _deac_build_receipt_require_plain(
        "${${variable}}" "${variable} executable identity")
    if(NOT IS_ABSOLUTE "${${variable}}")
        message(FATAL_ERROR
            "build-receipt ${variable} must be an absolute executable path")
    endif()
    get_filename_component(_tool_real "${${variable}}" REALPATH)
    if(NOT EXISTS "${_tool_real}" OR IS_DIRECTORY "${_tool_real}")
        message(FATAL_ERROR
            "build-receipt ${variable} is not a regular file: ${${variable}}")
    endif()
    file(SHA256 "${_tool_real}" _tool_sha256)
    _deac_build_receipt_require_plain(
        "${_tool_real}" "${variable} executable identity")

    set(_arguments
        "-DDEAC_BUILD_RECEIPT_CONFIGURED_${variable}_PATH:FILEPATH=${${variable}}"
        "-DDEAC_BUILD_RECEIPT_CONFIGURED_${variable}_REAL_PATH:FILEPATH=${_tool_real}"
        "-DDEAC_BUILD_RECEIPT_CONFIGURED_${variable}_SHA256:STRING=${_tool_sha256}")
    set(_fingerprint_material "")
    foreach(_field PATH REAL_PATH SHA256)
        if(_field STREQUAL "PATH")
            set(_value "${${variable}}")
        elseif(_field STREQUAL "REAL_PATH")
            set(_value "${_tool_real}")
        else()
            set(_value "${_tool_sha256}")
        endif()
        string(LENGTH "${_value}" _value_length)
        string(APPEND _fingerprint_material
            "${variable}.${_field}:${_value_length}:${_value}\n")
    endforeach()
    set(${output_arguments} "${_arguments}" PARENT_SCOPE)
    set(${output_fingerprint_material}
        "${_fingerprint_material}" PARENT_SCOPE)
    set(${output_path} "${${variable}}" PARENT_SCOPE)
    set(${output_real_path} "${_tool_real}" PARENT_SCOPE)
    set(${output_sha256} "${_tool_sha256}" PARENT_SCOPE)
endfunction()

function(_deac_build_receipt_reject_unsupported_languages)
    foreach(_unsupported_language CUDA HIP)
        if(_unsupported_language IN_LIST ARGN)
            message(FATAL_ERROR
                "build receipts fail closed for the native CMake "
                "${_unsupported_language} language until its intermediate "
                "device-link rules are represented and compiler-gated")
        endif()
    endforeach()
endfunction()

function(_deac_build_receipt_query_enabled_languages output)
    get_property(_enabled_languages GLOBAL PROPERTY ENABLED_LANGUAGES)
    set(${output} "${_enabled_languages}" PARENT_SCOPE)
endfunction()

function(_deac_build_receipt_configuration_snapshot
        directory output_generator output_mode output_configurations)
    get_directory_property(_generator DIRECTORY "${directory}"
        DEFINITION CMAKE_GENERATOR)
    get_property(_is_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
    if(_is_multi_config)
        set(_mode "multi")
        set(_active_configuration_variable CMAKE_CONFIGURATION_TYPES)
        get_directory_property(_configurations DIRECTORY "${directory}"
            DEFINITION CMAKE_CONFIGURATION_TYPES)
        if("${_configurations}" STREQUAL "")
            message(FATAL_ERROR
                "build-receipt multi-config generator has no configurations")
        endif()
    else()
        set(_mode "single")
        set(_active_configuration_variable CMAKE_BUILD_TYPE)
        get_directory_property(_configurations DIRECTORY "${directory}"
            DEFINITION CMAKE_BUILD_TYPE)
    endif()
    foreach(_state_variable CMAKE_GENERATOR ${_active_configuration_variable})
        if(_state_variable STREQUAL "CMAKE_GENERATOR")
            set(_effective_value "${_generator}")
        else()
            set(_effective_value "${_configurations}")
        endif()
        get_property(_cache_value_is_set CACHE "${_state_variable}"
            PROPERTY VALUE SET)
        if(_cache_value_is_set)
            get_property(_cache_value CACHE "${_state_variable}"
                PROPERTY VALUE)
            if(NOT "${_cache_value}" STREQUAL "${_effective_value}")
                message(FATAL_ERROR
                    "build-receipt ${_state_variable} cache value disagrees "
                    "with effective directory state: ${directory}")
            endif()
        endif()
    endforeach()
    list(LENGTH _configurations _configuration_count)
    if(_configuration_count EQUAL 0)
        message(FATAL_ERROR
            "build-receipt requires at least one configured build type")
    endif()
    if(_mode STREQUAL "single" AND NOT _configuration_count EQUAL 1)
        message(FATAL_ERROR
            "build-receipt single-config generator requires exactly one "
            "configured build type")
    endif()
    set(_folded_configurations)
    foreach(_configuration IN LISTS _configurations)
        if(NOT _configuration MATCHES "^[A-Za-z0-9_.+-]+$" OR
                _configuration STREQUAL "." OR
                _configuration STREQUAL "..")
            message(FATAL_ERROR
                "build-receipt configuration is not path-safe: "
                "${_configuration}")
        endif()
        string(TOUPPER "${_configuration}" _folded_configuration)
        if(_folded_configuration IN_LIST _folded_configurations)
            message(FATAL_ERROR
                "build-receipt configurations must be unique ignoring "
                "ASCII case: ${_configuration}")
        endif()
        list(APPEND _folded_configurations "${_folded_configuration}")
    endforeach()
    set(${output_generator} "${_generator}" PARENT_SCOPE)
    set(${output_mode} "${_mode}" PARENT_SCOPE)
    set(${output_configurations} "${_configurations}" PARENT_SCOPE)
endfunction()

function(_deac_build_receipt_snapshot_tool
        language output_arguments output_fingerprint_material
        output_path output_real_path output_sha256)
    if(NOT DEFINED CMAKE_${language}_COMPILER OR
            "${CMAKE_${language}_COMPILER}" STREQUAL "")
        message(FATAL_ERROR
            "build receipt requires CMAKE_${language}_COMPILER")
    endif()
    get_filename_component(
        _compiler_real "${CMAKE_${language}_COMPILER}" REALPATH)
    if(NOT EXISTS "${_compiler_real}" OR IS_DIRECTORY "${_compiler_real}")
        message(FATAL_ERROR
            "build receipt compiler is not a regular file: "
            "${CMAKE_${language}_COMPILER}")
    endif()
    file(SHA256 "${_compiler_real}" _compiler_sha256)
    foreach(_value
            "${CMAKE_${language}_COMPILER}"
            "${_compiler_real}"
            "${CMAKE_${language}_COMPILER_ID}"
            "${CMAKE_${language}_COMPILER_VERSION}")
        _deac_build_receipt_require_plain(
            "${_value}" "${language} compiler identity")
    endforeach()
    set(_arguments
        "-DDEAC_BUILD_RECEIPT_CONFIGURED_${language}_PATH:FILEPATH=${CMAKE_${language}_COMPILER}"
        "-DDEAC_BUILD_RECEIPT_CONFIGURED_${language}_REAL_PATH:FILEPATH=${_compiler_real}"
        "-DDEAC_BUILD_RECEIPT_CONFIGURED_${language}_SHA256:STRING=${_compiler_sha256}"
        "-DDEAC_BUILD_RECEIPT_CONFIGURED_${language}_ID:STRING=${CMAKE_${language}_COMPILER_ID}"
        "-DDEAC_BUILD_RECEIPT_CONFIGURED_${language}_VERSION:STRING=${CMAKE_${language}_COMPILER_VERSION}")
    if(DEFINED CMAKE_${language}_COMPILER_TARGET)
        _deac_build_receipt_require_plain(
            "${CMAKE_${language}_COMPILER_TARGET}"
            "${language} compiler target")
        list(APPEND _arguments
            "-DDEAC_BUILD_RECEIPT_CONFIGURED_${language}_TARGET:STRING=${CMAKE_${language}_COMPILER_TARGET}")
    endif()
    set(_fingerprint_material "")
    foreach(_field PATH REAL_PATH SHA256 ID VERSION TARGET)
        if(_field STREQUAL "PATH")
            set(_value "${CMAKE_${language}_COMPILER}")
        elseif(_field STREQUAL "REAL_PATH")
            set(_value "${_compiler_real}")
        elseif(_field STREQUAL "SHA256")
            set(_value "${_compiler_sha256}")
        elseif(_field STREQUAL "ID")
            set(_value "${CMAKE_${language}_COMPILER_ID}")
        elseif(_field STREQUAL "VERSION")
            set(_value "${CMAKE_${language}_COMPILER_VERSION}")
        elseif(DEFINED CMAKE_${language}_COMPILER_TARGET)
            set(_value "${CMAKE_${language}_COMPILER_TARGET}")
        else()
            set(_value "")
        endif()
        string(LENGTH "${_value}" _length)
        string(APPEND _fingerprint_material
            "${language}.${_field}:${_length}:${_value}\n")
    endforeach()
    set(${output_arguments} "${_arguments}" PARENT_SCOPE)
    set(${output_fingerprint_material}
        "${_fingerprint_material}" PARENT_SCOPE)
    set(${output_path} "${CMAKE_${language}_COMPILER}" PARENT_SCOPE)
    set(${output_real_path} "${_compiler_real}" PARENT_SCOPE)
    set(${output_sha256} "${_compiler_sha256}" PARENT_SCOPE)
endfunction()

function(_deac_build_receipt_reject_launchers target_name directory)
    foreach(_language CXX CUDA)
        foreach(_suffix COMPILER_LAUNCHER LINKER_LAUNCHER)
            set(_variable "CMAKE_${_language}_${_suffix}")
            get_directory_property(_launcher DIRECTORY "${directory}"
                DEFINITION "${_variable}")
            if(NOT "${_launcher}" STREQUAL "")
                message(FATAL_ERROR
                    "build receipts do not yet support ${_variable}")
            endif()
            get_target_property(_launcher "${target_name}"
                "${_language}_${_suffix}")
            if(NOT "${_launcher}" STREQUAL "" AND
                    NOT "${_launcher}" MATCHES "-NOTFOUND$")
                message(FATAL_ERROR
                    "build receipts do not yet support target "
                    "${target_name} ${_language}_${_suffix}")
            endif()
        endforeach()
        foreach(_suffix
                CLANG_TIDY CPPCHECK CPPLINT INCLUDE_WHAT_YOU_USE)
            get_target_property(_launcher "${target_name}"
                "${_language}_${_suffix}")
            if(NOT "${_launcher}" STREQUAL "" AND
                    NOT "${_launcher}" MATCHES "-NOTFOUND$")
                message(FATAL_ERROR
                    "build receipts do not yet support target "
                    "${target_name} ${_language}_${_suffix}")
            endif()
        endforeach()
        set(_compiler_arg1 "CMAKE_${_language}_COMPILER_ARG1")
        get_directory_property(_compiler_arg1_value
            DIRECTORY "${directory}" DEFINITION "${_compiler_arg1}")
        if(NOT "${_compiler_arg1_value}" STREQUAL "")
            message(FATAL_ERROR
                "build receipts do not yet support ${_compiler_arg1}")
        endif()
    endforeach()

    get_target_property(_link_what_you_use "${target_name}" LINK_WHAT_YOU_USE)
    if(_link_what_you_use AND NOT _link_what_you_use MATCHES "-NOTFOUND$")
        message(FATAL_ERROR
            "build receipts do not yet support target ${target_name} "
            "LINK_WHAT_YOU_USE")
    endif()

    set(_ipo_properties INTERPROCEDURAL_OPTIMIZATION)
    get_directory_property(_configuration_types DIRECTORY "${directory}"
        DEFINITION CMAKE_CONFIGURATION_TYPES)
    get_directory_property(_build_type DIRECTORY "${directory}"
        DEFINITION CMAKE_BUILD_TYPE)
    if(NOT "${_configuration_types}" STREQUAL "")
        set(_ipo_configurations ${_configuration_types})
    else()
        set(_ipo_configurations "${_build_type}")
    endif()
    foreach(_configuration IN LISTS _ipo_configurations)
        string(TOUPPER "${_configuration}" _configuration_upper)
        list(APPEND _ipo_properties
            "INTERPROCEDURAL_OPTIMIZATION_${_configuration_upper}")
    endforeach()
    list(REMOVE_DUPLICATES _ipo_properties)
    foreach(_ipo_property IN LISTS _ipo_properties)
        get_target_property(
            _interprocedural_optimization
            "${target_name}" "${_ipo_property}")
        if(_interprocedural_optimization AND
                NOT _interprocedural_optimization MATCHES "-NOTFOUND$")
            message(FATAL_ERROR
                "build receipts do not yet support target ${target_name} "
                "${_ipo_property}")
        endif()
    endforeach()

    foreach(_launcher_property
            RULE_LAUNCH_COMPILE RULE_LAUNCH_CUSTOM RULE_LAUNCH_LINK)
        get_target_property(
            _launcher_value "${target_name}" "${_launcher_property}")
        if(NOT "${_launcher_value}" STREQUAL "" AND
                NOT "${_launcher_value}" MATCHES "-NOTFOUND$")
            message(FATAL_ERROR
                "build receipts do not yet support target ${target_name} "
                "${_launcher_property}")
        endif()
        get_property(_launcher_value GLOBAL
            PROPERTY "${_launcher_property}")
        if(NOT "${_launcher_value}" STREQUAL "")
            message(FATAL_ERROR
                "build receipts do not yet support GLOBAL "
                "${_launcher_property}")
        endif()
        get_property(_launcher_value DIRECTORY "${directory}"
            PROPERTY "${_launcher_property}")
        if(NOT "${_launcher_value}" STREQUAL "")
            message(FATAL_ERROR
                "build receipts do not yet support DIRECTORY "
                "${_launcher_property}")
        endif()
    endforeach()
endfunction()

function(_deac_build_receipt_reject_rule_override_routes directory)
    foreach(_variable
            CMAKE_TOOLCHAIN_FILE
            CMAKE_USER_MAKE_RULES_OVERRIDE
            CMAKE_USER_MAKE_RULES_OVERRIDE_CXX
            CMAKE_USER_MAKE_RULES_OVERRIDE_CUDA
            CMAKE_USER_MAKE_RULES_OVERRIDE_HIP
            CMAKE_PROJECT_INCLUDE
            CMAKE_PROJECT_INCLUDE_BEFORE
            CMAKE_PROJECT_TOP_LEVEL_INCLUDES)
        get_directory_property(_variable_value DIRECTORY "${directory}"
            DEFINITION "${_variable}")
        if(NOT "${_variable_value}" STREQUAL "")
            message(FATAL_ERROR
                "build receipts do not support rule-override route ${_variable}")
        endif()
    endforeach()
    get_property(_module_path_is_cached
        CACHE CMAKE_MODULE_PATH PROPERTY TYPE SET)
    if(_module_path_is_cached)
        message(FATAL_ERROR
            "build receipts reject cached rule-override route CMAKE_MODULE_PATH")
    endif()
    get_directory_property(_project_name DIRECTORY "${directory}"
        DEFINITION PROJECT_NAME)
    foreach(_suffix INCLUDE INCLUDE_BEFORE)
        set(_variable "CMAKE_PROJECT_${_project_name}_${_suffix}")
        get_directory_property(_variable_value DIRECTORY "${directory}"
            DEFINITION "${_variable}")
        if(NOT "${_variable_value}" STREQUAL "")
            message(FATAL_ERROR
                "build receipts do not support rule-override route ${_variable}")
        endif()
    endforeach()

    foreach(_language CXX CUDA HIP)
        foreach(_rule
                COMPILE_OBJECT
                LINK_EXECUTABLE
                CREATE_STATIC_LIBRARY
                ARCHIVE_CREATE
                ARCHIVE_APPEND
                ARCHIVE_FINISH)
            set(_variable "CMAKE_${_language}_${_rule}")
            get_property(_is_cached CACHE "${_variable}" PROPERTY TYPE SET)
            if(_is_cached)
                message(FATAL_ERROR
                    "build receipts reject cached rule-template override "
                    "${_variable}")
            endif()
        endforeach()
    endforeach()
    foreach(_rule
            DEVICE_LINK_COMPILE DEVICE_LINK_EXECUTABLE DEVICE_LINK_LIBRARY)
        set(_variable "CMAKE_CUDA_${_rule}")
        get_property(_is_cached CACHE "${_variable}" PROPERTY TYPE SET)
        if(_is_cached)
            message(FATAL_ERROR
                "build receipts reject cached rule-template override "
                "${_variable}")
        endif()
    endforeach()
endfunction()

function(_deac_build_receipt_rule_fingerprint
        language directory output_fingerprint_material)
    if(NOT language STREQUAL "CXX")
        message(FATAL_ERROR
            "schema-1 build receipts only support CXX rule templates")
    endif()
    set(_material "")
    foreach(_rule COMPILE_OBJECT LINK_EXECUTABLE)
        set(_variable "CMAKE_${language}_${_rule}")
        get_directory_property(_template DIRECTORY "${directory}"
            DEFINITION "${_variable}")
        if("${_template}" STREQUAL "")
            message(FATAL_ERROR "build receipt requires ${_variable}")
        endif()
        if(_rule STREQUAL "COMPILE_OBJECT")
            set(_allowed_placeholders
                CMAKE_CXX_COMPILER DEFINES INCLUDES FLAGS OBJECT SOURCE)
            set(_required_placeholders
                DEFINES INCLUDES FLAGS OBJECT SOURCE)
        else()
            set(_allowed_placeholders
                CMAKE_CXX_COMPILER FLAGS CMAKE_CXX_LINK_FLAGS LINK_FLAGS
                OBJECTS TARGET LINK_LIBRARIES)
            set(_required_placeholders
                FLAGS CMAKE_CXX_LINK_FLAGS LINK_FLAGS OBJECTS TARGET
                LINK_LIBRARIES)
        endif()
        _deac_build_receipt_require_posix_shell_literal(
            "${_template}" "${_variable}" ${_allowed_placeholders})
        set(_allowed_prefix "<CMAKE_CXX_COMPILER>")
        string(FIND "${_template}" "${_allowed_prefix}" _prefix_position)
        if(NOT _prefix_position EQUAL 0)
            message(FATAL_ERROR
                "build-receipt ${_variable} has an unsupported launcher")
        endif()
        foreach(_placeholder IN LISTS _required_placeholders)
            if(NOT _template MATCHES "<${_placeholder}>")
                message(FATAL_ERROR
                    "build-receipt ${_variable} omits <${_placeholder}>")
            endif()
        endforeach()
        string(LENGTH "${_template}" _length)
        string(APPEND _material
            "${language}.${_rule}:${_length}:${_template}\n")
    endforeach()

    set(_legacy_archive "CMAKE_${language}_CREATE_STATIC_LIBRARY")
    get_directory_property(_legacy_archive_value DIRECTORY "${directory}"
        DEFINITION "${_legacy_archive}")
    if(NOT "${_legacy_archive_value}" STREQUAL "")
        message(FATAL_ERROR
            "build receipts do not support ${_legacy_archive}")
    endif()
    foreach(_rule ARCHIVE_CREATE ARCHIVE_APPEND ARCHIVE_FINISH)
        set(_variable "CMAKE_${language}_${_rule}")
        get_directory_property(_template DIRECTORY "${directory}"
            DEFINITION "${_variable}")
        if("${_template}" STREQUAL "")
            message(FATAL_ERROR "build receipt requires ${_variable}")
        endif()
        if(_rule STREQUAL "ARCHIVE_FINISH")
            set(_allowed_prefix "<CMAKE_RANLIB>")
            set(_allowed_placeholders CMAKE_RANLIB TARGET)
            set(_required_placeholders TARGET)
        else()
            set(_allowed_prefix "<CMAKE_AR>")
            set(_allowed_placeholders CMAKE_AR LINK_FLAGS OBJECTS TARGET)
            set(_required_placeholders LINK_FLAGS OBJECTS TARGET)
        endif()
        _deac_build_receipt_require_posix_shell_literal(
            "${_template}" "${_variable}" ${_allowed_placeholders})
        string(FIND "${_template}" "${_allowed_prefix}" _prefix_position)
        if(NOT _prefix_position EQUAL 0)
            message(FATAL_ERROR
                "build-receipt ${_variable} has an unsupported launcher")
        endif()
        foreach(_placeholder IN LISTS _required_placeholders)
            if(NOT _template MATCHES "<${_placeholder}>")
                message(FATAL_ERROR
                    "build-receipt ${_variable} omits <${_placeholder}>")
            endif()
        endforeach()
        string(LENGTH "${_template}" _length)
        string(APPEND _material
            "${language}.${_rule}:${_length}:${_template}\n")
    endforeach()
    set(${output_fingerprint_material} "${_material}" PARENT_SCOPE)
endfunction()

function(_deac_build_receipt_canonical_target target_name output)
    if(NOT TARGET "${target_name}")
        set(${output} "" PARENT_SCOPE)
        return()
    endif()
    get_target_property(_aliased_target "${target_name}" ALIASED_TARGET)
    if("${_aliased_target}" STREQUAL "" OR
            "${_aliased_target}" STREQUAL "_aliased_target-NOTFOUND")
        set(_aliased_target "${target_name}")
    endif()
    set(${output} "${_aliased_target}" PARENT_SCOPE)
endfunction()

function(_deac_build_receipt_target_directory target_name output)
    if(NOT TARGET "${target_name}")
        message(FATAL_ERROR
            "build-receipt validated target no longer exists: ${target_name}")
    endif()
    get_target_property(_target_directory "${target_name}" SOURCE_DIR)
    if("${_target_directory}" STREQUAL "" OR
            "${_target_directory}" STREQUAL "_target_directory-NOTFOUND" OR
            NOT IS_ABSOLUTE "${_target_directory}")
        message(FATAL_ERROR
            "build-receipt target has no absolute SOURCE_DIR: ${target_name}")
    endif()
    set(${output} "${_target_directory}" PARENT_SCOPE)
endfunction()

function(_deac_build_receipt_validate_target_scopes
        receipt_target registration_directory)
    set(_validated_targets ${ARGN})
    _deac_build_receipt_reject_rule_override_routes(
        "${registration_directory}")
    _deac_build_receipt_reject_launchers(
        "${receipt_target}" "${registration_directory}")
    foreach(_validated_target IN LISTS _validated_targets)
        _deac_build_receipt_target_directory(
            "${_validated_target}" _target_directory)
        _deac_build_receipt_reject_rule_override_routes(
            "${_target_directory}")
        _deac_build_receipt_reject_launchers(
            "${_validated_target}" "${_target_directory}")
    endforeach()
endfunction()

function(_deac_build_receipt_target_rule_fingerprint
        language registration_directory output_fingerprint_material)
    set(_validated_targets ${ARGN})
    list(SORT _validated_targets)
    _deac_build_receipt_rule_fingerprint(
        "${language}" "${registration_directory}" _registration_rule_material)
    set(_material
        "deac-build-target-rules-v2\n${_registration_rule_material}")
    set(_rule_target_types
        EXECUTABLE STATIC_LIBRARY SHARED_LIBRARY MODULE_LIBRARY OBJECT_LIBRARY)
    foreach(_validated_target IN LISTS _validated_targets)
        get_target_property(_validated_target_type
            "${_validated_target}" TYPE)
        if(_validated_target_type IN_LIST _rule_target_types)
            _deac_build_receipt_target_directory(
                "${_validated_target}" _target_directory)
            _deac_build_receipt_rule_fingerprint(
                "${language}" "${_target_directory}" _target_rule_material)
            if(NOT _target_rule_material STREQUAL
                    _registration_rule_material)
                message(FATAL_ERROR
                    "build-receipt ${language} rule templates for target "
                    "${_validated_target} differ from the receipt "
                    "registration directory")
            endif()
        endif()
    endforeach()
    set(${output_fingerprint_material} "${_material}" PARENT_SCOPE)
endfunction()

function(_deac_build_receipt_validate_named_tool_in_directory
        variable directory expected_path expected_real_path expected_sha256)
    get_directory_property(_tool_path DIRECTORY "${directory}"
        DEFINITION "${variable}")
    _deac_build_receipt_require_plain(
        "${_tool_path}" "${variable} executable identity")
    if(NOT "${_tool_path}" STREQUAL "${expected_path}" OR
            NOT IS_ABSOLUTE "${_tool_path}")
        message(FATAL_ERROR
            "build-receipt ${variable} changed after registration")
    endif()
    get_filename_component(_tool_real_path "${_tool_path}" REALPATH)
    if(NOT EXISTS "${_tool_real_path}" OR IS_DIRECTORY "${_tool_real_path}")
        message(FATAL_ERROR
            "build-receipt ${variable} is no longer a regular file")
    endif()
    file(SHA256 "${_tool_real_path}" _tool_sha256)
    if(NOT "${_tool_real_path}" STREQUAL "${expected_real_path}" OR
            NOT "${_tool_sha256}" STREQUAL "${expected_sha256}")
        message(FATAL_ERROR
            "build-receipt ${variable} changed after registration")
    endif()
endfunction()

function(_deac_build_receipt_validate_interface_dependencies target_name)
    get_target_property(_expected_interface_targets "${target_name}"
        DEAC_BUILD_RECEIPT_INTERFACE_DEPENDENCIES)
    if("${_expected_interface_targets}" STREQUAL
            "_expected_interface_targets-NOTFOUND")
        message(FATAL_ERROR
            "build-receipt interface dependency seal is missing for "
            "${target_name}")
    endif()

    get_target_property(_direct_links "${target_name}" LINK_LIBRARIES)
    if("${_direct_links}" STREQUAL "_direct_links-NOTFOUND")
        set(_direct_links)
    endif()
    set(_observed_interface_targets)
    foreach(_direct_link IN LISTS _direct_links)
        set(_linked_target "")
        if(TARGET "${_direct_link}")
            _deac_build_receipt_canonical_target(
                "${_direct_link}" _linked_target)
        elseif(_direct_link MATCHES "^\\$<LINK_ONLY:([^$<>]+)>$")
            set(_link_only_item "${CMAKE_MATCH_1}")
            if(TARGET "${_link_only_item}")
                _deac_build_receipt_canonical_target(
                    "${_link_only_item}" _linked_target)
            endif()
        elseif(_direct_link MATCHES "\\$<")
            message(FATAL_ERROR
                "has an unsupported generator-expression link item for "
                "build-receipt target ${target_name}")
        endif()
        if(NOT "${_linked_target}" STREQUAL "")
            get_target_property(_linked_target_type
                "${_linked_target}" TYPE)
            if(_linked_target_type STREQUAL "INTERFACE_LIBRARY")
                list(APPEND _observed_interface_targets "${_linked_target}")
            endif()
        endif()
    endforeach()
    list(SORT _observed_interface_targets)
    if(NOT "${_observed_interface_targets}" STREQUAL
            "${_expected_interface_targets}")
        message(FATAL_ERROR
            "build-receipt interface dependencies changed after "
            "registration for ${target_name}: expected "
            "[${_expected_interface_targets}], got "
            "[${_observed_interface_targets}]")
    endif()
endfunction()

function(_deac_build_receipt_validate_interface_link_items
        target_name property)
    get_target_property(_link_items "${target_name}" "${property}")
    if("${_link_items}" STREQUAL "_link_items-NOTFOUND")
        return()
    endif()
    foreach(_link_item IN LISTS _link_items)
        if(_link_item MATCHES "\\$<" AND
                NOT _link_item MATCHES "^\\$<LINK_ONLY:[^$<>]+>$")
            message(FATAL_ERROR
                "build-receipt interface target ${target_name} property "
                "${property} has an unsupported generator-expression "
                "link item")
        endif()
    endforeach()
endfunction()

function(_deac_build_receipt_interface_fingerprint output)
    set(_interface_targets ${ARGN})
    list(SORT _interface_targets)
    set(_material "deac-build-receipt-interface-v1\n")
    set(_interface_properties
        INTERFACE_COMPILE_DEFINITIONS
        INTERFACE_COMPILE_FEATURES
        INTERFACE_COMPILE_OPTIONS
        INTERFACE_INCLUDE_DIRECTORIES
        INTERFACE_LINK_DEPENDS
        INTERFACE_LINK_DIRECTORIES
        INTERFACE_LINK_LIBRARIES
        INTERFACE_LINK_LIBRARIES_DIRECT
        INTERFACE_LINK_LIBRARIES_DIRECT_EXCLUDE
        INTERFACE_LINK_OPTIONS
        INTERFACE_POSITION_INDEPENDENT_CODE
        INTERFACE_PRECOMPILE_HEADERS
        INTERFACE_SOURCES
        INTERFACE_SYSTEM_INCLUDE_DIRECTORIES)
    foreach(_interface_target IN LISTS _interface_targets)
        if(NOT TARGET "${_interface_target}")
            message(FATAL_ERROR
                "build-receipt sealed interface target no longer exists: "
                "${_interface_target}")
        endif()
        get_target_property(_interface_type "${_interface_target}" TYPE)
        if(NOT _interface_type STREQUAL "INTERFACE_LIBRARY")
            message(FATAL_ERROR
                "build-receipt sealed interface target changed type: "
                "${_interface_target}")
        endif()
        string(LENGTH "${_interface_target}" _target_length)
        string(APPEND _material
            "target:${_target_length}:${_interface_target}\n")
        foreach(_link_property
                INTERFACE_LINK_LIBRARIES
                INTERFACE_LINK_LIBRARIES_DIRECT
                INTERFACE_LINK_LIBRARIES_DIRECT_EXCLUDE)
            _deac_build_receipt_validate_interface_link_items(
                "${_interface_target}" "${_link_property}")
        endforeach()
        foreach(_property IN LISTS _interface_properties)
            get_target_property(_property_value
                "${_interface_target}" "${_property}")
            if("${_property_value}" STREQUAL
                    "_property_value-NOTFOUND")
                set(_property_value "")
            endif()
            string(LENGTH "${_property_value}" _property_length)
            string(APPEND _material
                "${_property}:${_property_length}:${_property_value}\n")
        endforeach()
    endforeach()
    string(SHA256 _fingerprint "${_material}")
    set(${output} "${_fingerprint}" PARENT_SCOPE)
endfunction()

function(_deac_build_receipt_validate_seal target_name)
    get_target_property(_registration_directory "${target_name}"
        DEAC_BUILD_RECEIPT_REGISTRATION_DIRECTORY)
    get_target_property(_validated_targets "${target_name}"
        DEAC_BUILD_RECEIPT_VALIDATED_TARGETS)
    get_target_property(_expected_rule_fingerprint "${target_name}"
        DEAC_BUILD_RECEIPT_CXX_RULE_FINGERPRINT)
    get_target_property(_expected_interface_fingerprint "${target_name}"
        DEAC_BUILD_RECEIPT_INTERFACE_FINGERPRINT)
    get_target_property(_build_root "${target_name}"
        DEAC_BUILD_RECEIPT_BUILD_ROOT)
    get_target_property(_output_paths "${target_name}"
        DEAC_BUILD_RECEIPT_OUTPUT_PATHS)
    get_target_property(_expected_generator "${target_name}"
        DEAC_BUILD_RECEIPT_GENERATOR)
    get_target_property(_expected_configuration_mode "${target_name}"
        DEAC_BUILD_RECEIPT_CONFIGURATION_MODE)
    get_target_property(_expected_configurations "${target_name}"
        DEAC_BUILD_RECEIPT_CONFIGURATIONS)
    get_target_property(_top_level_source_directory "${target_name}"
        DEAC_BUILD_RECEIPT_TOP_LEVEL_SOURCE_DIRECTORY)
    foreach(_seal_property_value
            _registration_directory
            _validated_targets
            _expected_rule_fingerprint
            _expected_interface_fingerprint
            _build_root
            _output_paths
            _expected_generator
            _expected_configuration_mode
            _expected_configurations
            _top_level_source_directory)
        if("${${_seal_property_value}}" STREQUAL
                "${_seal_property_value}-NOTFOUND")
            message(FATAL_ERROR
                "build-receipt validation seal is incomplete for "
                "${target_name}")
        endif()
    endforeach()

    set(_configuration_directories
        "${_top_level_source_directory}" "${_registration_directory}")
    list(REMOVE_DUPLICATES _configuration_directories)
    foreach(_configuration_directory IN LISTS _configuration_directories)
        _deac_build_receipt_configuration_snapshot(
            "${_configuration_directory}" _observed_generator
            _observed_configuration_mode _observed_configurations)
        if(NOT _observed_generator STREQUAL _expected_generator OR
                NOT _observed_configuration_mode STREQUAL
                    _expected_configuration_mode OR
                NOT "${_observed_configurations}" STREQUAL
                    "${_expected_configurations}")
            message(FATAL_ERROR
                "build-receipt generator configuration state changed after "
                "registration or differs between the top-level and receipt "
                "directories for "
                "${target_name}: ${_configuration_directory}")
        endif()
    endforeach()

    _deac_build_receipt_validate_target_scopes(
        "${target_name}" "${_registration_directory}"
        ${_validated_targets})
    foreach(_output_path IN LISTS _output_paths)
        _deac_build_receipt_reject_descendant_symlinks(
            "${_output_path}" "${_build_root}" "output path")
    endforeach()
    _deac_build_receipt_validate_interface_dependencies("${target_name}")

    get_target_property(_interface_targets "${target_name}"
        DEAC_BUILD_RECEIPT_INTERFACE_DEPENDENCIES)
    _deac_build_receipt_interface_fingerprint(
        _observed_interface_fingerprint ${_interface_targets})
    if(NOT _observed_interface_fingerprint STREQUAL
            _expected_interface_fingerprint)
        message(FATAL_ERROR
            "build-receipt interface properties changed after registration "
            "for ${target_name}")
    endif()

    _deac_build_receipt_target_rule_fingerprint(
        CXX "${_registration_directory}" _observed_rule_material
        ${_validated_targets})
    string(SHA256 _observed_rule_fingerprint
        "${_observed_rule_material}")
    if(NOT _observed_rule_fingerprint STREQUAL
            _expected_rule_fingerprint)
        message(FATAL_ERROR
            "build-receipt CXX rule templates changed after registration "
            "for ${target_name}")
    endif()

    set(_tool_target_types
        EXECUTABLE STATIC_LIBRARY SHARED_LIBRARY MODULE_LIBRARY OBJECT_LIBRARY)
    foreach(_tool_variable
            CMAKE_CXX_COMPILER CMAKE_AR CMAKE_RANLIB)
        get_target_property(_expected_tool_path "${target_name}"
            "DEAC_BUILD_RECEIPT_${_tool_variable}_PATH")
        get_target_property(_expected_tool_real_path "${target_name}"
            "DEAC_BUILD_RECEIPT_${_tool_variable}_REAL_PATH")
        get_target_property(_expected_tool_sha256 "${target_name}"
            "DEAC_BUILD_RECEIPT_${_tool_variable}_SHA256")
        foreach(_expected_tool_value
                _expected_tool_path
                _expected_tool_real_path
                _expected_tool_sha256)
            if("${${_expected_tool_value}}" STREQUAL
                    "${_expected_tool_value}-NOTFOUND")
                message(FATAL_ERROR
                    "build-receipt tool validation seal is incomplete for "
                    "${target_name}")
            endif()
        endforeach()
        foreach(_validated_target IN LISTS _validated_targets)
            get_target_property(_validated_target_type
                "${_validated_target}" TYPE)
            if(_validated_target_type IN_LIST _tool_target_types)
                _deac_build_receipt_target_directory(
                    "${_validated_target}" _target_directory)
                _deac_build_receipt_validate_named_tool_in_directory(
                    "${_tool_variable}" "${_target_directory}"
                    "${_expected_tool_path}" "${_expected_tool_real_path}"
                    "${_expected_tool_sha256}")
            endif()
        endforeach()
    endforeach()
endfunction()

function(_deac_build_receipt_validate_current_directory_seals)
    cmake_language(DEFER GET_CALL_IDS _pending_deferred_calls)
    list(REMOVE_ITEM _pending_deferred_calls
        deac_build_receipt_directory_seal
        deac_build_receipt_top_level_seal)
    if(_pending_deferred_calls)
        cmake_language(DEFER ID deac_build_receipt_directory_seal CALL
            _deac_build_receipt_validate_current_directory_seals)
        return()
    endif()
    get_property(_sealed_targets DIRECTORY
        PROPERTY DEAC_BUILD_RECEIPT_SEALED_TARGETS)
    foreach(_sealed_target IN LISTS _sealed_targets)
        _deac_build_receipt_validate_seal("${_sealed_target}")
    endforeach()
endfunction()

function(_deac_build_receipt_validate_top_level_seals)
    cmake_language(DEFER GET_CALL_IDS _pending_deferred_calls)
    list(REMOVE_ITEM _pending_deferred_calls
        deac_build_receipt_directory_seal
        deac_build_receipt_top_level_seal)
    if(_pending_deferred_calls)
        cmake_language(DEFER ID deac_build_receipt_top_level_seal CALL
            _deac_build_receipt_validate_top_level_seals)
        return()
    endif()
    get_property(_sealed_targets GLOBAL
        PROPERTY DEAC_BUILD_RECEIPT_SEALED_TARGETS)
    foreach(_sealed_target IN LISTS _sealed_targets)
        _deac_build_receipt_validate_seal("${_sealed_target}")
    endforeach()
endfunction()

function(deac_target_add_build_receipt target_name)
    if(CMAKE_VERSION VERSION_LESS 3.27)
        message(FATAL_ERROR
            "canonical DEAC build receipts require CMake 3.27 or newer")
    endif()
    set(_supported_receipt_generators
        "Ninja" "Ninja Multi-Config" "Unix Makefiles")
    list(FIND _supported_receipt_generators "${CMAKE_GENERATOR}"
        _supported_generator_index)
    if(NOT CMAKE_HOST_UNIX OR _supported_generator_index EQUAL -1)
        message(FATAL_ERROR
            "schema-1 build receipts require a POSIX host with Ninja, "
            "Ninja Multi-Config, or Unix Makefiles")
    endif()
    if(NOT TARGET "${target_name}")
        message(FATAL_ERROR
            "deac_target_add_build_receipt requires an existing target")
    endif()
    get_target_property(_target_type "${target_name}" TYPE)
    if(NOT _target_type STREQUAL "EXECUTABLE")
        message(FATAL_ERROR
            "deac_target_add_build_receipt requires an executable target")
    endif()
    _deac_build_receipt_canonical_target(
        "${target_name}" _canonical_receipt_target)
    if(NOT _canonical_receipt_target STREQUAL target_name)
        message(FATAL_ERROR
            "deac_target_add_build_receipt requires a non-alias target")
    endif()
    get_target_property(_target_is_imported "${target_name}" IMPORTED)
    if(_target_is_imported)
        message(FATAL_ERROR
            "deac_target_add_build_receipt requires a buildable target")
    endif()
    if(NOT target_name MATCHES "^[A-Za-z0-9_.+-]+$")
        message(FATAL_ERROR
            "deac_target_add_build_receipt target name is not safely "
            "representable")
    endif()
    get_target_property(_existing_receipt_registration "${target_name}"
        DEAC_BUILD_RECEIPT_REGISTRATION_DIRECTORY)
    if(NOT "${_existing_receipt_registration}" STREQUAL
            "_existing_receipt_registration-NOTFOUND")
        message(FATAL_ERROR
            "deac_target_add_build_receipt target is already registered: "
            "${target_name}")
    endif()

    cmake_parse_arguments(
        PARSE_ARGV 1
        DEAC_RECEIPT
        ""
        "SOURCE_ROOT;GENERATED_DIRECTORY;IDENTITY_NAME;RECEIPT;BACKEND"
        "CACHE_KEYS;DEPENDENCY_TARGETS;REQUIRED_LINK_LIBRARY_NAMES;REQUIRED_LINK_LIBRARY_ARTIFACTS")
    if(DEAC_RECEIPT_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR
            "deac_target_add_build_receipt received unsupported arguments: "
            "${DEAC_RECEIPT_UNPARSED_ARGUMENTS}")
    endif()
    foreach(_required_argument
            SOURCE_ROOT
            GENERATED_DIRECTORY
            IDENTITY_NAME
            RECEIPT
            BACKEND)
        if(NOT DEAC_RECEIPT_${_required_argument})
            message(FATAL_ERROR
                "deac_target_add_build_receipt requires ${_required_argument}")
        endif()
    endforeach()
    if(NOT DEAC_RECEIPT_IDENTITY_NAME MATCHES "^[A-Za-z0-9_.-]+$")
        message(FATAL_ERROR
            "build-receipt IDENTITY_NAME must be path-safe")
    endif()
    if(NOT DEAC_RECEIPT_BACKEND MATCHES "^(none|sycl|cuda|hip)$")
        message(FATAL_ERROR
            "build-receipt BACKEND must be none, sycl, cuda, or hip")
    endif()
    if(NOT DEAC_RECEIPT_CACHE_KEYS)
        message(FATAL_ERROR
            "deac_target_add_build_receipt requires CACHE_KEYS")
    endif()
    list(REMOVE_DUPLICATES DEAC_RECEIPT_CACHE_KEYS)
    foreach(_cache_key IN LISTS DEAC_RECEIPT_CACHE_KEYS)
        if(NOT _cache_key MATCHES "^[A-Za-z_][A-Za-z0-9_]*$")
            message(FATAL_ERROR
                "invalid build-receipt cache key: ${_cache_key}")
        endif()
    endforeach()
    list(SORT DEAC_RECEIPT_CACHE_KEYS)
    string(JOIN "," _cache_keys_csv ${DEAC_RECEIPT_CACHE_KEYS})

    set(_registration_directory "${CMAKE_CURRENT_SOURCE_DIR}")
    _deac_build_receipt_configuration_snapshot(
        "${_registration_directory}" _receipt_generator
        _receipt_configuration_mode _receipt_configurations)
    if(_receipt_configuration_mode STREQUAL "multi")
        set(_receipt_requires_config true)
    else()
        set(_receipt_requires_config false)
    endif()
    list(LENGTH _receipt_configurations _receipt_configuration_count)
    _deac_build_receipt_normalize_generated_directory(
        "${DEAC_RECEIPT_GENERATED_DIRECTORY}" "${CMAKE_BINARY_DIR}"
        DEAC_RECEIPT_GENERATED_DIRECTORY)
    _deac_build_receipt_normalize_receipt_path(
        "${DEAC_RECEIPT_RECEIPT}" "${_receipt_requires_config}"
        "${CMAKE_BINARY_DIR}" DEAC_RECEIPT_RECEIPT)

    set(_canonical_dependency_targets)
    set(_supported_dependency_types
        EXECUTABLE STATIC_LIBRARY SHARED_LIBRARY MODULE_LIBRARY
        OBJECT_LIBRARY INTERFACE_LIBRARY)
    foreach(_dependency_target IN LISTS DEAC_RECEIPT_DEPENDENCY_TARGETS)
        if(NOT _dependency_target MATCHES "^[A-Za-z0-9_.:+-]+$")
            message(FATAL_ERROR
                "build-receipt dependency target name is not safely "
                "representable")
        endif()
        if(NOT TARGET "${_dependency_target}")
            message(FATAL_ERROR
                "build-receipt dependency target does not exist: "
                "${_dependency_target}")
        endif()
        _deac_build_receipt_canonical_target(
            "${_dependency_target}" _dependency_target)
        get_target_property(_dependency_is_imported
            "${_dependency_target}" IMPORTED)
        if(_dependency_is_imported)
            message(FATAL_ERROR
                "build-receipt dependency target must be buildable, not "
                "imported: ${_dependency_target}")
        endif()
        get_target_property(_dependency_type "${_dependency_target}" TYPE)
        list(FIND _supported_dependency_types "${_dependency_type}"
            _dependency_type_index)
        if(_dependency_type_index EQUAL -1)
            message(FATAL_ERROR
                "build-receipt dependency target has unsupported type "
                "${_dependency_type}: ${_dependency_target}")
        endif()
        list(FIND _canonical_dependency_targets "${_dependency_target}"
            _dependency_index)
        if(NOT _dependency_index EQUAL -1)
            message(FATAL_ERROR
                "build-receipt dependency targets must be unique after "
                "alias resolution: ${_dependency_target}")
        endif()
        list(APPEND _canonical_dependency_targets "${_dependency_target}")
    endforeach()
    set(DEAC_RECEIPT_DEPENDENCY_TARGETS
        ${_canonical_dependency_targets})
    list(SORT DEAC_RECEIPT_DEPENDENCY_TARGETS)
    set(_codemodel_dependency_targets)
    set(_interface_dependency_targets)
    foreach(_dependency_target IN LISTS DEAC_RECEIPT_DEPENDENCY_TARGETS)
        get_target_property(
            _dependency_type "${_dependency_target}" TYPE)
        if(_dependency_type STREQUAL "INTERFACE_LIBRARY")
            # Interface libraries are real direct graph contracts but CMake's
            # File API omits them from both the codemodel target list and the
            # consumer dependency IDs.  Verify the final target property here,
            # then carry an explicit name/type record into the build-time
            # receipt while the resolved provider remains independently
            # checked in the File API link fragments below.
            list(APPEND _interface_dependency_targets
                "${_dependency_target}")
        else()
            list(APPEND _codemodel_dependency_targets
                "${_dependency_target}")
        endif()
    endforeach()
    set_property(TARGET "${target_name}" PROPERTY
        DEAC_BUILD_RECEIPT_INTERFACE_DEPENDENCIES
        "${_interface_dependency_targets}")
    _deac_build_receipt_validate_interface_dependencies("${target_name}")
    string(JOIN "," _codemodel_dependency_targets_csv
        ${_codemodel_dependency_targets})
    string(JOIN "," _interface_dependency_targets_csv
        ${_interface_dependency_targets})

    list(LENGTH DEAC_RECEIPT_REQUIRED_LINK_LIBRARY_NAMES
        _required_link_library_name_count)
    list(LENGTH DEAC_RECEIPT_REQUIRED_LINK_LIBRARY_ARTIFACTS
        _required_link_library_artifact_count)
    if(NOT _required_link_library_name_count EQUAL
            _required_link_library_artifact_count)
        message(FATAL_ERROR
            "build-receipt required link-library names and artifacts must "
            "have equal lengths")
    endif()
    set(_required_link_library_count
        ${_required_link_library_name_count})
    set(_required_link_library_arguments
        "-DDEAC_BUILD_RECEIPT_REQUIRED_LINK_LIBRARY_COUNT:STRING=${_required_link_library_count}")
    set(_required_link_library_names)
    set(_required_link_library_artifacts)
    list(LENGTH _receipt_configurations _artifact_configuration_count)
    if(_artifact_configuration_count GREATER 0)
        math(EXPR _artifact_configuration_last
            "${_artifact_configuration_count} - 1")
        foreach(_artifact_configuration_index
                RANGE 0 ${_artifact_configuration_last})
            set(_selected_link_artifacts_${_artifact_configuration_index})
        endforeach()
    endif()
    if(_required_link_library_count GREATER 0)
        math(EXPR _required_link_library_last
            "${_required_link_library_count} - 1")
        foreach(_required_link_library_index
                RANGE 0 ${_required_link_library_last})
            list(GET DEAC_RECEIPT_REQUIRED_LINK_LIBRARY_NAMES
                ${_required_link_library_index} _link_library_name)
            list(GET DEAC_RECEIPT_REQUIRED_LINK_LIBRARY_ARTIFACTS
                ${_required_link_library_index} _link_library_artifact)
            if(NOT _link_library_name MATCHES "^[A-Za-z0-9_.:+-]+$")
                message(FATAL_ERROR
                    "build-receipt link-library name is not path-safe: "
                    "${_link_library_name}")
            endif()
            if(_link_library_name IN_LIST _required_link_library_names)
                message(FATAL_ERROR
                    "build-receipt required link-library names must be unique: "
                    "${_link_library_name}")
            endif()
            _deac_build_receipt_normalize_link_library_artifact(
                "${_link_library_artifact}" _link_library_artifact
                _selected_link_library_artifacts)
            list(LENGTH _selected_link_library_artifacts
                _selected_link_library_artifact_count)
            if(NOT _selected_link_library_artifact_count EQUAL
                    _artifact_configuration_count)
                message(FATAL_ERROR
                    "build-receipt link-library artifact selection count "
                    "does not match configured build types")
            endif()
            foreach(_artifact_configuration_index
                    RANGE 0 ${_artifact_configuration_last})
                list(GET _selected_link_library_artifacts
                    ${_artifact_configuration_index} _selected_artifact)
                set(_selected_artifact_variable
                    "_selected_link_artifacts_${_artifact_configuration_index}")
                list(FIND ${_selected_artifact_variable}
                    "${_selected_artifact}" _selected_artifact_index)
                if(NOT _selected_artifact_index EQUAL -1)
                    list(GET _receipt_configurations
                        ${_artifact_configuration_index}
                        _artifact_configuration)
                    message(FATAL_ERROR
                        "build-receipt required link-library artifacts must "
                        "be unique in configuration "
                        "${_artifact_configuration}: ${_selected_artifact}")
                endif()
                list(APPEND ${_selected_artifact_variable}
                    "${_selected_artifact}")
            endforeach()
            if(_link_library_artifact IN_LIST
                    _required_link_library_artifacts)
                message(FATAL_ERROR
                    "build-receipt required link-library artifacts must be "
                    "unique: ${_link_library_artifact}")
            endif()
            list(APPEND _required_link_library_names
                "${_link_library_name}")
            list(APPEND _required_link_library_artifacts
                "${_link_library_artifact}")
            list(APPEND _required_link_library_arguments
                "-DDEAC_BUILD_RECEIPT_REQUIRED_LINK_LIBRARY_NAME_${_required_link_library_index}:STRING=${_link_library_name}"
                "-DDEAC_BUILD_RECEIPT_REQUIRED_LINK_LIBRARY_ARTIFACT_${_required_link_library_index}:FILEPATH=${_link_library_artifact}")
        endforeach()
    endif()

    get_filename_component(
        _source_root "${DEAC_RECEIPT_SOURCE_ROOT}" REALPATH)
    if(NOT EXISTS "${_source_root}/VERSION")
        message(FATAL_ERROR
            "build-receipt source root has no VERSION file: ${_source_root}")
    endif()

    cmake_file_api(
        QUERY
        API_VERSION 1
        CODEMODEL 2.6
        CACHE 2
        TOOLCHAINS 1)

    set(_validated_targets
        "${target_name}" ${DEAC_RECEIPT_DEPENDENCY_TARGETS})
    _deac_build_receipt_validate_target_scopes(
        "${target_name}" "${_registration_directory}"
        ${_validated_targets})

    _deac_build_receipt_query_enabled_languages(_enabled_languages)
    _deac_build_receipt_reject_unsupported_languages(${_enabled_languages})
    set(_receipt_languages)
    set(_tool_arguments)
    set(_toolchain_fingerprint_material "deac-build-toolchain-v2\n")
    foreach(_language CXX CUDA)
        if(_language IN_LIST _enabled_languages)
            _deac_build_receipt_snapshot_tool(
                "${_language}" _language_arguments _language_fingerprint
                _language_tool_path _language_tool_real_path
                _language_tool_sha256)
            list(APPEND _receipt_languages "${_language}")
            list(APPEND _tool_arguments ${_language_arguments})
            string(APPEND _toolchain_fingerprint_material
                "${_language_fingerprint}")
            _deac_build_receipt_target_rule_fingerprint(
                "${_language}" "${_registration_directory}"
                _language_rule_fingerprint
                ${_validated_targets})
            if(_language STREQUAL "CXX")
                string(SHA256 _cxx_rule_fingerprint
                    "${_language_rule_fingerprint}")
            endif()
            string(APPEND _toolchain_fingerprint_material
                "${_language_rule_fingerprint}")
            set(_compiler_variable "CMAKE_${_language}_COMPILER")
            set_property(TARGET "${target_name}" PROPERTY
                "DEAC_BUILD_RECEIPT_${_compiler_variable}_PATH"
                "${_language_tool_path}")
            set_property(TARGET "${target_name}" PROPERTY
                "DEAC_BUILD_RECEIPT_${_compiler_variable}_REAL_PATH"
                "${_language_tool_real_path}")
            set_property(TARGET "${target_name}" PROPERTY
                "DEAC_BUILD_RECEIPT_${_compiler_variable}_SHA256"
                "${_language_tool_sha256}")
        endif()
    endforeach()
    if(NOT "CXX" IN_LIST _receipt_languages)
        message(FATAL_ERROR "build receipt requires the CXX language")
    endif()
    string(JOIN "," _receipt_languages_csv ${_receipt_languages})

    set(_archive_tools CMAKE_AR CMAKE_RANLIB)
    foreach(_archive_tool IN LISTS _archive_tools)
        _deac_build_receipt_snapshot_named_tool(
            "${_archive_tool}" _archive_tool_arguments
            _archive_tool_fingerprint _archive_tool_path
            _archive_tool_real_path _archive_tool_sha256)
        list(APPEND _tool_arguments ${_archive_tool_arguments})
        string(APPEND _toolchain_fingerprint_material
            "${_archive_tool_fingerprint}")
        set_property(TARGET "${target_name}" PROPERTY
            "DEAC_BUILD_RECEIPT_${_archive_tool}_PATH"
            "${_archive_tool_path}")
        set_property(TARGET "${target_name}" PROPERTY
            "DEAC_BUILD_RECEIPT_${_archive_tool}_REAL_PATH"
            "${_archive_tool_real_path}")
        set_property(TARGET "${target_name}" PROPERTY
            "DEAC_BUILD_RECEIPT_${_archive_tool}_SHA256"
            "${_archive_tool_sha256}")
    endforeach()
    string(JOIN "," _archive_tools_csv ${_archive_tools})

    get_filename_component(_cmake_real "${CMAKE_COMMAND}" REALPATH)
    if(NOT EXISTS "${_cmake_real}" OR IS_DIRECTORY "${_cmake_real}")
        message(FATAL_ERROR
            "build-receipt CMake command is not a regular file: ${CMAKE_COMMAND}")
    endif()
    file(SHA256 "${_cmake_real}" _cmake_sha256)
    foreach(_value "${CMAKE_COMMAND}" "${_cmake_real}")
        _deac_build_receipt_require_plain("${_value}" "CMake identity")
    endforeach()
    foreach(_field PATH REAL_PATH SHA256)
        if(_field STREQUAL "PATH")
            set(_value "${CMAKE_COMMAND}")
        elseif(_field STREQUAL "REAL_PATH")
            set(_value "${_cmake_real}")
        else()
            set(_value "${_cmake_sha256}")
        endif()
        string(LENGTH "${_value}" _length)
        string(APPEND _toolchain_fingerprint_material
            "CMAKE.${_field}:${_length}:${_value}\n")
    endforeach()
    string(SHA256 _toolchain_fingerprint
        "${_toolchain_fingerprint_material}")

    # Changing compiler, archive-tool, or CMake bytes followed by a required
    # reconfigure must invalidate every object, even when a tool pathname is
    # unchanged.
    # The same definition is visible in the File API compile groups recorded
    # below, binding that invalidation key into the receipt itself.
    foreach(_fingerprinted_target
            "${target_name}" ${DEAC_RECEIPT_DEPENDENCY_TARGETS})
        get_target_property(_fingerprinted_type
            "${_fingerprinted_target}" TYPE)
        if(_fingerprinted_type MATCHES
                "^(EXECUTABLE|MODULE_LIBRARY|OBJECT_LIBRARY|SHARED_LIBRARY|STATIC_LIBRARY)$")
            get_target_property(_injected_fingerprint
                "${_fingerprinted_target}"
                DEAC_BUILD_RECEIPT_INJECTED_TOOLCHAIN_FINGERPRINT_SHA256)
            if(_injected_fingerprint MATCHES "-NOTFOUND$")
                target_compile_definitions("${_fingerprinted_target}" PRIVATE
                    "DEAC_BUILD_TOOLCHAIN_FINGERPRINT_SHA256=${_toolchain_fingerprint}")
                set_property(TARGET "${_fingerprinted_target}" PROPERTY
                    DEAC_BUILD_RECEIPT_INJECTED_TOOLCHAIN_FINGERPRINT_SHA256
                    "${_toolchain_fingerprint}")
            elseif(NOT _injected_fingerprint STREQUAL
                    _toolchain_fingerprint)
                message(FATAL_ERROR
                    "build-receipt target ${_fingerprinted_target} is shared "
                    "by incompatible toolchain fingerprints")
            endif()
        endif()
    endforeach()

    get_filename_component(
        _support_directory
        "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../deac/src"
        REALPATH)
    string(MAKE_C_IDENTIFIER
        "${DEAC_RECEIPT_IDENTITY_NAME}_${target_name}_build_receipt"
        _receipt_identifier)
    set(_receipt "${DEAC_RECEIPT_RECEIPT}")

    set(_git_argument)
    if(Git_FOUND)
        list(APPEND _git_argument
            "-DGIT_EXECUTABLE:FILEPATH=${GIT_EXECUTABLE}")
    endif()

    set(_configuration_directory
        "${DEAC_RECEIPT_GENERATED_DIRECTORY}/$<CONFIG>")
    set(_generated_source
        "${_configuration_directory}/${_receipt_identifier}.cpp")
    set(_refresh
        "${_configuration_directory}/${_receipt_identifier}.refresh")
    set(_rebuild
        "${_configuration_directory}/${_receipt_identifier}.rebuild")
    set(_compiled_marker
        "${_configuration_directory}/${_receipt_identifier}.compiled")

    set(_concrete_output_paths)
    foreach(_configuration IN LISTS _receipt_configurations)
        list(APPEND _concrete_output_paths
            "${DEAC_RECEIPT_GENERATED_DIRECTORY}/${_configuration}/${_receipt_identifier}.cpp"
            "${DEAC_RECEIPT_GENERATED_DIRECTORY}/${_configuration}/${_receipt_identifier}.refresh"
            "${DEAC_RECEIPT_GENERATED_DIRECTORY}/${_configuration}/${_receipt_identifier}.rebuild"
            "${DEAC_RECEIPT_GENERATED_DIRECTORY}/${_configuration}/${_receipt_identifier}.compiled")
        string(REPLACE "$<CONFIG>" "${_configuration}"
            _configured_receipt "${_receipt}")
        list(APPEND _concrete_output_paths "${_configured_receipt}")
    endforeach()
    list(REMOVE_DUPLICATES _concrete_output_paths)
    foreach(_concrete_output_path IN LISTS _concrete_output_paths)
        _deac_build_receipt_reject_descendant_symlinks(
            "${_concrete_output_path}" "${CMAKE_BINARY_DIR}" "output path")
    endforeach()

    # Source properties are keyed by the literal path passed to CMake.  A
    # property attached to the $<CONFIG>-spelled source is therefore not
    # applied to the concrete source path emitted by Makefile generators.
    # Register each configured source explicitly, while exposing only the
    # selected configuration to the target graph.
    set(_configured_receipt_sources)
    foreach(_configuration IN LISTS _receipt_configurations)
        set(_configured_source
            "${DEAC_RECEIPT_GENERATED_DIRECTORY}/${_configuration}/${_receipt_identifier}.cpp")
        set(_configured_rebuild
            "${DEAC_RECEIPT_GENERATED_DIRECTORY}/${_configuration}/${_receipt_identifier}.rebuild")
        set_source_files_properties(
            "${_configured_source}" PROPERTIES
            GENERATED TRUE
            OBJECT_DEPENDS "${_configured_rebuild}")
        list(APPEND _configured_receipt_sources
            "$<$<CONFIG:${_configuration}>:${_configured_source}>")
    endforeach()

    add_custom_command(
        # The command deliberately never creates this symbolic primary output.
        # A single $<CONFIG>-qualified edge keeps Ninja Multi-Config graphs
        # isolated while making every ordinary selected-config build refresh.
        OUTPUT "${_refresh}"
        BYPRODUCTS "${_generated_source}" "${_receipt}"
        COMMAND
            "${CMAKE_COMMAND}"
            "-DDEAC_BUILD_RECEIPT_VALIDATE_OUTPUT_COUNT:STRING=5"
            "-DDEAC_BUILD_RECEIPT_VALIDATE_BUILD_ROOT:PATH=${CMAKE_BINARY_DIR}"
            "-DDEAC_BUILD_RECEIPT_VALIDATE_OUTPUT_0:FILEPATH=${_generated_source}"
            "-DDEAC_BUILD_RECEIPT_VALIDATE_OUTPUT_1:FILEPATH=${_receipt}"
            "-DDEAC_BUILD_RECEIPT_VALIDATE_OUTPUT_2:FILEPATH=${_compiled_marker}"
            "-DDEAC_BUILD_RECEIPT_VALIDATE_OUTPUT_3:FILEPATH=${_refresh}"
            "-DDEAC_BUILD_RECEIPT_VALIDATE_OUTPUT_4:FILEPATH=${_rebuild}"
            -P "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/DeacBuildReceipt.cmake"
        COMMAND
            "${CMAKE_COMMAND}"
            "-DDEAC_BUILD_RECEIPT_SOURCE_ROOT:PATH=${_source_root}"
            "-DDEAC_BUILD_RECEIPT_CMAKE_SOURCE_ROOT:PATH=${CMAKE_SOURCE_DIR}"
            "-DDEAC_BUILD_RECEIPT_BUILD_ROOT:PATH=${CMAKE_BINARY_DIR}"
            "-DDEAC_BUILD_RECEIPT_REPLY_DIRECTORY:PATH=${CMAKE_BINARY_DIR}/.cmake/api/v1/reply"
            "-DDEAC_BUILD_RECEIPT_TARGET:STRING=${target_name}"
            "-DDEAC_BUILD_RECEIPT_CONFIGURATION:STRING=$<CONFIG>"
            "-DDEAC_BUILD_RECEIPT_BACKEND:STRING=${DEAC_RECEIPT_BACKEND}"
            "-DDEAC_BUILD_RECEIPT_CACHE_KEYS:STRING=${_cache_keys_csv}"
            "-DDEAC_BUILD_RECEIPT_CODEMODEL_DEPENDENCY_TARGETS:STRING=${_codemodel_dependency_targets_csv}"
            "-DDEAC_BUILD_RECEIPT_INTERFACE_DEPENDENCY_TARGETS:STRING=${_interface_dependency_targets_csv}"
            "-DDEAC_BUILD_RECEIPT_LANGUAGES:STRING=${_receipt_languages_csv}"
            "-DDEAC_BUILD_RECEIPT_ARCHIVE_TOOLS:STRING=${_archive_tools_csv}"
            "-DDEAC_BUILD_RECEIPT_TOOLCHAIN_FINGERPRINT:STRING=${_toolchain_fingerprint}"
            "-DDEAC_BUILD_RECEIPT_CMAKE_PATH:FILEPATH=${CMAKE_COMMAND}"
            "-DDEAC_BUILD_RECEIPT_CMAKE_REAL_PATH:FILEPATH=${_cmake_real}"
            "-DDEAC_BUILD_RECEIPT_CMAKE_SHA256:STRING=${_cmake_sha256}"
            "-DDEAC_BUILD_RECEIPT_OUTPUT_SOURCE:FILEPATH=${_generated_source}"
            "-DDEAC_BUILD_RECEIPT_OUTPUT_RECEIPT:FILEPATH=${_receipt}"
            "-DDEAC_BUILD_RECEIPT_PREVIOUS_COMPILE_MARKER:FILEPATH=${_compiled_marker}"
            ${_required_link_library_arguments}
            ${_git_argument}
            ${_tool_arguments}
            -P "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/GenerateDeacBuildReceipt.cmake"
        DEPENDS
            "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/DeacBuildIdentity.cmake"
            "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/DeacBuildReceipt.cmake"
            "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/GenerateDeacBuildReceipt.cmake"
            "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/VerifyDeacBuildReceiptTools.cmake"
            "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/deac_build_receipt_data.cpp.in"
            "${_source_root}/VERSION"
        COMMENT "Embedding effective build receipt for ${target_name} ($<CONFIG>)"
        VERBATIM)
    add_custom_command(
        # This symbolic output separates receipt generation from the normal
        # generated-source relationship that CMake otherwise treats as
        # timestamp-only.  Make object and link rules depend on the
        # always-missing token, so coarse filesystem timestamps cannot suppress
        # a rebuild.  Ninja additionally observes the explicitly touched
        # generated source from the refresh edge.
        OUTPUT "${_rebuild}"
        COMMAND "${CMAKE_COMMAND}" -E true
        DEPENDS "${_refresh}"
        VERBATIM)
    set_source_files_properties(
        "${_refresh}" "${_rebuild}"
        PROPERTIES GENERATED TRUE SYMBOLIC TRUE)
    set_property(TARGET "${target_name}" APPEND PROPERTY
        LINK_DEPENDS "${_rebuild}")
    target_sources("${target_name}" PRIVATE
        ${_configured_receipt_sources} "${_rebuild}")
    target_include_directories("${target_name}" PRIVATE
        "${_support_directory}")

    # This catches persistent compiler/archive-tool/CMake replacement after
    # receipt generation but before the final link.  As with any ordinary
    # build graph, an adversarial swap-and-restore between process invocations
    # is outside CMake's attestation boundary.
    add_custom_command(TARGET "${target_name}" PRE_LINK
        COMMAND
            "${CMAKE_COMMAND}"
            "-DDEAC_BUILD_RECEIPT_LANGUAGES:STRING=${_receipt_languages_csv}"
            "-DDEAC_BUILD_RECEIPT_ARCHIVE_TOOLS:STRING=${_archive_tools_csv}"
            "-DDEAC_BUILD_RECEIPT_CMAKE_PATH:FILEPATH=${CMAKE_COMMAND}"
            "-DDEAC_BUILD_RECEIPT_CMAKE_REAL_PATH:FILEPATH=${_cmake_real}"
            "-DDEAC_BUILD_RECEIPT_CMAKE_SHA256:STRING=${_cmake_sha256}"
            ${_tool_arguments}
            -P "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/VerifyDeacBuildReceiptTools.cmake"
        COMMAND
            "${CMAKE_COMMAND}"
            "-DDEAC_BUILD_RECEIPT_VALIDATE_OUTPUT_COUNT:STRING=1"
            "-DDEAC_BUILD_RECEIPT_VALIDATE_BUILD_ROOT:PATH=${CMAKE_BINARY_DIR}"
            "-DDEAC_BUILD_RECEIPT_VALIDATE_OUTPUT_0:FILEPATH=${_compiled_marker}"
            -P "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/DeacBuildReceipt.cmake"
        COMMAND "${CMAKE_COMMAND}" -E touch "${_compiled_marker}"
        COMMENT "Verifying build-receipt tool bytes for ${target_name}"
        VERBATIM)
    set_property(TARGET "${target_name}" APPEND PROPERTY
        ADDITIONAL_CLEAN_FILES
            "${_generated_source};${_receipt};${_compiled_marker}")

    _deac_build_receipt_interface_fingerprint(
        _interface_fingerprint ${_interface_dependency_targets})
    set_property(TARGET "${target_name}" PROPERTY
        DEAC_BUILD_RECEIPT_REGISTRATION_DIRECTORY
        "${_registration_directory}")
    set_property(TARGET "${target_name}" PROPERTY
        DEAC_BUILD_RECEIPT_VALIDATED_TARGETS "${_validated_targets}")
    set_property(TARGET "${target_name}" PROPERTY
        DEAC_BUILD_RECEIPT_CXX_RULE_FINGERPRINT
        "${_cxx_rule_fingerprint}")
    set_property(TARGET "${target_name}" PROPERTY
        DEAC_BUILD_RECEIPT_INTERFACE_FINGERPRINT
        "${_interface_fingerprint}")
    set_property(TARGET "${target_name}" PROPERTY
        DEAC_BUILD_RECEIPT_BUILD_ROOT "${CMAKE_BINARY_DIR}")
    set_property(TARGET "${target_name}" PROPERTY
        DEAC_BUILD_RECEIPT_OUTPUT_PATHS "${_concrete_output_paths}")
    set_property(TARGET "${target_name}" PROPERTY
        DEAC_BUILD_RECEIPT_GENERATOR "${_receipt_generator}")
    set_property(TARGET "${target_name}" PROPERTY
        DEAC_BUILD_RECEIPT_CONFIGURATION_MODE
        "${_receipt_configuration_mode}")
    set_property(TARGET "${target_name}" PROPERTY
        DEAC_BUILD_RECEIPT_CONFIGURATIONS "${_receipt_configurations}")
    set_property(TARGET "${target_name}" PROPERTY
        DEAC_BUILD_RECEIPT_TOP_LEVEL_SOURCE_DIRECTORY
        "${CMAKE_SOURCE_DIR}")
    _deac_build_receipt_validate_seal("${target_name}")

    set_property(DIRECTORY APPEND PROPERTY
        DEAC_BUILD_RECEIPT_SEALED_TARGETS "${target_name}")
    get_property(_directory_seal_scheduled DIRECTORY
        PROPERTY DEAC_BUILD_RECEIPT_SEAL_SCHEDULED)
    if(NOT _directory_seal_scheduled)
        set_property(DIRECTORY PROPERTY
            DEAC_BUILD_RECEIPT_SEAL_SCHEDULED true)
        cmake_language(DEFER ID deac_build_receipt_directory_seal CALL
            _deac_build_receipt_validate_current_directory_seals)
    endif()

    set_property(GLOBAL APPEND PROPERTY
        DEAC_BUILD_RECEIPT_SEALED_TARGETS "${target_name}")
    get_property(_top_level_seal_scheduled GLOBAL
        PROPERTY DEAC_BUILD_RECEIPT_TOP_LEVEL_SEAL_SCHEDULED)
    if(NOT _top_level_seal_scheduled)
        set_property(GLOBAL PROPERTY
            DEAC_BUILD_RECEIPT_TOP_LEVEL_SEAL_SCHEDULED true)
        cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}"
            ID deac_build_receipt_top_level_seal CALL
            _deac_build_receipt_validate_top_level_seals)
    endif()

    set(DEAC_BUILD_RECEIPT "${_receipt}" PARENT_SCOPE)
    set(DEAC_BUILD_RECEIPT_GENERATED_SOURCE
        "${DEAC_RECEIPT_GENERATED_DIRECTORY}/$<CONFIG>/${_receipt_identifier}.cpp"
        PARENT_SCOPE)
    # Expose the primary custom-command output for callers that need a
    # receipt-only verification target.  Depending on the generated source
    # BYPRODUCT directly is not portable with Unix Makefiles because its rule
    # remains owned by the consuming target's build.make.
    set(DEAC_BUILD_RECEIPT_REFRESH "${_refresh}" PARENT_SCOPE)
    set(DEAC_BUILD_RECEIPT_TOOLCHAIN_FINGERPRINT
        "${_toolchain_fingerprint}" PARENT_SCOPE)
endfunction()

if(CMAKE_SCRIPT_MODE_FILE STREQUAL CMAKE_CURRENT_LIST_FILE AND
        DEFINED DEAC_BUILD_RECEIPT_VALIDATE_OUTPUT_COUNT)
    if(NOT DEAC_BUILD_RECEIPT_VALIDATE_OUTPUT_COUNT MATCHES "^[1-9][0-9]*$")
        message(FATAL_ERROR
            "DEAC_BUILD_RECEIPT_VALIDATE_OUTPUT_COUNT must be positive")
    endif()
    if(NOT DEFINED DEAC_BUILD_RECEIPT_VALIDATE_BUILD_ROOT OR
            "${DEAC_BUILD_RECEIPT_VALIDATE_BUILD_ROOT}" STREQUAL "")
        message(FATAL_ERROR
            "DEAC_BUILD_RECEIPT_VALIDATE_BUILD_ROOT is required")
    endif()
    _deac_build_receipt_require_safe_path_text(
        "${DEAC_BUILD_RECEIPT_VALIDATE_BUILD_ROOT}" "build root")
    if(NOT IS_ABSOLUTE "${DEAC_BUILD_RECEIPT_VALIDATE_BUILD_ROOT}")
        message(FATAL_ERROR "build-receipt build root must be absolute")
    endif()
    set(_validated_build_root
        "${DEAC_BUILD_RECEIPT_VALIDATE_BUILD_ROOT}")
    cmake_path(NORMAL_PATH _validated_build_root)
    if(NOT _validated_build_root STREQUAL
            DEAC_BUILD_RECEIPT_VALIDATE_BUILD_ROOT)
        message(FATAL_ERROR "build-receipt build root must be normalized")
    endif()

    math(EXPR _validated_output_last
        "${DEAC_BUILD_RECEIPT_VALIDATE_OUTPUT_COUNT} - 1")
    foreach(_validated_output_index RANGE 0 ${_validated_output_last})
        set(_validated_output_variable
            "DEAC_BUILD_RECEIPT_VALIDATE_OUTPUT_${_validated_output_index}")
        if(NOT DEFINED ${_validated_output_variable} OR
                "${${_validated_output_variable}}" STREQUAL "")
            message(FATAL_ERROR
                "${_validated_output_variable} is required")
        endif()
        set(_validated_output_path "${${_validated_output_variable}}")
        _deac_build_receipt_require_safe_path_text(
            "${_validated_output_path}" "output path")
        if(NOT IS_ABSOLUTE "${_validated_output_path}" OR
                "${_validated_output_path}" MATCHES "(^|/)\\.\\.?(/|$)")
            message(FATAL_ERROR
                "build-receipt output path must be absolute and normalized")
        endif()
        set(_normalized_output_path "${_validated_output_path}")
        cmake_path(NORMAL_PATH _normalized_output_path)
        if(NOT _normalized_output_path STREQUAL _validated_output_path)
            message(FATAL_ERROR
                "build-receipt output path must be normalized")
        endif()
        _deac_build_receipt_reject_descendant_symlinks(
            "${_validated_output_path}" "${_validated_build_root}"
            "output path")
    endforeach()
endif()
