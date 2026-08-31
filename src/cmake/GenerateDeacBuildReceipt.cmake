cmake_minimum_required(VERSION 3.27)

foreach(_required_variable
        DEAC_BUILD_RECEIPT_SOURCE_ROOT
        DEAC_BUILD_RECEIPT_CMAKE_SOURCE_ROOT
        DEAC_BUILD_RECEIPT_BUILD_ROOT
        DEAC_BUILD_RECEIPT_REPLY_DIRECTORY
        DEAC_BUILD_RECEIPT_TARGET
        DEAC_BUILD_RECEIPT_CONFIGURATION
        DEAC_BUILD_RECEIPT_BACKEND
        DEAC_BUILD_RECEIPT_CACHE_KEYS
        DEAC_BUILD_RECEIPT_LANGUAGES
        DEAC_BUILD_RECEIPT_TOOLCHAIN_FINGERPRINT
        DEAC_BUILD_RECEIPT_CMAKE_PATH
        DEAC_BUILD_RECEIPT_CMAKE_REAL_PATH
        DEAC_BUILD_RECEIPT_CMAKE_SHA256
        DEAC_BUILD_RECEIPT_OUTPUT_SOURCE
        DEAC_BUILD_RECEIPT_OUTPUT_RECEIPT)
    if(NOT DEFINED ${_required_variable} OR
            "${${_required_variable}}" STREQUAL "")
        message(FATAL_ERROR "${_required_variable} is required")
    endif()
endforeach()
if(NOT DEFINED DEAC_BUILD_RECEIPT_DEPENDENCY_TARGETS)
    message(FATAL_ERROR "DEAC_BUILD_RECEIPT_DEPENDENCY_TARGETS is required")
endif()

function(_deac_receipt_json_quote output value)
    set(_value "${value}")
    string(REPLACE "\\" "\\\\" _value "${_value}")
    string(REPLACE "\"" "\\\"" _value "${_value}")
    string(ASCII 8 _backspace)
    string(ASCII 12 _form_feed)
    string(REPLACE "${_backspace}" "\\b" _value "${_value}")
    string(REPLACE "${_form_feed}" "\\f" _value "${_value}")
    string(REPLACE "\n" "\\n" _value "${_value}")
    string(REPLACE "\r" "\\r" _value "${_value}")
    string(REPLACE "\t" "\\t" _value "${_value}")
    set(${output} "\"${_value}\"" PARENT_SCOPE)
endfunction()

function(_deac_receipt_json_get output label document)
    string(JSON _value ERROR_VARIABLE _error GET "${document}" ${ARGN})
    if(NOT _error STREQUAL "NOTFOUND")
        message(FATAL_ERROR
            "build-receipt ${label} is missing or invalid: ${_error}")
    endif()
    set(${output} "${_value}" PARENT_SCOPE)
endfunction()

function(_deac_receipt_json_length output label document)
    string(JSON _value ERROR_VARIABLE _error LENGTH "${document}" ${ARGN})
    if(NOT _error STREQUAL "NOTFOUND")
        message(FATAL_ERROR
            "build-receipt ${label} is missing or invalid: ${_error}")
    endif()
    set(${output} "${_value}" PARENT_SCOPE)
endfunction()

function(_deac_receipt_json_type output exists label document)
    string(JSON _value ERROR_VARIABLE _error TYPE "${document}" ${ARGN})
    if(_error STREQUAL "NOTFOUND")
        set(${output} "${_value}" PARENT_SCOPE)
        set(${exists} true PARENT_SCOPE)
    else()
        set(${output} "" PARENT_SCOPE)
        set(${exists} false PARENT_SCOPE)
    endif()
endfunction()

function(_deac_receipt_optional_string output label document)
    _deac_receipt_json_type(
        _type _exists "${label}" "${document}" ${ARGN})
    if(NOT _exists)
        set(${output} "null" PARENT_SCOPE)
        return()
    endif()
    if(NOT _type STREQUAL "STRING")
        message(FATAL_ERROR "build-receipt ${label} must be a string")
    endif()
    _deac_receipt_json_get(_value "${label}" "${document}" ${ARGN})
    _deac_receipt_json_quote(_quoted "${_value}")
    set(${output} "${_quoted}" PARENT_SCOPE)
endfunction()

function(_deac_receipt_optional_path output label document)
    _deac_receipt_json_type(
        _type _exists "${label}" "${document}" ${ARGN})
    if(NOT _exists)
        set(${output} "null" PARENT_SCOPE)
        return()
    endif()
    if(NOT _type STREQUAL "STRING")
        message(FATAL_ERROR "build-receipt ${label} must be a string")
    endif()
    _deac_receipt_json_get(_value "${label}" "${document}" ${ARGN})
    _deac_receipt_normalize_path(_value "${_value}")
    _deac_receipt_json_quote(_quoted "${_value}")
    set(${output} "${_quoted}" PARENT_SCOPE)
endfunction()

function(_deac_receipt_optional_boolean output label document)
    _deac_receipt_json_type(
        _type _exists "${label}" "${document}" ${ARGN})
    if(NOT _exists)
        set(${output} "false" PARENT_SCOPE)
        return()
    endif()
    if(NOT _type STREQUAL "BOOLEAN")
        message(FATAL_ERROR "build-receipt ${label} must be a boolean")
    endif()
    _deac_receipt_json_get(_value "${label}" "${document}" ${ARGN})
    if(_value)
        set(${output} "true" PARENT_SCOPE)
    else()
        set(${output} "false" PARENT_SCOPE)
    endif()
endfunction()

function(_deac_receipt_latest_index output reply_directory)
    file(GLOB _matches LIST_DIRECTORIES false
        "${reply_directory}/index-*.json")
    list(LENGTH _matches _count)
    if(_count LESS 1)
        message(FATAL_ERROR "build-receipt File API index is unavailable")
    endif()
    # The File API specifies the lexicographically greatest index as current
    # during the brief interval in which two generation indexes coexist.
    list(SORT _matches)
    list(GET _matches -1 _match)
    set(${output} "${_match}" PARENT_SCOPE)
endfunction()

function(_deac_receipt_reply_file
        output label reply_key expected_kind expected_major index_document)
    _deac_receipt_json_get(
        _json_file "${label} JSON file" "${index_document}"
        reply "${reply_key}" jsonFile)
    _deac_receipt_json_get(
        _kind "${label} kind" "${index_document}"
        reply "${reply_key}" kind)
    _deac_receipt_json_get(
        _major "${label} major version" "${index_document}"
        reply "${reply_key}" version major)
    if(NOT _kind STREQUAL "${expected_kind}" OR
            NOT _major EQUAL expected_major)
        message(FATAL_ERROR
            "build-receipt ${label} reply has an unsupported kind/version")
    endif()
    if(NOT _json_file MATCHES "^[A-Za-z0-9_.-]+\\.json$")
        message(FATAL_ERROR
            "build-receipt ${label} reply has an unsafe filename")
    endif()
    set(${output}
        "${DEAC_BUILD_RECEIPT_REPLY_DIRECTORY}/${_json_file}"
        PARENT_SCOPE)
endfunction()

function(_deac_receipt_one_file output label pattern)
    file(GLOB _matches LIST_DIRECTORIES false "${pattern}")
    list(LENGTH _matches _count)
    if(NOT _count EQUAL 1)
        message(FATAL_ERROR
            "build-receipt expected one ${label}, found ${_count}")
    endif()
    list(GET _matches 0 _match)
    set(${output} "${_match}" PARENT_SCOPE)
endfunction()

function(_deac_receipt_read_json output label path)
    if(NOT EXISTS "${path}" OR IS_DIRECTORY "${path}")
        message(FATAL_ERROR "build-receipt ${label} is unavailable: ${path}")
    endif()
    file(SIZE "${path}" _size)
    if(_size GREATER 8388608)
        message(FATAL_ERROR "build-receipt ${label} exceeds 8 MiB")
    endif()
    file(READ "${path}" _document)
    string(JSON _type ERROR_VARIABLE _error TYPE "${_document}")
    if(NOT _error STREQUAL "NOTFOUND" OR NOT _type STREQUAL "OBJECT")
        message(FATAL_ERROR "build-receipt ${label} is not a JSON object")
    endif()
    set(${output} "${_document}" PARENT_SCOPE)
endfunction()

function(_deac_receipt_normalize_path output value)
    set(_path "${value}")
    if(NOT IS_ABSOLUTE "${_path}")
        cmake_path(
            ABSOLUTE_PATH _path
            BASE_DIRECTORY "${DEAC_BUILD_RECEIPT_CMAKE_SOURCE_ROOT}"
            NORMALIZE
            OUTPUT_VARIABLE _path)
    else()
        cmake_path(NORMAL_PATH _path OUTPUT_VARIABLE _path)
    endif()
    foreach(_root_kind SOURCE BUILD)
        if(_root_kind STREQUAL "SOURCE")
            set(_root "${DEAC_BUILD_RECEIPT_CMAKE_SOURCE_ROOT}")
            set(_token "<SOURCE_ROOT>")
        else()
            set(_root "${DEAC_BUILD_RECEIPT_BUILD_ROOT}")
            set(_token "<BUILD_ROOT>")
        endif()
        cmake_path(NORMAL_PATH _root OUTPUT_VARIABLE _root)
        file(RELATIVE_PATH _relative "${_root}" "${_path}")
        if(_relative STREQUAL "")
            set(${output} "${_token}" PARENT_SCOPE)
            return()
        endif()
        if(NOT IS_ABSOLUTE "${_relative}" AND
                NOT _relative MATCHES "^\.\.(/|$)")
            set(${output} "${_token}/${_relative}" PARENT_SCOPE)
            return()
        endif()
    endforeach()
    set(${output} "${_path}" PARENT_SCOPE)
endfunction()

function(_deac_receipt_normalize_path_list output value)
    set(_normalized "")
    set(_separator "")
    foreach(_element IN LISTS value)
        if(_element STREQUAL "")
            set(_normalized_element "")
        else()
            _deac_receipt_normalize_path(
                _normalized_element "${_element}")
        endif()
        string(APPEND _normalized
            "${_separator}${_normalized_element}")
        set(_separator ";")
    endforeach()
    set(${output} "${_normalized}" PARENT_SCOPE)
endfunction()

function(_deac_receipt_string_array output label document)
    _deac_receipt_json_type(
        _type _exists "${label}" "${document}" ${ARGN})
    if(NOT _exists)
        set(${output} "[]" PARENT_SCOPE)
        return()
    endif()
    if(NOT _type STREQUAL "ARRAY")
        message(FATAL_ERROR "build-receipt ${label} must be an array")
    endif()
    _deac_receipt_json_length(_length "${label}" "${document}" ${ARGN})
    set(_json "[")
    set(_separator "")
    if(_length GREATER 0)
        math(EXPR _last "${_length} - 1")
        foreach(_index RANGE 0 ${_last})
            _deac_receipt_json_get(
                _value "${label}[${_index}]" "${document}" ${ARGN} ${_index})
            _deac_receipt_json_quote(_quoted "${_value}")
            string(APPEND _json "${_separator}${_quoted}")
            set(_separator ",")
        endforeach()
    endif()
    string(APPEND _json "]")
    set(${output} "${_json}" PARENT_SCOPE)
endfunction()

function(_deac_receipt_path_array output label document)
    _deac_receipt_json_type(
        _type _exists "${label}" "${document}" ${ARGN})
    if(NOT _exists)
        set(${output} "[]" PARENT_SCOPE)
        return()
    endif()
    if(NOT _type STREQUAL "ARRAY")
        message(FATAL_ERROR "build-receipt ${label} must be an array")
    endif()
    _deac_receipt_json_length(_length "${label}" "${document}" ${ARGN})
    set(_json "[")
    set(_separator "")
    if(_length GREATER 0)
        math(EXPR _last "${_length} - 1")
        foreach(_index RANGE 0 ${_last})
            _deac_receipt_json_get(
                _value "${label}[${_index}]" "${document}" ${ARGN} ${_index})
            _deac_receipt_normalize_path(_normalized "${_value}")
            _deac_receipt_json_quote(_quoted "${_normalized}")
            string(APPEND _json "${_separator}${_quoted}")
            set(_separator ",")
        endforeach()
    endif()
    string(APPEND _json "]")
    set(${output} "${_json}" PARENT_SCOPE)
endfunction()

function(_deac_receipt_fragment_array output label document)
    _deac_receipt_json_type(
        _type _exists "${label}" "${document}" ${ARGN})
    if(NOT _exists)
        set(${output} "[]" PARENT_SCOPE)
        return()
    endif()
    if(NOT _type STREQUAL "ARRAY")
        message(FATAL_ERROR "build-receipt ${label} must be an array")
    endif()
    _deac_receipt_json_length(_length "${label}" "${document}" ${ARGN})
    set(_json "[")
    set(_separator "")
    if(_length GREATER 0)
        math(EXPR _last "${_length} - 1")
        foreach(_index RANGE 0 ${_last})
            _deac_receipt_json_get(
                _fragment "${label}[${_index}].fragment"
                "${document}" ${ARGN} ${_index} fragment)
            _deac_receipt_json_quote(_fragment_json "${_fragment}")
            _deac_receipt_optional_string(
                _role_json "${label}[${_index}].role"
                "${document}" ${ARGN} ${_index} role)
            string(APPEND _json
                "${_separator}{\"fragment\":${_fragment_json},\"role\":${_role_json}}")
            set(_separator ",")
        endforeach()
    endif()
    string(APPEND _json "]")
    set(${output} "${_json}" PARENT_SCOPE)
endfunction()

function(_deac_receipt_named_array output label field document)
    _deac_receipt_json_type(
        _type _exists "${label}" "${document}" ${ARGN})
    if(NOT _exists)
        set(${output} "[]" PARENT_SCOPE)
        return()
    endif()
    if(NOT _type STREQUAL "ARRAY")
        message(FATAL_ERROR "build-receipt ${label} must be an array")
    endif()
    _deac_receipt_json_length(_length "${label}" "${document}" ${ARGN})
    set(_json "[")
    set(_separator "")
    if(_length GREATER 0)
        math(EXPR _last "${_length} - 1")
        foreach(_index RANGE 0 ${_last})
            _deac_receipt_json_get(
                _value "${label}[${_index}].${field}"
                "${document}" ${ARGN} ${_index} "${field}")
            _deac_receipt_json_quote(_quoted "${_value}")
            string(APPEND _json "${_separator}${_quoted}")
            set(_separator ",")
        endforeach()
    endif()
    string(APPEND _json "]")
    set(${output} "${_json}" PARENT_SCOPE)
endfunction()

function(_deac_receipt_named_path_array output label field document)
    _deac_receipt_json_type(
        _type _exists "${label}" "${document}" ${ARGN})
    if(NOT _exists)
        set(${output} "[]" PARENT_SCOPE)
        return()
    endif()
    if(NOT _type STREQUAL "ARRAY")
        message(FATAL_ERROR "build-receipt ${label} must be an array")
    endif()
    _deac_receipt_json_length(_length "${label}" "${document}" ${ARGN})
    set(_json "[")
    set(_separator "")
    if(_length GREATER 0)
        math(EXPR _last "${_length} - 1")
        foreach(_index RANGE 0 ${_last})
            _deac_receipt_json_get(
                _value "${label}[${_index}].${field}"
                "${document}" ${ARGN} ${_index} "${field}")
            _deac_receipt_normalize_path(_value "${_value}")
            _deac_receipt_json_quote(_quoted "${_value}")
            string(APPEND _json "${_separator}${_quoted}")
            set(_separator ",")
        endforeach()
    endif()
    string(APPEND _json "]")
    set(${output} "${_json}" PARENT_SCOPE)
endfunction()

function(_deac_receipt_include_array output label document)
    _deac_receipt_json_type(
        _type _exists "${label}" "${document}" ${ARGN})
    if(NOT _exists)
        set(${output} "[]" PARENT_SCOPE)
        return()
    endif()
    if(NOT _type STREQUAL "ARRAY")
        message(FATAL_ERROR "build-receipt ${label} must be an array")
    endif()
    _deac_receipt_json_length(_length "${label}" "${document}" ${ARGN})
    set(_json "[")
    set(_separator "")
    if(_length GREATER 0)
        math(EXPR _last "${_length} - 1")
        foreach(_index RANGE 0 ${_last})
            _deac_receipt_json_get(
                _path "${label}[${_index}].path"
                "${document}" ${ARGN} ${_index} path)
            _deac_receipt_normalize_path(_path "${_path}")
            _deac_receipt_json_quote(_path_json "${_path}")
            _deac_receipt_optional_boolean(
                _system_json "${label}[${_index}].isSystem"
                "${document}" ${ARGN} ${_index} isSystem)
            string(APPEND _json
                "${_separator}{\"path\":${_path_json},\"system\":${_system_json}}")
            set(_separator ",")
        endforeach()
    endif()
    string(APPEND _json "]")
    set(${output} "${_json}" PARENT_SCOPE)
endfunction()

function(_deac_receipt_source_array output group_index target_document)
    _deac_receipt_json_length(
        _length "compile group ${group_index} source indexes"
        "${target_document}" compileGroups ${group_index} sourceIndexes)
    set(_json "[")
    set(_separator "")
    if(_length GREATER 0)
        math(EXPR _last "${_length} - 1")
        foreach(_index RANGE 0 ${_last})
            _deac_receipt_json_get(
                _source_index "compile group source index"
                "${target_document}"
                compileGroups ${group_index} sourceIndexes ${_index})
            _deac_receipt_json_get(
                _path "compile group source path"
                "${target_document}" sources ${_source_index} path)
            _deac_receipt_normalize_path(_path "${_path}")
            _deac_receipt_json_quote(_path_json "${_path}")
            string(APPEND _json "${_separator}${_path_json}")
            set(_separator ",")
        endforeach()
    endif()
    string(APPEND _json "]")
    set(${output} "${_json}" PARENT_SCOPE)
endfunction()

function(_deac_receipt_compile_groups
        output languages_output label target_document require_fingerprint)
    _deac_receipt_json_type(
        _type _exists "${label} compile groups"
        "${target_document}" compileGroups)
    if(NOT _exists)
        set(${output} "[]" PARENT_SCOPE)
        set(${languages_output} "" PARENT_SCOPE)
        return()
    endif()
    if(NOT _type STREQUAL "ARRAY")
        message(FATAL_ERROR
            "build-receipt ${label} compile groups must be an array")
    endif()
    _deac_receipt_json_length(
        _count "${label} compile groups" "${target_document}" compileGroups)
    if(require_fingerprint AND _count LESS 1)
        message(FATAL_ERROR
            "build-receipt ${label} has no compile groups")
    endif()

    set(_json "[")
    set(_separator "")
    set(_languages)
    if(_count GREATER 0)
        math(EXPR _last "${_count} - 1")
        foreach(_group RANGE 0 ${_last})
            _deac_receipt_json_get(
                _language "${label} compile group language"
                "${target_document}" compileGroups ${_group} language)
            list(APPEND _languages "${_language}")
            _deac_receipt_json_quote(_language_json "${_language}")
            _deac_receipt_source_array(
                _sources_json ${_group} "${target_document}")
            _deac_receipt_fragment_array(
                _fragments_json "${label} compile group fragments"
                "${target_document}"
                compileGroups ${_group} compileCommandFragments)
            _deac_receipt_named_array(
                _defines_json "${label} compile group definitions" define
                "${target_document}" compileGroups ${_group} defines)
            _deac_receipt_include_array(
                _includes_json "${label} compile group includes"
                "${target_document}" compileGroups ${_group} includes)
            _deac_receipt_include_array(
                _frameworks_json "${label} compile group frameworks"
                "${target_document}" compileGroups ${_group} frameworks)
            _deac_receipt_named_path_array(
                _pch_json "${label} compile group precompiled headers" header
                "${target_document}"
                compileGroups ${_group} precompileHeaders)
            _deac_receipt_optional_string(
                _standard_json "${label} compile group language standard"
                "${target_document}"
                compileGroups ${_group} languageStandard standard)
            _deac_receipt_optional_path(
                _sysroot_json "${label} compile group sysroot"
                "${target_document}" compileGroups ${_group} sysroot path)

            if(require_fingerprint)
                set(_expected_definition
                    "DEAC_BUILD_TOOLCHAIN_FINGERPRINT_SHA256=${DEAC_BUILD_RECEIPT_TOOLCHAIN_FINGERPRINT}")
                _deac_receipt_json_type(
                    _definitions_type _definitions_exist
                    "${label} compile group definitions"
                    "${target_document}" compileGroups ${_group} defines)
                set(_fingerprint_matches 0)
                if(_definitions_exist)
                    _deac_receipt_json_length(
                        _definition_count "${label} compile group definitions"
                        "${target_document}" compileGroups ${_group} defines)
                    if(_definition_count GREATER 0)
                        math(EXPR _definition_last "${_definition_count} - 1")
                        foreach(_definition_index RANGE 0 ${_definition_last})
                            _deac_receipt_json_get(
                                _definition "${label} compile definition"
                                "${target_document}" compileGroups ${_group}
                                defines ${_definition_index} define)
                            if(_definition STREQUAL _expected_definition)
                                math(EXPR _fingerprint_matches
                                    "${_fingerprint_matches} + 1")
                            endif()
                        endforeach()
                    endif()
                endif()
                if(NOT _fingerprint_matches EQUAL 1)
                    message(FATAL_ERROR
                        "build-receipt ${label} compile group does not contain "
                        "exactly one configured toolchain fingerprint")
                endif()
            endif()

            string(APPEND _json
                "${_separator}{\"command_fragments\":${_fragments_json},"
                "\"definitions\":${_defines_json},"
                "\"frameworks\":${_frameworks_json},"
                "\"includes\":${_includes_json},"
                "\"language\":${_language_json},"
                "\"language_standard\":${_standard_json},"
                "\"precompiled_headers\":${_pch_json},"
                "\"sources\":${_sources_json},"
                "\"sysroot\":${_sysroot_json}}")
            set(_separator ",")
        endforeach()
    endif()
    string(APPEND _json "]")
    list(REMOVE_DUPLICATES _languages)
    list(SORT _languages)
    set(${output} "${_json}" PARENT_SCOPE)
    set(${languages_output} "${_languages}" PARENT_SCOPE)
endfunction()

function(_deac_receipt_link output language_output label target_document required)
    _deac_receipt_json_type(
        _type _exists "${label} link" "${target_document}" link)
    if(NOT _exists)
        if(required)
            message(FATAL_ERROR "build-receipt ${label} has no link object")
        endif()
        set(${output} "null" PARENT_SCOPE)
        set(${language_output} "" PARENT_SCOPE)
        return()
    endif()
    if(NOT _type STREQUAL "OBJECT")
        message(FATAL_ERROR "build-receipt ${label} link must be an object")
    endif()
    _deac_receipt_json_get(
        _language "${label} link language" "${target_document}" link language)
    _deac_receipt_json_quote(_language_json "${_language}")
    _deac_receipt_fragment_array(
        _fragments_json "${label} link fragments"
        "${target_document}" link commandFragments)
    _deac_receipt_optional_boolean(
        _lto_json "${label} link LTO" "${target_document}" link lto)
    _deac_receipt_optional_path(
        _sysroot_json "${label} link sysroot"
        "${target_document}" link sysroot path)
    string(CONCAT _json
        "{\"command_fragments\":" "${_fragments_json}" ","
        "\"language\":" "${_language_json}" ","
        "\"lto\":" "${_lto_json}" ","
        "\"sysroot\":" "${_sysroot_json}" "}")
    set(${output} "${_json}" PARENT_SCOPE)
    set(${language_output} "${_language}" PARENT_SCOPE)
endfunction()

function(_deac_receipt_archive output label target_document)
    _deac_receipt_json_type(
        _type _exists "${label} archive" "${target_document}" archive)
    if(NOT _exists)
        set(${output} "null" PARENT_SCOPE)
        return()
    endif()
    if(NOT _type STREQUAL "OBJECT")
        message(FATAL_ERROR "build-receipt ${label} archive must be an object")
    endif()
    _deac_receipt_fragment_array(
        _fragments_json "${label} archive fragments"
        "${target_document}" archive commandFragments)
    _deac_receipt_optional_boolean(
        _lto_json "${label} archive LTO" "${target_document}" archive lto)
    set(_json
        "{\"command_fragments\":${_fragments_json},\"lto\":${_lto_json}}")
    set(${output} "${_json}" PARENT_SCOPE)
endfunction()

function(_deac_receipt_cache_entry output found name cache_document)
    _deac_receipt_json_length(_length "cache entries" "${cache_document}" entries)
    set(_match_count 0)
    if(_length GREATER 0)
        math(EXPR _last "${_length} - 1")
        foreach(_index RANGE 0 ${_last})
            _deac_receipt_json_get(
                _candidate "cache entry name"
                "${cache_document}" entries ${_index} name)
            if(_candidate STREQUAL "${name}")
                math(EXPR _match_count "${_match_count} + 1")
                _deac_receipt_json_get(
                    _type "cache entry ${name} type"
                    "${cache_document}" entries ${_index} type)
                _deac_receipt_json_get(
                    _value "cache entry ${name} value"
                    "${cache_document}" entries ${_index} value)
            endif()
        endforeach()
    endif()
    if(_match_count GREATER 1)
        message(FATAL_ERROR "build-receipt cache contains duplicate ${name}")
    endif()
    if(_match_count EQUAL 0)
        set(${output}
            "{\"name\":\"${name}\",\"type\":null,\"value\":null}"
            PARENT_SCOPE)
        set(${found} false PARENT_SCOPE)
        return()
    endif()
    _deac_receipt_json_quote(_name_json "${name}")
    _deac_receipt_json_quote(_type_json "${_type}")
    set(_canonical_value "${_value}")
    if(NOT _canonical_value STREQUAL "" AND
            NOT _canonical_value MATCHES "-NOTFOUND$")
        if(_type MATCHES "^(FILEPATH|PATH)$" OR
                name STREQUAL "CMAKE_PREFIX_PATH")
            _deac_receipt_normalize_path_list(
                _canonical_value "${_canonical_value}")
        elseif(name MATCHES
                "^(CMAKE_GENERATOR_INSTANCE|CMAKE_HOME_DIRECTORY)$")
            _deac_receipt_normalize_path(
                _canonical_value "${_canonical_value}")
        endif()
    endif()
    _deac_receipt_json_quote(_value_json "${_canonical_value}")
    set(${output}
        "{\"name\":${_name_json},\"type\":${_type_json},\"value\":${_value_json}}"
        PARENT_SCOPE)
    set(${found} true PARENT_SCOPE)
    set(${name}_VALUE "${_value}" PARENT_SCOPE)
endfunction()

include("${CMAKE_CURRENT_LIST_DIR}/DeacBuildIdentity.cmake")
_deac_compute_build_identity("${DEAC_BUILD_RECEIPT_SOURCE_ROOT}")

foreach(_path_variable
        DEAC_BUILD_RECEIPT_SOURCE_ROOT
        DEAC_BUILD_RECEIPT_CMAKE_SOURCE_ROOT
        DEAC_BUILD_RECEIPT_BUILD_ROOT
        DEAC_BUILD_RECEIPT_REPLY_DIRECTORY)
    cmake_path(NORMAL_PATH ${_path_variable})
endforeach()

_deac_receipt_latest_index(
    _index_file "${DEAC_BUILD_RECEIPT_REPLY_DIRECTORY}")
_deac_receipt_read_json(_index "File API index" "${_index_file}")
_deac_receipt_reply_file(
    _codemodel_file "codemodel" "codemodel-v2" "codemodel" 2 "${_index}")
_deac_receipt_reply_file(
    _toolchains_file "toolchains" "toolchains-v1" "toolchains" 1 "${_index}")
_deac_receipt_reply_file(
    _cache_file "cache" "cache-v2" "cache" 2 "${_index}")
_deac_receipt_read_json(_codemodel "File API codemodel" "${_codemodel_file}")
_deac_receipt_read_json(_toolchains "File API toolchains" "${_toolchains_file}")
_deac_receipt_read_json(_cache "File API cache" "${_cache_file}")

_deac_receipt_json_get(_codemodel_major "codemodel major" "${_codemodel}" version major)
_deac_receipt_json_get(_codemodel_minor "codemodel minor" "${_codemodel}" version minor)
if(NOT _codemodel_major EQUAL 2 OR _codemodel_minor LESS 6)
    message(FATAL_ERROR
        "build receipt requires File API codemodel 2.6 or newer")
endif()
_deac_receipt_json_get(_toolchains_major "toolchains major" "${_toolchains}" version major)
if(NOT _toolchains_major EQUAL 1)
    message(FATAL_ERROR "unsupported File API toolchains schema")
endif()
_deac_receipt_json_get(_cache_major "cache major" "${_cache}" version major)
if(NOT _cache_major EQUAL 2)
    message(FATAL_ERROR "unsupported File API cache schema")
endif()

_deac_receipt_json_get(_model_source "codemodel source root" "${_codemodel}" paths source)
_deac_receipt_json_get(_model_build "codemodel build root" "${_codemodel}" paths build)
cmake_path(NORMAL_PATH _model_source)
cmake_path(NORMAL_PATH _model_build)
if(NOT _model_source STREQUAL DEAC_BUILD_RECEIPT_CMAKE_SOURCE_ROOT OR
        NOT _model_build STREQUAL DEAC_BUILD_RECEIPT_BUILD_ROOT)
    message(FATAL_ERROR
        "build-receipt File API roots disagree with the configured build")
endif()

_deac_receipt_json_length(
    _configuration_count "codemodel configurations"
    "${_codemodel}" configurations)
set(_configuration_index "")
set(_configuration_matches 0)
if(_configuration_count GREATER 0)
    math(EXPR _configuration_last "${_configuration_count} - 1")
    foreach(_index RANGE 0 ${_configuration_last})
        _deac_receipt_json_get(
            _name "configuration name" "${_codemodel}" configurations ${_index} name)
        if(_name STREQUAL DEAC_BUILD_RECEIPT_CONFIGURATION)
            set(_configuration_index ${_index})
            math(EXPR _configuration_matches "${_configuration_matches} + 1")
        endif()
    endforeach()
endif()
if(NOT _configuration_matches EQUAL 1)
    message(FATAL_ERROR
        "build-receipt configuration ${DEAC_BUILD_RECEIPT_CONFIGURATION} "
        "matched ${_configuration_matches} File API configurations")
endif()

_deac_receipt_json_length(
    _target_count "codemodel targets" "${_codemodel}"
    configurations ${_configuration_index} targets)
set(_target_json_file "")
set(_target_matches 0)
if(_target_count GREATER 0)
    math(EXPR _target_last "${_target_count} - 1")
    foreach(_index RANGE 0 ${_target_last})
        _deac_receipt_json_get(
            _name "target name" "${_codemodel}"
            configurations ${_configuration_index} targets ${_index} name)
        if(_name STREQUAL DEAC_BUILD_RECEIPT_TARGET)
            _deac_receipt_json_get(
                _target_json_file "target JSON file" "${_codemodel}"
                configurations ${_configuration_index} targets ${_index} jsonFile)
            math(EXPR _target_matches "${_target_matches} + 1")
        endif()
    endforeach()
endif()
if(NOT _target_matches EQUAL 1)
    message(FATAL_ERROR
        "build-receipt target ${DEAC_BUILD_RECEIPT_TARGET} matched "
        "${_target_matches} File API targets")
endif()
_deac_receipt_read_json(
    _target "target codemodel"
    "${DEAC_BUILD_RECEIPT_REPLY_DIRECTORY}/${_target_json_file}")
_deac_receipt_json_get(_target_type "target type" "${_target}" type)
if(NOT _target_type STREQUAL "EXECUTABLE")
    message(FATAL_ERROR "build-receipt target is not an executable")
endif()
_deac_receipt_json_get(_target_name "target name" "${_target}" name)
_deac_receipt_json_get(_target_disk_name "target disk name" "${_target}" nameOnDisk)
if(NOT _target_name STREQUAL DEAC_BUILD_RECEIPT_TARGET)
    message(FATAL_ERROR "build-receipt target object name changed")
endif()

string(REPLACE "," ";" _cache_keys "${DEAC_BUILD_RECEIPT_CACHE_KEYS}")
set(_cache_json "[")
set(_separator "")
foreach(_key IN LISTS _cache_keys)
    _deac_receipt_cache_entry(_entry _found "${_key}" "${_cache}")
    string(APPEND _cache_json "${_separator}${_entry}")
    set(_separator ",")
endforeach()
string(APPEND _cache_json "]")
if(NOT DEFINED GPU_BACKEND_VALUE OR
        NOT GPU_BACKEND_VALUE STREQUAL DEAC_BUILD_RECEIPT_BACKEND)
    message(FATAL_ERROR
        "build-receipt backend disagrees with the generated CMake cache")
endif()

_deac_receipt_compile_groups(
    _compile_groups_json _target_compile_languages
    "target ${_target_name}" "${_target}" true)
_deac_receipt_link(
    _link_json _link_language "target ${_target_name}" "${_target}" true)
set(_used_languages ${_target_compile_languages} "${_link_language}")

_deac_receipt_json_type(
    _dependencies_type _dependencies_exist
    "target ${_target_name} dependencies" "${_target}" dependencies)
set(_dependency_ids)
if(_dependencies_exist)
    if(NOT _dependencies_type STREQUAL "ARRAY")
        message(FATAL_ERROR "build-receipt target dependencies must be an array")
    endif()
    _deac_receipt_json_length(
        _dependency_count "target dependencies" "${_target}" dependencies)
    if(_dependency_count GREATER 0)
        math(EXPR _dependency_last "${_dependency_count} - 1")
        foreach(_dependency_index RANGE 0 ${_dependency_last})
            _deac_receipt_json_get(
                _dependency_id "target dependency ID" "${_target}"
                dependencies ${_dependency_index} id)
            list(APPEND _dependency_ids "${_dependency_id}")
        endforeach()
    endif()
endif()
list(REMOVE_DUPLICATES _dependency_ids)

set(_actual_dependency_names)
foreach(_dependency_id IN LISTS _dependency_ids)
    set(_dependency_matches 0)
    if(_target_count GREATER 0)
        foreach(_candidate_index RANGE 0 ${_target_last})
            _deac_receipt_json_get(
                _candidate_id "dependency candidate ID" "${_codemodel}"
                configurations ${_configuration_index}
                targets ${_candidate_index} id)
            if(_candidate_id STREQUAL "${_dependency_id}")
                _deac_receipt_json_get(
                    _candidate_name "dependency candidate name" "${_codemodel}"
                    configurations ${_configuration_index}
                    targets ${_candidate_index} name)
                list(APPEND _actual_dependency_names "${_candidate_name}")
                math(EXPR _dependency_matches "${_dependency_matches} + 1")
            endif()
        endforeach()
    endif()
    if(NOT _dependency_matches EQUAL 1)
        message(FATAL_ERROR
            "build-receipt dependency ${_dependency_id} matched "
            "${_dependency_matches} configuration targets")
    endif()
endforeach()
list(SORT _actual_dependency_names)
string(REPLACE "," ";" _expected_dependency_names
    "${DEAC_BUILD_RECEIPT_DEPENDENCY_TARGETS}")
list(SORT _expected_dependency_names)
if(NOT "${_actual_dependency_names}" STREQUAL
        "${_expected_dependency_names}")
    message(FATAL_ERROR
        "build-receipt target dependencies changed: expected "
        "[${_expected_dependency_names}], got [${_actual_dependency_names}]")
endif()

set(_dependencies_json "[")
set(_dependency_separator "")
foreach(_expected_name IN LISTS _expected_dependency_names)
    set(_dependency_json_file "")
    set(_dependency_matches 0)
    if(_target_count GREATER 0)
        foreach(_candidate_index RANGE 0 ${_target_last})
            _deac_receipt_json_get(
                _candidate_id "dependency candidate ID" "${_codemodel}"
                configurations ${_configuration_index}
                targets ${_candidate_index} id)
            _deac_receipt_json_get(
                _candidate_name "dependency candidate name" "${_codemodel}"
                configurations ${_configuration_index}
                targets ${_candidate_index} name)
            if(_candidate_name STREQUAL "${_expected_name}" AND
                    _candidate_id IN_LIST _dependency_ids)
                _deac_receipt_json_get(
                    _dependency_json_file "dependency target JSON file"
                    "${_codemodel}" configurations ${_configuration_index}
                    targets ${_candidate_index} jsonFile)
                math(EXPR _dependency_matches "${_dependency_matches} + 1")
            endif()
        endforeach()
    endif()
    if(NOT _dependency_matches EQUAL 1)
        message(FATAL_ERROR
            "build-receipt dependency target ${_expected_name} matched "
            "${_dependency_matches} File API targets")
    endif()
    _deac_receipt_read_json(
        _dependency "dependency target ${_expected_name}"
        "${DEAC_BUILD_RECEIPT_REPLY_DIRECTORY}/${_dependency_json_file}")
    _deac_receipt_json_get(
        _dependency_name "dependency target name" "${_dependency}" name)
    _deac_receipt_json_get(
        _dependency_type "dependency target type" "${_dependency}" type)
    if(NOT _dependency_name STREQUAL "${_expected_name}")
        message(FATAL_ERROR "build-receipt dependency target name changed")
    endif()
    _deac_receipt_compile_groups(
        _dependency_compile_groups _dependency_languages
        "dependency ${_dependency_name}" "${_dependency}" false)
    _deac_receipt_link(
        _dependency_link _dependency_link_language
        "dependency ${_dependency_name}" "${_dependency}" false)
    _deac_receipt_archive(
        _dependency_archive "dependency ${_dependency_name}" "${_dependency}")
    list(APPEND _used_languages
        ${_dependency_languages} "${_dependency_link_language}")
    _deac_receipt_json_quote(_dependency_name_json "${_dependency_name}")
    _deac_receipt_json_quote(_dependency_type_json "${_dependency_type}")
    string(APPEND _dependencies_json
        "${_dependency_separator}{\"archive\":${_dependency_archive},"
        "\"compile_groups\":${_dependency_compile_groups},"
        "\"link\":${_dependency_link},"
        "\"name\":${_dependency_name_json},"
        "\"type\":${_dependency_type_json}}")
    set(_dependency_separator ",")
endforeach()
string(APPEND _dependencies_json "]")

list(FILTER _used_languages EXCLUDE REGEX "^$")
list(REMOVE_DUPLICATES _used_languages)
list(SORT _used_languages)

string(REPLACE "," ";" _configured_languages "${DEAC_BUILD_RECEIPT_LANGUAGES}")
foreach(_language IN LISTS _used_languages)
    if(NOT _language IN_LIST _configured_languages)
        message(FATAL_ERROR
            "build-receipt target uses unbound toolchain language ${_language}")
    endif()
endforeach()

_deac_receipt_json_length(
    _toolchain_count "toolchains" "${_toolchains}" toolchains)
set(_toolchains_json "[")
set(_separator "")
foreach(_language IN LISTS _used_languages)
    set(_toolchain_index "")
    set(_matches 0)
    if(_toolchain_count GREATER 0)
        math(EXPR _toolchain_last "${_toolchain_count} - 1")
        foreach(_index RANGE 0 ${_toolchain_last})
            _deac_receipt_json_get(
                _candidate "toolchain language" "${_toolchains}"
                toolchains ${_index} language)
            if(_candidate STREQUAL "${_language}")
                set(_toolchain_index ${_index})
                math(EXPR _matches "${_matches} + 1")
            endif()
        endforeach()
    endif()
    if(NOT _matches EQUAL 1)
        message(FATAL_ERROR
            "build-receipt language ${_language} matched ${_matches} toolchains")
    endif()
    _deac_receipt_json_get(
        _compiler_path "${_language} compiler path" "${_toolchains}"
        toolchains ${_toolchain_index} compiler path)
    _deac_receipt_json_get(
        _compiler_id "${_language} compiler ID" "${_toolchains}"
        toolchains ${_toolchain_index} compiler id)
    _deac_receipt_json_get(
        _compiler_version "${_language} compiler version" "${_toolchains}"
        toolchains ${_toolchain_index} compiler version)
    set(_configured_path_variable
        "DEAC_BUILD_RECEIPT_CONFIGURED_${_language}_PATH")
    set(_configured_real_variable
        "DEAC_BUILD_RECEIPT_CONFIGURED_${_language}_REAL_PATH")
    set(_configured_sha_variable
        "DEAC_BUILD_RECEIPT_CONFIGURED_${_language}_SHA256")
    set(_configured_id_variable
        "DEAC_BUILD_RECEIPT_CONFIGURED_${_language}_ID")
    set(_configured_version_variable
        "DEAC_BUILD_RECEIPT_CONFIGURED_${_language}_VERSION")
    foreach(_variable
            ${_configured_path_variable}
            ${_configured_real_variable}
            ${_configured_sha_variable}
            ${_configured_id_variable}
            ${_configured_version_variable})
        if(NOT DEFINED ${_variable})
            message(FATAL_ERROR "build-receipt missing ${_variable}")
        endif()
    endforeach()
    if(NOT _compiler_path STREQUAL "${${_configured_path_variable}}" OR
            NOT _compiler_id STREQUAL "${${_configured_id_variable}}" OR
            NOT _compiler_version STREQUAL "${${_configured_version_variable}}")
        message(FATAL_ERROR
            "build-receipt ${_language} toolchain disagrees with configure-time identity")
    endif()
    get_filename_component(_compiler_real "${_compiler_path}" REALPATH)
    if(NOT EXISTS "${_compiler_real}" OR IS_DIRECTORY "${_compiler_real}")
        message(FATAL_ERROR
            "build-receipt ${_language} compiler is no longer a regular file")
    endif()
    file(SHA256 "${_compiler_real}" _compiler_sha256)
    if(NOT _compiler_real STREQUAL "${${_configured_real_variable}}" OR
            NOT _compiler_sha256 STREQUAL "${${_configured_sha_variable}}")
        message(FATAL_ERROR
            "build-receipt ${_language} compiler changed after configuration; "
            "rerun CMake before building")
    endif()
    _deac_receipt_json_quote(_language_json "${_language}")
    _deac_receipt_normalize_path(_compiler_path_json_value "${_compiler_path}")
    _deac_receipt_normalize_path(_compiler_real_json_value "${_compiler_real}")
    _deac_receipt_json_quote(_path_json "${_compiler_path_json_value}")
    _deac_receipt_json_quote(_real_path_json "${_compiler_real_json_value}")
    _deac_receipt_json_quote(_sha_json "${_compiler_sha256}")
    _deac_receipt_json_quote(_id_json "${_compiler_id}")
    _deac_receipt_json_quote(_version_json "${_compiler_version}")
    _deac_receipt_optional_string(
        _target_json "${_language} compiler target" "${_toolchains}"
        toolchains ${_toolchain_index} compiler target)
    _deac_receipt_path_array(
        _implicit_includes "${_language} implicit include directories"
        "${_toolchains}"
        toolchains ${_toolchain_index} compiler implicit includeDirectories)
    _deac_receipt_path_array(
        _implicit_link_dirs "${_language} implicit link directories"
        "${_toolchains}"
        toolchains ${_toolchain_index} compiler implicit linkDirectories)
    _deac_receipt_path_array(
        _implicit_framework_dirs "${_language} implicit framework directories"
        "${_toolchains}"
        toolchains ${_toolchain_index} compiler implicit linkFrameworkDirectories)
    _deac_receipt_string_array(
        _implicit_libraries "${_language} implicit link libraries"
        "${_toolchains}"
        toolchains ${_toolchain_index} compiler implicit linkLibraries)
    string(APPEND _toolchains_json
        "${_separator}{\"compiler\":{"
        "\"id\":${_id_json},"
        "\"implicit_framework_directories\":${_implicit_framework_dirs},"
        "\"implicit_include_directories\":${_implicit_includes},"
        "\"implicit_link_directories\":${_implicit_link_dirs},"
        "\"implicit_link_libraries\":${_implicit_libraries},"
        "\"path\":${_path_json},"
        "\"real_path\":${_real_path_json},"
        "\"sha256\":${_sha_json},"
        "\"target\":${_target_json},"
        "\"version\":${_version_json}},"
        "\"language\":${_language_json}}")
    set(_separator ",")
endforeach()
string(APPEND _toolchains_json "]")

get_filename_component(_cmake_real "${DEAC_BUILD_RECEIPT_CMAKE_PATH}" REALPATH)
if(NOT EXISTS "${_cmake_real}" OR IS_DIRECTORY "${_cmake_real}")
    message(FATAL_ERROR "build-receipt CMake executable is unavailable")
endif()
file(SHA256 "${_cmake_real}" _cmake_sha256)
if(NOT _cmake_real STREQUAL DEAC_BUILD_RECEIPT_CMAKE_REAL_PATH OR
        NOT _cmake_sha256 STREQUAL DEAC_BUILD_RECEIPT_CMAKE_SHA256)
    message(FATAL_ERROR
        "build-receipt CMake executable changed after configuration")
endif()
_deac_receipt_json_get(_index_cmake_path "index CMake path" "${_index}" cmake paths cmake)
get_filename_component(_index_cmake_real "${_index_cmake_path}" REALPATH)
if(NOT _index_cmake_real STREQUAL _cmake_real)
    message(FATAL_ERROR "build-receipt File API was generated by another CMake")
endif()
_deac_receipt_json_get(
    _cmake_version "index CMake version" "${_index}" cmake version string)
_deac_receipt_json_get(
    _generator_name "index generator" "${_index}" cmake generator name)
_deac_receipt_json_get(
    _generator_multi "index multi-config flag" "${_index}" cmake generator multiConfig)
if(_generator_multi)
    set(_generator_multi_json true)
else()
    set(_generator_multi_json false)
endif()
_deac_receipt_optional_string(
    _generator_platform_json "index generator platform" "${_index}"
    cmake generator platform)
_deac_receipt_optional_string(
    _generator_toolset_json "index generator toolset" "${_index}"
    cmake generator toolset)
_deac_receipt_normalize_path(
    _cmake_path_json_value "${DEAC_BUILD_RECEIPT_CMAKE_PATH}")
_deac_receipt_normalize_path(_cmake_real_json_value "${_cmake_real}")
_deac_receipt_json_quote(_cmake_path_json "${_cmake_path_json_value}")
_deac_receipt_json_quote(_cmake_real_json "${_cmake_real_json_value}")
_deac_receipt_json_quote(_cmake_sha_json "${_cmake_sha256}")
_deac_receipt_json_quote(_cmake_version_json "${_cmake_version}")
_deac_receipt_json_quote(_generator_name_json "${_generator_name}")
_deac_receipt_json_quote(_configuration_json "${DEAC_BUILD_RECEIPT_CONFIGURATION}")
string(CONCAT _build_system_json
    "{\"cmake\":{\"path\":" "${_cmake_path_json}" ","
    "\"real_path\":" "${_cmake_real_json}" ","
    "\"sha256\":" "${_cmake_sha_json}" ","
    "\"version\":" "${_cmake_version_json}" "},"
    "\"configuration\":" "${_configuration_json}" ","
    "\"generator\":{\"multi_config\":" "${_generator_multi_json}" ","
    "\"name\":" "${_generator_name_json}" ","
    "\"platform\":" "${_generator_platform_json}" ","
    "\"toolset\":" "${_generator_toolset_json}" "}}")

_deac_receipt_json_quote(_backend_json "${DEAC_BUILD_RECEIPT_BACKEND}")
_deac_receipt_json_quote(_target_name_json "${_target_name}")
_deac_receipt_json_quote(_target_disk_json "${_target_disk_name}")
_deac_receipt_json_quote(
    _toolchain_fingerprint_json
    "${DEAC_BUILD_RECEIPT_TOOLCHAIN_FINGERPRINT}")
string(CONCAT _payload
    "{\"backend\":" "${_backend_json}" ","
    "\"build_system\":" "${_build_system_json}" ","
    "\"cache_entries\":" "${_cache_json}" ","
    "\"compile_groups\":" "${_compile_groups_json}" ","
    "\"link\":" "${_link_json}" ","
    "\"source_identity\":" "${DEAC_BUILD_IDENTITY_CANONICAL_JSON}" ","
    "\"target\":{\"name\":" "${_target_name_json}" ","
    "\"name_on_disk\":" "${_target_disk_json}" ","
    "\"toolchain_fingerprint_sha256\":"
    "${_toolchain_fingerprint_json}" ","
    "\"type\":\"EXECUTABLE\"},"
    "\"target_dependencies\":" "${_dependencies_json}" ","
    "\"toolchains\":" "${_toolchains_json}" "}")
string(SHA256 _payload_sha256 "${_payload}")
string(CONCAT _canonical_json
    "{\"schema_version\":1,\"receipt_sha256\":\""
    "${_payload_sha256}" "\",\"receipt\":" "${_payload}" "}")
string(LENGTH "${_canonical_json}" _receipt_length)
if(_receipt_length GREATER 1048576)
    message(FATAL_ERROR "canonical build receipt exceeds 1 MiB")
endif()

set(DEAC_BUILD_RECEIPT_CXX_JSON "${_canonical_json}")
string(REPLACE "\\" "\\\\" DEAC_BUILD_RECEIPT_CXX_JSON
    "${DEAC_BUILD_RECEIPT_CXX_JSON}")
string(REPLACE "\"" "\\\"" DEAC_BUILD_RECEIPT_CXX_JSON
    "${DEAC_BUILD_RECEIPT_CXX_JSON}")
get_filename_component(
    _source_output_directory "${DEAC_BUILD_RECEIPT_OUTPUT_SOURCE}" DIRECTORY)
get_filename_component(
    _receipt_output_directory "${DEAC_BUILD_RECEIPT_OUTPUT_RECEIPT}" DIRECTORY)
file(MAKE_DIRECTORY "${_source_output_directory}" "${_receipt_output_directory}")
configure_file(
    "${CMAKE_CURRENT_LIST_DIR}/deac_build_receipt_data.cpp.in"
    "${DEAC_BUILD_RECEIPT_OUTPUT_SOURCE}"
    @ONLY)
file(WRITE "${DEAC_BUILD_RECEIPT_OUTPUT_RECEIPT}" "${_canonical_json}\n")
