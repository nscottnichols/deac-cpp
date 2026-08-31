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

function(_deac_build_receipt_snapshot_tool
        language output_arguments output_fingerprint_material)
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
endfunction()

function(_deac_build_receipt_reject_launchers target_name)
    foreach(_language CXX CUDA)
        foreach(_suffix COMPILER_LAUNCHER LINKER_LAUNCHER)
            set(_variable "CMAKE_${_language}_${_suffix}")
            if(DEFINED ${_variable} AND NOT "${${_variable}}" STREQUAL "")
                message(FATAL_ERROR
                    "build receipts do not yet support ${_variable}")
            endif()
            get_target_property(_launcher "${target_name}"
                "${_language}_${_suffix}")
            if(_launcher AND NOT _launcher MATCHES "-NOTFOUND$")
                message(FATAL_ERROR
                    "build receipts do not yet support target "
                    "${target_name} ${_language}_${_suffix}")
            endif()
        endforeach()
        foreach(_suffix
                CLANG_TIDY CPPCHECK CPPLINT INCLUDE_WHAT_YOU_USE)
            get_target_property(_launcher "${target_name}"
                "${_language}_${_suffix}")
            if(_launcher AND NOT _launcher MATCHES "-NOTFOUND$")
                message(FATAL_ERROR
                    "build receipts do not yet support target "
                    "${target_name} ${_language}_${_suffix}")
            endif()
        endforeach()
        set(_compiler_arg1 "CMAKE_${_language}_COMPILER_ARG1")
        if(DEFINED ${_compiler_arg1} AND
                NOT "${${_compiler_arg1}}" STREQUAL "")
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

    get_target_property(
        _interprocedural_optimization
        "${target_name}" INTERPROCEDURAL_OPTIMIZATION)
    if(_interprocedural_optimization AND
            NOT _interprocedural_optimization MATCHES "-NOTFOUND$")
        message(FATAL_ERROR
            "build receipts do not yet support target ${target_name} "
            "INTERPROCEDURAL_OPTIMIZATION")
    endif()

    foreach(_launcher_property
            RULE_LAUNCH_COMPILE RULE_LAUNCH_CUSTOM RULE_LAUNCH_LINK)
        get_target_property(
            _launcher_value "${target_name}" "${_launcher_property}")
        if(_launcher_value AND NOT _launcher_value MATCHES "-NOTFOUND$")
            message(FATAL_ERROR
                "build receipts do not yet support target ${target_name} "
                "${_launcher_property}")
        endif()
        foreach(_scope GLOBAL DIRECTORY)
            get_property(_launcher_value ${_scope}
                PROPERTY "${_launcher_property}")
            if(NOT "${_launcher_value}" STREQUAL "")
                message(FATAL_ERROR
                    "build receipts do not yet support ${_scope} "
                    "${_launcher_property}")
            endif()
        endforeach()
    endforeach()
endfunction()

function(_deac_build_receipt_reject_rule_override_routes)
    foreach(_variable
            CMAKE_TOOLCHAIN_FILE
            CMAKE_USER_MAKE_RULES_OVERRIDE
            CMAKE_USER_MAKE_RULES_OVERRIDE_CXX
            CMAKE_USER_MAKE_RULES_OVERRIDE_CUDA
            CMAKE_USER_MAKE_RULES_OVERRIDE_HIP
            CMAKE_PROJECT_INCLUDE
            CMAKE_PROJECT_INCLUDE_BEFORE
            CMAKE_PROJECT_TOP_LEVEL_INCLUDES)
        if(DEFINED ${_variable} AND NOT "${${_variable}}" STREQUAL "")
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
    foreach(_suffix INCLUDE INCLUDE_BEFORE)
        set(_variable "CMAKE_PROJECT_${PROJECT_NAME}_${_suffix}")
        if(DEFINED ${_variable} AND NOT "${${_variable}}" STREQUAL "")
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
        language output_fingerprint_material)
    set(_material "")
    foreach(_rule COMPILE_OBJECT LINK_EXECUTABLE)
        set(_variable "CMAKE_${language}_${_rule}")
        if(NOT DEFINED ${_variable} OR "${${_variable}}" STREQUAL "")
            message(FATAL_ERROR "build receipt requires ${_variable}")
        endif()
        set(_template "${${_variable}}")
        if(_template MATCHES "[;\r\n]")
            message(FATAL_ERROR
                "build receipt does not support multi-command ${_variable}")
        endif()
        if(_rule STREQUAL "COMPILE_OBJECT")
            set(_allowed_prefix "<CMAKE_${language}_COMPILER>")
            foreach(_placeholder DEFINES INCLUDES SOURCE OBJECT FLAGS)
                if(NOT _template MATCHES "<${_placeholder}>")
                    message(FATAL_ERROR
                        "build-receipt ${_variable} omits <${_placeholder}>")
                endif()
            endforeach()
        elseif(language STREQUAL "CUDA")
            if(_template MATCHES "^<CMAKE_CUDA_COMPILER>( |$)")
                set(_allowed_prefix "<CMAKE_CUDA_COMPILER>")
            elseif(_template MATCHES "^<CMAKE_CUDA_HOST_LINK_LAUNCHER>( |$)")
                set(_allowed_prefix "<CMAKE_CUDA_HOST_LINK_LAUNCHER>")
            else()
                message(FATAL_ERROR
                    "build-receipt ${_variable} has an unsupported launcher")
            endif()
        else()
            set(_allowed_prefix "<CMAKE_${language}_COMPILER>")
        endif()
        string(FIND "${_template}" "${_allowed_prefix}" _prefix_position)
        if(NOT _prefix_position EQUAL 0)
            message(FATAL_ERROR
                "build-receipt ${_variable} has an unsupported launcher")
        endif()
        if(_rule STREQUAL "LINK_EXECUTABLE")
            foreach(_placeholder
                    LINK_FLAGS LINK_LIBRARIES OBJECTS TARGET)
                if(NOT _template MATCHES "<${_placeholder}>")
                    message(FATAL_ERROR
                        "build-receipt ${_variable} omits <${_placeholder}>")
                endif()
            endforeach()
            if(language STREQUAL "CXX")
                foreach(_placeholder FLAGS CMAKE_CXX_LINK_FLAGS)
                    if(NOT _template MATCHES "<${_placeholder}>")
                        message(FATAL_ERROR
                            "build-receipt ${_variable} omits <${_placeholder}>")
                    endif()
                endforeach()
            endif()
        endif()
        string(LENGTH "${_template}" _length)
        string(APPEND _material
            "${language}.${_rule}:${_length}:${_template}\n")
    endforeach()

    set(_legacy_archive "CMAKE_${language}_CREATE_STATIC_LIBRARY")
    if(DEFINED ${_legacy_archive} AND NOT "${${_legacy_archive}}" STREQUAL "")
        message(FATAL_ERROR
            "build receipts do not support ${_legacy_archive}")
    endif()
    foreach(_rule ARCHIVE_CREATE ARCHIVE_APPEND ARCHIVE_FINISH)
        set(_variable "CMAKE_${language}_${_rule}")
        if(NOT DEFINED ${_variable} OR "${${_variable}}" STREQUAL "")
            message(FATAL_ERROR "build receipt requires ${_variable}")
        endif()
        set(_template "${${_variable}}")
        if(_template MATCHES "[;\r\n]")
            message(FATAL_ERROR
                "build receipt does not support multi-command ${_variable}")
        endif()
        if(_rule STREQUAL "ARCHIVE_FINISH")
            set(_allowed_prefix "<CMAKE_RANLIB>")
        else()
            set(_allowed_prefix "<CMAKE_AR>")
        endif()
        string(FIND "${_template}" "${_allowed_prefix}" _prefix_position)
        if(NOT _prefix_position EQUAL 0)
            message(FATAL_ERROR
                "build-receipt ${_variable} has an unsupported launcher")
        endif()
        if(_rule STREQUAL "ARCHIVE_FINISH")
            if(NOT _template MATCHES "<TARGET>")
                message(FATAL_ERROR
                    "build-receipt ${_variable} omits <TARGET>")
            endif()
        else()
            foreach(_placeholder LINK_FLAGS OBJECTS TARGET)
                if(NOT _template MATCHES "<${_placeholder}>")
                    message(FATAL_ERROR
                        "build-receipt ${_variable} omits <${_placeholder}>")
                endif()
            endforeach()
        endif()
        string(LENGTH "${_template}" _length)
        string(APPEND _material
            "${language}.${_rule}:${_length}:${_template}\n")
    endforeach()
    set(${output_fingerprint_material} "${_material}" PARENT_SCOPE)
endfunction()

function(deac_target_add_build_receipt target_name)
    if(CMAKE_VERSION VERSION_LESS 3.27)
        message(FATAL_ERROR
            "canonical DEAC build receipts require CMake 3.27 or newer")
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

    cmake_parse_arguments(
        PARSE_ARGV 1
        DEAC_RECEIPT
        ""
        "SOURCE_ROOT;GENERATED_DIRECTORY;IDENTITY_NAME;RECEIPT;BACKEND"
        "CACHE_KEYS;DEPENDENCY_TARGETS")
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

    list(REMOVE_DUPLICATES DEAC_RECEIPT_DEPENDENCY_TARGETS)
    list(SORT DEAC_RECEIPT_DEPENDENCY_TARGETS)
    foreach(_dependency_target IN LISTS DEAC_RECEIPT_DEPENDENCY_TARGETS)
        if(NOT TARGET "${_dependency_target}")
            message(FATAL_ERROR
                "build-receipt dependency target does not exist: "
                "${_dependency_target}")
        endif()
        _deac_build_receipt_require_plain(
            "${_dependency_target}" "build-receipt dependency target")
    endforeach()
    string(JOIN "," _dependency_targets_csv
        ${DEAC_RECEIPT_DEPENDENCY_TARGETS})

    if(CMAKE_CONFIGURATION_TYPES)
        string(FIND "${DEAC_RECEIPT_RECEIPT}" "$<CONFIG>" _config_position)
        if(_config_position EQUAL -1)
            message(FATAL_ERROR
                "multi-config build receipts require $<CONFIG> in RECEIPT")
        endif()
        set(_receipt_configurations ${CMAKE_CONFIGURATION_TYPES})
    else()
        set(_receipt_configurations "${CMAKE_BUILD_TYPE}")
    endif()
    foreach(_configuration IN LISTS _receipt_configurations)
        if(NOT _configuration MATCHES "^[A-Za-z0-9_.+-]+$" OR
                _configuration STREQUAL "." OR
                _configuration STREQUAL "..")
            message(FATAL_ERROR
                "build-receipt configuration is not path-safe: "
                "${_configuration}")
        endif()
    endforeach()

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

    _deac_build_receipt_reject_rule_override_routes()

    foreach(_validated_target
            "${target_name}" ${DEAC_RECEIPT_DEPENDENCY_TARGETS})
        _deac_build_receipt_reject_launchers("${_validated_target}")
    endforeach()

    get_property(_enabled_languages GLOBAL PROPERTY ENABLED_LANGUAGES)
    foreach(_unsupported_language CUDA HIP)
        if(_unsupported_language IN_LIST _enabled_languages)
            message(FATAL_ERROR
                "build receipts fail closed for the native CMake "
                "${_unsupported_language} language until its intermediate "
                "device-link rules are represented and compiler-gated")
        endif()
    endforeach()
    set(_receipt_languages)
    set(_tool_arguments)
    set(_toolchain_fingerprint_material "deac-build-toolchain-v1\n")
    foreach(_language CXX CUDA)
        if(_language IN_LIST _enabled_languages)
            _deac_build_receipt_snapshot_tool(
                "${_language}" _language_arguments _language_fingerprint)
            list(APPEND _receipt_languages "${_language}")
            list(APPEND _tool_arguments ${_language_arguments})
            string(APPEND _toolchain_fingerprint_material
                "${_language_fingerprint}")
            _deac_build_receipt_rule_fingerprint(
                "${_language}" _language_rule_fingerprint)
            string(APPEND _toolchain_fingerprint_material
                "${_language_rule_fingerprint}")
        endif()
    endforeach()
    if(NOT "CXX" IN_LIST _receipt_languages)
        message(FATAL_ERROR "build receipt requires the CXX language")
    endif()
    string(JOIN "," _receipt_languages_csv ${_receipt_languages})

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

    # Changing compiler or CMake bytes followed by a required reconfigure must
    # invalidate every object, even when the compiler pathname is unchanged.
    # The same definition is visible in the File API compile groups recorded
    # below, binding that invalidation key into the receipt itself.
    foreach(_fingerprinted_target
            "${target_name}" ${DEAC_RECEIPT_DEPENDENCY_TARGETS})
        get_target_property(_fingerprinted_type
            "${_fingerprinted_target}" TYPE)
        if(_fingerprinted_type MATCHES
                "^(EXECUTABLE|MODULE_LIBRARY|OBJECT_LIBRARY|SHARED_LIBRARY|STATIC_LIBRARY)$")
            target_compile_definitions("${_fingerprinted_target}" PRIVATE
                "DEAC_BUILD_TOOLCHAIN_FINGERPRINT_SHA256=${_toolchain_fingerprint}")
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

    add_custom_command(
        # The command deliberately never creates this symbolic primary output.
        # A single $<CONFIG>-qualified edge keeps Ninja Multi-Config graphs
        # isolated while making every ordinary selected-config build refresh.
        OUTPUT "${_refresh}"
        BYPRODUCTS "${_generated_source}" "${_receipt}"
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
            "-DDEAC_BUILD_RECEIPT_DEPENDENCY_TARGETS:STRING=${_dependency_targets_csv}"
            "-DDEAC_BUILD_RECEIPT_LANGUAGES:STRING=${_receipt_languages_csv}"
            "-DDEAC_BUILD_RECEIPT_TOOLCHAIN_FINGERPRINT:STRING=${_toolchain_fingerprint}"
            "-DDEAC_BUILD_RECEIPT_CMAKE_PATH:FILEPATH=${CMAKE_COMMAND}"
            "-DDEAC_BUILD_RECEIPT_CMAKE_REAL_PATH:FILEPATH=${_cmake_real}"
            "-DDEAC_BUILD_RECEIPT_CMAKE_SHA256:STRING=${_cmake_sha256}"
            "-DDEAC_BUILD_RECEIPT_OUTPUT_SOURCE:FILEPATH=${_generated_source}"
            "-DDEAC_BUILD_RECEIPT_OUTPUT_RECEIPT:FILEPATH=${_receipt}"
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
        # timestamp-only.  Object and link rules depend on the always-missing
        # token, so coarse filesystem timestamps cannot suppress a rebuild.
        OUTPUT "${_rebuild}"
        COMMAND "${CMAKE_COMMAND}" -E true
        DEPENDS "${_refresh}"
        VERBATIM)
    set_source_files_properties(
        "${_refresh}" "${_rebuild}"
        PROPERTIES GENERATED TRUE SYMBOLIC TRUE)
    set_source_files_properties(
        "${_generated_source}" PROPERTIES
        GENERATED TRUE
        OBJECT_DEPENDS "${_rebuild}")
    set_property(TARGET "${target_name}" APPEND PROPERTY
        LINK_DEPENDS "${_rebuild}")
    target_sources("${target_name}" PRIVATE
        "${_generated_source}" "${_rebuild}")
    target_include_directories("${target_name}" PRIVATE
        "${_support_directory}")

    # This catches persistent compiler/CMake replacement after receipt
    # generation but before the final link.  As with any ordinary build graph,
    # an adversarial swap-and-restore between process invocations is outside
    # CMake's attestation boundary.
    add_custom_command(TARGET "${target_name}" PRE_LINK
        COMMAND
            "${CMAKE_COMMAND}"
            "-DDEAC_BUILD_RECEIPT_LANGUAGES:STRING=${_receipt_languages_csv}"
            "-DDEAC_BUILD_RECEIPT_CMAKE_PATH:FILEPATH=${CMAKE_COMMAND}"
            "-DDEAC_BUILD_RECEIPT_CMAKE_REAL_PATH:FILEPATH=${_cmake_real}"
            "-DDEAC_BUILD_RECEIPT_CMAKE_SHA256:STRING=${_cmake_sha256}"
            ${_tool_arguments}
            -P "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/VerifyDeacBuildReceiptTools.cmake"
        COMMENT "Verifying build-receipt tool bytes for ${target_name}"
        VERBATIM)
    set_property(TARGET "${target_name}" APPEND PROPERTY
        ADDITIONAL_CLEAN_FILES "${_generated_source};${_receipt}")

    set(DEAC_BUILD_RECEIPT "${_receipt}" PARENT_SCOPE)
    set(DEAC_BUILD_RECEIPT_GENERATED_SOURCE
        "${DEAC_RECEIPT_GENERATED_DIRECTORY}/$<CONFIG>/${_receipt_identifier}.cpp"
        PARENT_SCOPE)
    set(DEAC_BUILD_RECEIPT_TOOLCHAIN_FINGERPRINT
        "${_toolchain_fingerprint}" PARENT_SCOPE)
endfunction()
