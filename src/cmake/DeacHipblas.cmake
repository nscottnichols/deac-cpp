include_guard(GLOBAL)

function(_deac_hipblas_canonical_provider target_name output)
    if(NOT TARGET "${target_name}")
        message(FATAL_ERROR
            "The hipBLAS provider target does not exist: ${target_name}.")
    endif()

    set(_deac_provider "${target_name}")
    while(TRUE)
        get_target_property(_deac_aliased_target
            "${_deac_provider}" ALIASED_TARGET)
        if(NOT _deac_aliased_target OR
                _deac_aliased_target MATCHES "-NOTFOUND$")
            break()
        endif()
        set(_deac_provider "${_deac_aliased_target}")
    endwhile()

    get_target_property(_deac_imported "${_deac_provider}" IMPORTED)
    get_target_property(_deac_provider_type "${_deac_provider}" TYPE)
    set(_deac_supported_provider_types
        STATIC_LIBRARY SHARED_LIBRARY UNKNOWN_LIBRARY)
    if(NOT _deac_imported OR NOT _deac_provider_type IN_LIST
            _deac_supported_provider_types)
        message(FATAL_ERROR
            "The hipBLAS provider must resolve to an imported STATIC, SHARED, "
            "or UNKNOWN library target: ${target_name} (resolved to "
            "${_deac_provider}, type ${_deac_provider_type}).")
    endif()
    set(${output} "${_deac_provider}" PARENT_SCOPE)
endfunction()

function(_deac_hipblas_location_property target_name property_name output)
    get_target_property(_deac_location "${target_name}" "${property_name}")
    if(NOT _deac_location OR _deac_location MATCHES "-NOTFOUND$")
        set(_deac_location "")
    endif()
    set(${output} "${_deac_location}" PARENT_SCOPE)
endfunction()

function(_deac_hipblas_imported_location target_name configuration output)
    _deac_hipblas_canonical_provider("${target_name}" _deac_provider)

    set(_deac_location "")
    if(NOT "${configuration}" STREQUAL "")
        string(TOUPPER "${configuration}" _deac_configuration_upper)
        set(_deac_map_property
            "MAP_IMPORTED_CONFIG_${_deac_configuration_upper}")
        get_property(_deac_map_is_set TARGET "${_deac_provider}"
            PROPERTY "${_deac_map_property}" SET)
        if(_deac_map_is_set)
            get_target_property(_deac_candidate_configurations
                "${_deac_provider}" "${_deac_map_property}")
            if("${_deac_candidate_configurations}" STREQUAL "")
                set(_deac_candidate_configurations "<CONFIGLESS>")
            endif()
        else()
            set(_deac_candidate_configurations
                "${_deac_configuration_upper}" "<CONFIGLESS>")
        endif()
    else()
        set(_deac_candidate_configurations "<CONFIGLESS>")
    endif()

    foreach(_deac_candidate IN LISTS _deac_candidate_configurations)
        if("${_deac_candidate}" STREQUAL "" OR
                _deac_candidate STREQUAL "<CONFIGLESS>")
            set(_deac_location_property IMPORTED_LOCATION)
        else()
            string(TOUPPER "${_deac_candidate}" _deac_candidate_upper)
            set(_deac_location_property
                "IMPORTED_LOCATION_${_deac_candidate_upper}")
        endif()
        _deac_hipblas_location_property(
            "${_deac_provider}" "${_deac_location_property}"
            _deac_candidate_location)
        if(NOT "${_deac_candidate_location}" STREQUAL "")
            set(_deac_location "${_deac_candidate_location}")
            break()
        endif()
    endforeach()

    if(_deac_location STREQUAL "" AND NOT _deac_map_is_set)
        get_target_property(_deac_imported_configurations
            "${_deac_provider}" IMPORTED_CONFIGURATIONS)
        foreach(_deac_candidate IN LISTS _deac_imported_configurations)
            string(TOUPPER "${_deac_candidate}" _deac_candidate_upper)
            _deac_hipblas_location_property(
                "${_deac_provider}"
                "IMPORTED_LOCATION_${_deac_candidate_upper}"
                _deac_candidate_location)
            if(NOT "${_deac_candidate_location}" STREQUAL "")
                set(_deac_location "${_deac_candidate_location}")
                break()
            endif()
        endforeach()
    endif()

    if(_deac_location STREQUAL "" OR
            _deac_location MATCHES "[;<>\r\n]" OR
            _deac_location MATCHES "\\$<" OR
            NOT IS_ABSOLUTE "${_deac_location}" OR
            NOT EXISTS "${_deac_location}" OR
            IS_DIRECTORY "${_deac_location}")
        message(FATAL_ERROR
            "The hipBLAS provider ${_deac_provider} has no receipt-safe regular "
            "imported artifact for configuration '${configuration}'.")
    endif()
    set(${output} "${_deac_location}" PARENT_SCOPE)
endfunction()

function(_deac_hipblas_provider_artifact target_name output)
    _deac_hipblas_canonical_provider("${target_name}" _deac_provider)
    if(CMAKE_CONFIGURATION_TYPES)
        set(_deac_locations)
        set(_deac_configuration_keys)
        foreach(_deac_configuration IN LISTS CMAKE_CONFIGURATION_TYPES)
            if("${_deac_configuration}" STREQUAL "" OR
                    _deac_configuration MATCHES "[;<>,$\r\n]")
                message(FATAL_ERROR
                    "CMAKE_CONFIGURATION_TYPES contains a receipt-unsafe "
                    "configuration name: '${_deac_configuration}'.")
            endif()
            string(TOUPPER "${_deac_configuration}"
                _deac_configuration_key)
            if(_deac_configuration_key IN_LIST _deac_configuration_keys)
                message(FATAL_ERROR
                    "CMAKE_CONFIGURATION_TYPES contains duplicate "
                    "case-insensitive configuration '${_deac_configuration}'.")
            endif()
            list(APPEND _deac_configuration_keys
                "${_deac_configuration_key}")
            _deac_hipblas_imported_location(
                "${_deac_provider}" "${_deac_configuration}" _deac_location)
            list(APPEND _deac_locations "${_deac_location}")
        endforeach()
        set(_deac_unique_locations ${_deac_locations})
        list(REMOVE_DUPLICATES _deac_unique_locations)
        list(LENGTH _deac_unique_locations _deac_unique_location_count)
        if(_deac_unique_location_count EQUAL 1)
            list(GET _deac_unique_locations 0 _deac_artifact)
        else()
            set(_deac_artifact "")
            set(_deac_index 0)
            foreach(_deac_configuration IN LISTS CMAKE_CONFIGURATION_TYPES)
                list(GET _deac_locations ${_deac_index} _deac_location)
                string(APPEND _deac_artifact
                    "$<$<CONFIG:${_deac_configuration}>:${_deac_location}>")
                math(EXPR _deac_index "${_deac_index} + 1")
            endforeach()
        endif()
    else()
        _deac_hipblas_imported_location(
            "${_deac_provider}" "${CMAKE_BUILD_TYPE}" _deac_artifact)
    endif()
    set(${output} "${_deac_artifact}" PARENT_SCOPE)
endfunction()

function(_deac_hipblas_validate_link_contract target_name)
    get_target_property(_deac_provider "${target_name}"
        DEAC_HIPBLAS_PROVIDER_TARGET)
    _deac_hipblas_canonical_provider("${_deac_provider}"
        _deac_canonical_provider)
    get_target_property(_deac_links "${target_name}"
        INTERFACE_LINK_LIBRARIES)
    if(NOT "${_deac_provider}" STREQUAL "${_deac_canonical_provider}" OR
            NOT "${_deac_links}" STREQUAL "${_deac_canonical_provider}")
        message(FATAL_ERROR
            "The hipBLAS link contract must contain exactly its canonical "
            "provider once; provider=${_deac_provider}, links=${_deac_links}.")
    endif()
    _deac_hipblas_provider_artifact(
        "${_deac_canonical_provider}" _deac_expected_artifact)
    get_target_property(_deac_recorded_artifact "${target_name}"
        DEAC_HIPBLAS_PROVIDER_ARTIFACT)
    if(NOT "${_deac_recorded_artifact}" STREQUAL
            "${_deac_expected_artifact}")
        message(FATAL_ERROR
            "The hipBLAS link contract's recorded provider artifact drifted "
            "from its canonical provider.")
    endif()
endfunction()

function(_deac_define_hipblas_link_contract)
    if(TARGET deac_hipblas_link_contract)
        _deac_hipblas_validate_link_contract(deac_hipblas_link_contract)
        return()
    endif()

    set(_deac_hipblas_rocm_roots)
    foreach(_deac_rocm_root IN ITEMS
            "$ENV{HIP_PATH}"
            "$ENV{ROCM_PATH}"
            "/opt/rocm")
        if(NOT "${_deac_rocm_root}" STREQUAL "")
            list(APPEND _deac_hipblas_rocm_roots "${_deac_rocm_root}")
        endif()
    endforeach()

    # Modern ROCm installations publish this supported imported target from
    # their hipblas config package.  Keep all package usage requirements behind
    # one project-owned target so every consumer receives the same contract.
    if(NOT TARGET roc::hipblas)
        find_package(hipblas CONFIG QUIET
            HINTS ${_deac_hipblas_rocm_roots})
    endif()

    if(TARGET roc::hipblas)
        set(_deac_hipblas_provider roc::hipblas)
    else()
        if(hipblas_FOUND)
            set(_deac_hipblas_package_problem
                "The hipblas package was found, but it did not define the supported imported target roc::hipblas.")
        else()
            set(_deac_hipblas_package_problem
                "CMake could not find the hipblas config package or its supported imported target roc::hipblas.")
        endif()

        set(_deac_hipblas_include_hints)
        set(_deac_hipblas_library_hints)
        foreach(_deac_rocm_root IN LISTS _deac_hipblas_rocm_roots)
            list(APPEND _deac_hipblas_include_hints
                "${_deac_rocm_root}/include")
            list(APPEND _deac_hipblas_library_hints
                "${_deac_rocm_root}/lib"
                "${_deac_rocm_root}/lib64")
        endforeach()

        find_path(DEAC_HIPBLAS_INCLUDE_DIR
            NAMES hipblas/hipblas.h
            HINTS ${_deac_hipblas_include_hints})
        find_library(DEAC_HIPBLAS_LIBRARY
            NAMES hipblas
            HINTS ${_deac_hipblas_library_hints})
        mark_as_advanced(
            DEAC_HIPBLAS_INCLUDE_DIR
            DEAC_HIPBLAS_LIBRARY)

        set(_deac_hipblas_header
            "${DEAC_HIPBLAS_INCLUDE_DIR}/hipblas/hipblas.h")
        set(_deac_hipblas_missing)
        if(NOT DEAC_HIPBLAS_INCLUDE_DIR
                OR NOT EXISTS "${_deac_hipblas_header}")
            list(APPEND _deac_hipblas_missing
                "hipblas/hipblas.h (DEAC_HIPBLAS_INCLUDE_DIR)")
        endif()
        if(NOT DEAC_HIPBLAS_LIBRARY
                OR NOT EXISTS "${DEAC_HIPBLAS_LIBRARY}"
                OR IS_DIRECTORY "${DEAC_HIPBLAS_LIBRARY}")
            list(APPEND _deac_hipblas_missing
                "an actual hipBLAS library (DEAC_HIPBLAS_LIBRARY)")
        endif()

        if(_deac_hipblas_missing)
            list(JOIN _deac_hipblas_missing ", " _deac_hipblas_missing_text)
            message(FATAL_ERROR
                "USE_BLAS=ON with GPU_BACKEND=hip requires hipBLAS. "
                "${_deac_hipblas_package_problem} "
                "The compatibility fallback requires both the header and library; missing: "
                "${_deac_hipblas_missing_text}. Install the hipBLAS development package and set "
                "CMAKE_PREFIX_PATH or hipblas_ROOT so find_package(hipblas CONFIG) exposes "
                "roc::hipblas. For a legacy installation, set DEAC_HIPBLAS_INCLUDE_DIR and "
                "DEAC_HIPBLAS_LIBRARY to existing header-root and library-file paths.")
        endif()

        add_library(deac_hipblas_compat UNKNOWN IMPORTED)
        set_target_properties(deac_hipblas_compat PROPERTIES
            IMPORTED_LOCATION "${DEAC_HIPBLAS_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${DEAC_HIPBLAS_INCLUDE_DIR}")
        set(_deac_hipblas_provider deac_hipblas_compat)
    endif()

    _deac_hipblas_canonical_provider(
        "${_deac_hipblas_provider}" _deac_hipblas_provider)
    _deac_hipblas_provider_artifact(
        "${_deac_hipblas_provider}" _deac_hipblas_provider_artifact)
    add_library(deac_hipblas_link_contract INTERFACE)
    target_link_libraries(deac_hipblas_link_contract INTERFACE
        "${_deac_hipblas_provider}")
    set_property(TARGET deac_hipblas_link_contract PROPERTY
        DEAC_HIPBLAS_PROVIDER_TARGET "${_deac_hipblas_provider}")
    set_property(TARGET deac_hipblas_link_contract PROPERTY
        DEAC_HIPBLAS_PROVIDER_ARTIFACT
        "${_deac_hipblas_provider_artifact}")
    _deac_hipblas_validate_link_contract(deac_hipblas_link_contract)
    add_library(deac::hipblas ALIAS deac_hipblas_link_contract)
endfunction()

function(deac_target_link_hipblas)
    if(NOT USE_BLAS OR NOT "${GPU_BACKEND}" STREQUAL "hip")
        return()
    endif()
    if(ARGC EQUAL 0)
        message(FATAL_ERROR
            "deac_target_link_hipblas requires at least one consumer target.")
    endif()

    _deac_define_hipblas_link_contract()
    foreach(_deac_target IN LISTS ARGN)
        if(NOT TARGET "${_deac_target}")
            message(FATAL_ERROR
                "Cannot apply the hipBLAS link contract: target '${_deac_target}' does not exist.")
        endif()
        get_property(_deac_hipblas_applied
            TARGET "${_deac_target}"
            PROPERTY DEAC_HIPBLAS_LINK_CONTRACT_APPLIED)
        if(NOT _deac_hipblas_applied)
            target_link_libraries("${_deac_target}" PRIVATE deac::hipblas)
            set_property(TARGET "${_deac_target}" PROPERTY
                DEAC_HIPBLAS_LINK_CONTRACT_APPLIED TRUE)
        endif()
    endforeach()
endfunction()
