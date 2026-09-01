include_guard(GLOBAL)

function(_deac_define_hipblas_link_contract)
    if(TARGET deac_hipblas_link_contract)
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

    add_library(deac_hipblas_link_contract INTERFACE)
    target_link_libraries(deac_hipblas_link_contract INTERFACE
        "${_deac_hipblas_provider}")
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
