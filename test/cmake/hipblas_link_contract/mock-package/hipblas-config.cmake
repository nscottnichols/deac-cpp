if(NOT TARGET roc::hipblas)
    get_filename_component(_deac_mock_hipblas_include
        "${CMAKE_CURRENT_LIST_DIR}/../mock-include" ABSOLUTE)
    get_filename_component(_deac_mock_hipblas_library
        "${CMAKE_CURRENT_LIST_DIR}/../mock-library/libhipblas.a" ABSOLUTE)
    add_library(roc::hipblas UNKNOWN IMPORTED)
    set_target_properties(roc::hipblas PROPERTIES
        IMPORTED_LOCATION "${_deac_mock_hipblas_library}"
        INTERFACE_INCLUDE_DIRECTORIES "${_deac_mock_hipblas_include}")
    unset(_deac_mock_hipblas_include)
    unset(_deac_mock_hipblas_library)
endif()
