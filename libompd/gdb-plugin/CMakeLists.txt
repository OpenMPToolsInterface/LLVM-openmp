set (CMAKE_MODULE_PATH
    "${CMAKE_SOURCE_DIR}/libompd/"
    ${CMAKE_MODULE_PATH}
)

find_package (PythonInterp 2.7 REQUIRED)
find_package (PythonLibs 2.7 REQUIRED)

include_directories (${OMPD_INCLUDE_PATH})
add_custom_target(
    ompd_c_module
    COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/setup.py build -b ${CMAKE_CURRENT_BINARY_DIR}/python-module
)
install(CODE "execute_process(COMMAND python setup.py install --user)")
list(APPEND PYTHONPATH "${CMAKE_CURRENT_BINARY_DIR}/python-module")