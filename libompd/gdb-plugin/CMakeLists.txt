set (CMAKE_MODULE_PATH
    "${CMAKE_SOURCE_DIR}/libompd/"
    ${CMAKE_MODULE_PATH}
)

find_package (PythonInterp) #2.7 REQUIRED)
find_package (PythonLibs) #2.7 REQUIRED)

include_directories (${OMPD_INCLUDE_PATH})
include_directories (${LIBOMP_INCLUDE_DIR})
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/python-module/timestamp
    COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/setup.py config -I ${LIBOMP_INCLUDE_DIR} 
    COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/setup.py build -b ${CMAKE_CURRENT_BINARY_DIR}/python-module
    COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_CURRENT_BINARY_DIR}/python-module/timestamp
    DEPENDS ompdModule.c
)
add_custom_target(ompd_c_module ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/python-module/timestamp)

install(CODE "execute_process(COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/setup.py install --user)")
list(APPEND PYTHONPATH "${CMAKE_CURRENT_BINARY_DIR}/python-module")