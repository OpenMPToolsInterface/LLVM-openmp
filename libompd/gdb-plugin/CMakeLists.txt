set (CMAKE_MODULE_PATH
    "${CMAKE_SOURCE_DIR}/libompd/"
    ${CMAKE_MODULE_PATH}
)

find_package (PythonInterp) #2.7 REQUIRED)
find_package (PythonLibs) #2.7 REQUIRED)

include_directories (${OMPD_INCLUDE_PATH})
include_directories (${LIBOMP_INCLUDE_DIR})


add_custom_target(ompd_gdb_plugin ALL
                  COMMAND ${CMAKE_COMMAND} -E env LIBOMP_INCLUDE_DIR=${LIBOMP_INCLUDE_DIR}
                  ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/setup.py -v bdist_wheel -b ${CMAKE_CURRENT_BINARY_DIR}/build -d ${CMAKE_CURRENT_BINARY_DIR}
                  COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/setup.py clean --all
                  COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_SOURCE_DIR}/ompd.egg-info
                  COMMAND ${PYTHON_EXECUTABLE} -m pip install -U -t ${CMAKE_CURRENT_BINARY_DIR}/python-module --no-index --find-links=${CMAKE_CURRENT_BINARY_DIR} ompd
                  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                  COMMENT "Building the OMPD GDB plugin")

install(CODE "execute_process(COMMAND ${PYTHON_EXECUTABLE} -m pip install -U -t ${CMAKE_INSTALL_PREFIX}/share/gdb/python/gdb --no-index --find-links=${CMAKE_CURRENT_BINARY_DIR} ompd)")
