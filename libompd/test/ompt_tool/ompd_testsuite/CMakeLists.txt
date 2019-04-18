SET(CMAKE_MODULE_PATH
    "${CMAKE_SOURCE_DIR}/cmakemodules"
    ${CMAKE_MODULE_PATH}
)

# TODO: no absolute paths
include_directories (
        ${CMAKE_BINARY_DIR}/include
	${CMAKE_SOURCE_DIR}/ompd_callbacks
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${OMPD_INCLUDE_PATH}
        ${LIBOMP_INCLUDE_PATH}
)

add_library (ompt_plugin SHARED ompt_plugin.cpp)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# TODO
#file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/include)

INSTALL(TARGETS ompt_plugin
        LIBRARY DESTINATION lib 
        ARCHIVE DESTINATION lib/static 
        RUNTIME DESTINATION bin)

INSTALL(FILES ompd_testsuite.h DESTINATION include)

