project (libompd)

add_library (ompd SHARED TargetValue.cpp omp-debug.cpp)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include_directories (
        ${CMAKE_CURRENT_BINARY_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}
)

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include)

ADD_CUSTOM_TARGET (
  ompd-header ALL
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/include/ompd.h ${CMAKE_CURRENT_BINARY_DIR}/include/ompd_typedefs.h
)

add_custom_command ( 
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/include/ompd.h ${CMAKE_CURRENT_BINARY_DIR}/include/ompd_typedefs.h
        COMMAND ${BASH} 
        ARGS    ${CMAKE_CURRENT_SOURCE_DIR}/split_def_typedef.sh 
                ${CMAKE_CURRENT_SOURCE_DIR}/ompd-template.h
                ${CMAKE_CURRENT_BINARY_DIR}/include/ompd_typedefs.h
                ${CMAKE_CURRENT_BINARY_DIR}/include/ompd.h
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/ompd-template.h
)

INSTALL( TARGETS ompd
        LIBRARY DESTINATION lib 
        ARCHIVE DESTINATION lib/static 
        RUNTIME DESTINATION bin )

INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/ompd.h DESTINATION include)

