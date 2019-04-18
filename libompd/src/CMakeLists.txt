project (libompd)
add_library (ompd SHARED TargetValue.cpp omp-debug.cpp omp-state.cpp omp-icv.cpp)

add_dependencies(ompd omp) # ensure generated import library is created first

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include_directories (
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${LIBOMP_INCLUDE_DIR}
)

INSTALL( TARGETS ompd
        LIBRARY DESTINATION lib 
        ARCHIVE DESTINATION lib/static 
        RUNTIME DESTINATION bin )

INSTALL(FILES ompd.h DESTINATION include)

