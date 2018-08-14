project (odb)

cmake_minimum_required(VERSION 2.8)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

set (cppfiles
  InputOutputManager.cpp
  ChildProcess.cpp
#  ProcessSpawn.cpp
  StringParser.cpp
  CudaGdb.cpp
  Debug.cpp
  GdbProcess.cpp
  Callbacks.cpp
  OMPDCommand.cpp
  OMPDContext.cpp
  OutputString.cpp)

# Let's find GDB first.
find_package(GDB REQUIRED)
if (GDB_FOUND)
  MESSAGE( STATUS "GDB_PATH:  " ${GDB_COMMAND} )
  add_definitions (-DGDB_PATH="${GDB_COMMAND}")
endif (GDB_FOUND)

find_package(CudaGDB QUIET)
if (CUDA_GDB_FOUND)
        MESSAGE( STATUS "CUDA_GDB_PATH:  " ${CUDA_GDB_COMMAND} )
        add_definitions (-DCUDA_GDB_PATH="${CUDA_GDB_COMMAND}")
endif (CUDA_GDB_FOUND)

add_executable (odb-bin ${cppfiles} odb.cpp)
set_target_properties (odb-bin PROPERTIES OUTPUT_NAME odb)
add_library (odb ${cppfiles})

target_link_libraries (odb-bin dl)
target_link_libraries (odb dl)

include_directories (
	${CMAKE_CURRENT_SOURCE_DIR}
	${CMAKE_CURRENT_SOURCE_DIR}/../src/
    ${CMAKE_BINARY_DIR}/include
)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  add_definitions (-DDEBUG)
endif (CMAKE_BUILD_TYPE STREQUAL "Debug")

# Find readline library
find_package(Readline REQUIRED)
include_directories(${Readline_INCLUDE_DIRS})
set(LIBS ${LIBS} ${Readline_LIBRARY})

target_link_libraries(odb-bin ${LIBS})

install(TARGETS odb odb-bin 
	LIBRARY DESTINATION lib 
	ARCHIVE DESTINATION lib/static 
	RUNTIME DESTINATION bin )
