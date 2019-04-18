include_directories (
	${Callbacks_SOURCE_DIR}
	${ompd-testsuite_SOURCE_DIR}
	)

find_package (OpenMP)
find_package (Threads)

list(APPEND CMAKE_C_FLAGS "${OpenMP_C_FLAGS} -g")

# foreach(SRC example_1 example_2 example_3 example_4 example_5 example_task nested)
# 	add_executable(${SRC}_plugin ${SRC}.c)
# 	target_link_libraries (${SRC}_plugin ompt_plugin)
# 	ADD_CUSTOM_TARGET(link_${SRC}_target ALL
#                 COMMAND ${CMAKE_COMMAND} -E copy ${SRC}_plugin ${CMAKE_CURRENT_BINARY_DIR}/Output/${SRC}.c.tmp)
#        add_dependencies(link_${SRC}_target ${SRC}_plugin)
# endforeach(SRC)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../lit.site.cfg.in ${CMAKE_CURRENT_SOURCE_DIR}/../lit.site.cfg @ONLY)