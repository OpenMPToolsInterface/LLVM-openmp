#include "ompd.h"
#include "ompt.h"
#include <vector>

ompd_rc_t ompd_test_get_parallel_regions(ompd_thread_handle_t* thread, std::vector<ompd_parallel_handle_t*> regions);

ompd_rc_t ompd_test_compare_parallel_id();

ompd_rc_t ompd_test_get_task_regions(ompd_thread_handle_t* thread, std::vector<ompd_task_handle_t*> regions);
