#include "omp-tools.h"

const uint64_t  ompd_icv_undefined = 0;

typedef ompd_rc_t (*ompd_initialize_fn_t)(
  ompd_word_t api_version,
  const ompd_callbacks_t *callbacks);

typedef ompd_rc_t (*ompd_get_api_version_fn_t)(
  ompd_word_t *version);

typedef ompd_rc_t (*ompd_get_version_string_fn_t)(
  const char **string);

typedef ompd_rc_t (*ompd_finalize_fn_t)(void);

typedef ompd_rc_t (*ompd_process_initialize_fn_t)(
  ompd_address_space_context_t *context,
  ompd_address_space_handle_t **handle);

typedef ompd_rc_t (*ompd_device_initialize_fn_t)(
  ompd_address_space_handle_t *process_handle,
  ompd_address_space_context_t *device_context,
  ompd_device_t kind,
  ompd_size_t sizeof_id,
  void *id,
  ompd_address_space_handle_t **device_handle);

typedef ompd_rc_t (*ompd_rel_address_space_handle_fn_t)(
  ompd_address_space_handle_t *handle);

typedef ompd_rc_t (*ompd_get_omp_version_fn_t)(
  ompd_address_space_handle_t *address_space,
  ompd_word_t *omp_version);

typedef ompd_rc_t (*ompd_get_omp_version_string_fn_t)(
  ompd_address_space_handle_t *address_space,
  const char **string);

typedef ompd_rc_t (*ompd_get_thread_in_parallel_fn_t)(
  ompd_parallel_handle_t *parallel_handle,
  int thread_num,
  ompd_thread_handle_t **thread_handle);

typedef ompd_rc_t (*ompd_get_thread_handle_fn_t)(
  ompd_address_space_handle_t *handle,
  ompd_thread_id_t kind,
  ompd_size_t sizeof_thread_id,
  const void *thread_id,
  ompd_thread_handle_t **thread_handle);

typedef ompd_rc_t (*ompd_rel_thread_handle_fn_t)(
  ompd_thread_handle_t *thread_handle);

typedef ompd_rc_t (*ompd_thread_handle_compare_fn_t)(
  ompd_thread_handle_t *thread_handle_1,
  ompd_thread_handle_t *thread_handle_2,
  int *cmp_value);

typedef ompd_rc_t (*ompd_get_thread_id_fn_t)(
  ompd_thread_handle_t *thread_handle,
  ompd_thread_id_t kind,
  ompd_size_t sizeof_thread_id,
  void *thread_id);

typedef ompd_rc_t (*ompd_get_curr_parallel_handle_fn_t)(
  ompd_thread_handle_t *thread_handle,
  ompd_parallel_handle_t **parallel_handle);

typedef ompd_rc_t (*ompd_get_enclosing_parallel_handle_fn_t)(
  ompd_parallel_handle_t *parallel_handle,
  ompd_parallel_handle_t **enclosing_parallel_handle);

typedef ompd_rc_t (*ompd_get_task_parallel_handle_fn_t)(
  ompd_task_handle_t *task_handle,
  ompd_parallel_handle_t **task_parallel_handle);

typedef ompd_rc_t (*ompd_rel_parallel_handle_fn_t)(
  ompd_parallel_handle_t *parallel_handle);

typedef ompd_rc_t (*ompd_parallel_handle_compare_fn_t)(
  ompd_parallel_handle_t *parallel_handle_1,
  ompd_parallel_handle_t *parallel_handle_2,
  int *cmp_value);

typedef ompd_rc_t (*ompd_get_curr_task_handle_fn_t)(
  ompd_thread_handle_t *thread_handle,
  ompd_task_handle_t **task_handle);

typedef ompd_rc_t (*ompd_get_generating_task_handle_fn_t)(
  ompd_task_handle_t *task_handle,
  ompd_task_handle_t **generating_task_handle);

typedef ompd_rc_t (*ompd_get_scheduling_task_handle_fn_t)(
  ompd_task_handle_t *task_handle,
  ompd_task_handle_t **scheduling_task_handle);

typedef ompd_rc_t (*ompd_get_task_in_parallel_fn_t)(
  ompd_parallel_handle_t *parallel_handle,
  int thread_num,
  ompd_task_handle_t **task_handle);

typedef ompd_rc_t (*ompd_rel_task_handle_fn_t)(
  ompd_task_handle_t *task_handle);

typedef ompd_rc_t (*ompd_task_handle_compare_fn_t)(
  ompd_task_handle_t *task_handle_1,
  ompd_task_handle_t *task_handle_2,
  int *cmp_value);

typedef ompd_rc_t (*ompd_get_task_function_fn_t)(
  ompd_task_handle_t *task_handle,
  ompd_address_t *entry_point);

typedef ompd_rc_t (*ompd_get_task_frame_fn_t)(
  ompd_task_handle_t *task_handle,
  ompd_frame_info_t *exit_frame,
  ompd_frame_info_t *enter_frame);

typedef ompd_rc_t (*ompd_enumerate_states_fn_t)(
  ompd_address_space_handle_t *address_space_handle,
  ompd_word_t current_state,
  ompd_word_t *next_state,
  const char **next_state_name,
  ompd_word_t *more_enums);

typedef ompd_rc_t (*ompd_get_state_fn_t)(
  ompd_thread_handle_t *thread_handle,
  ompd_word_t *state,
  ompt_wait_id_t *wait_id);

typedef ompd_rc_t (*ompd_get_display_control_vars_fn_t)(
  ompd_address_space_handle_t *address_space_handle,
  const char *const **control_vars);

typedef ompd_rc_t (*ompd_rel_display_control_vars_fn_t)(
  const char *const **control_vars);

typedef ompd_rc_t (*ompd_enumerate_icvs_fn_t)(
  ompd_address_space_handle_t *handle,
  ompd_icv_id_t current,
  ompd_icv_id_t *next_id,
  const char **next_icv_name,
  ompd_scope_t *next_scope,
  int *more);

typedef ompd_rc_t (*ompd_get_icv_from_scope_fn_t)(
  void *handle,
  ompd_scope_t scope,
  ompd_icv_id_t icv_id,
  ompd_word_t *icv_value);

typedef ompd_rc_t (*ompd_get_icv_string_from_scope_fn_t)(
  void *handle,
  ompd_scope_t scope,
  ompd_icv_id_t icv_id,
  const char **icv_string);

typedef ompd_rc_t (*ompd_get_tool_data_fn_t)(
  void *handle,
  ompd_scope_t scope,
  ompd_word_t *value,
  ompd_address_t *ptr);
