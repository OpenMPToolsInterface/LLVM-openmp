#include "ompd.h"


/*  4.3.4.1
 * Global initialization and finalization
 */


typedef ompd_rc_t (*ompd_initialize_fn_t) (
  ompd_word_t api_version,
  const ompd_callbacks_t *callbacks
);

typedef ompd_rc_t (*ompd_get_api_version_fn_t) (
  ompd_word_t *version
);

typedef ompd_rc_t (*ompd_get_version_string_fn_t) (
  const char **string
);

typedef ompd_rc_t (*ompd_finalize_fn_t) (void);

/* 4.3.4.2
 * Per OpenMP Process Initialiyation and Finalization
 */

typedef ompd_rc_t (*ompd_process_initialize_fn_t) (
    ompd_address_space_context_t *context,
    ompd_address_space_handle_t **handle
  );

typedef ompd_rc_t (*ompd_device_initialize_fn_t) (
    ompd_address_space_handle_t *process_handle, /*IN: address space of the OpenMP process*/
    ompd_address_space_context_t *device_context, /*IN: Opaque tool handle for device address space*/
    ompd_device_t kind, /*IN: device identifier kind*/
    ompd_size_t sizeof_id, /*IN: size of device identifier*/
    void *id, /*IN: device identifier*/
    ompd_address_space_handle_t **device_handle /*OUT: device handle*/
  );


typedef ompd_rc_t (*ompd_rel_address_space_handle_fn_t) (
    ompd_address_space_handle_t *addr_handle    /* IN: handle for the address space */
  );

/* 4.3.4.4
 * Address space information
 */

typedef ompd_rc_t (*ompd_get_omp_version_fn_t) (
    ompd_address_space_handle_t *address_space,
    ompd_word_t *omp_version
  );

typedef ompd_rc_t (*ompd_get_omp_version_string_fn_t) (
    ompd_address_space_handle_t *address_space,
    const char **string
  );

/* 4.3.4.5
 * Thread Handles
 */

typedef ompd_rc_t (*ompd_get_thread_in_parallel_fn_t) (
    ompd_parallel_handle_t *parallel_handle, /*IN: handle for the parallel region*/
    int thread_num, /*IN: the nubmer of the thread that is returned*/
    ompd_thread_handle_t **thread_hanlde /*OUT: returned thread handle*/
  );


typedef ompd_rc_t (*ompd_get_thread_handle_fn_t) (
    ompd_address_space_handle_t *addr_handle,    /* IN: handle for the address space */
    ompd_thread_id_t         kind,
    ompd_size_t                  sizeof_osthread,
    const void*                  osthread,
    ompd_thread_handle_t       **thread_handle     /* OUT: OpenMP thread handle*/
  );

typedef ompd_rc_t (*ompd_rel_thread_handle_fn_t) (
    ompd_thread_handle_t *thread_handle
);

typedef ompd_rc_t (*ompd_thread_handle_compare_fn_t) (
    ompd_thread_handle_t *thread_handle_1,
    ompd_thread_handle_t *thread_handle_2,
    int *cmp_value
);

typedef ompd_rc_t (*ompd_get_thread_id_fn_t) (
    ompd_thread_handle_t *thread_handle,
    ompd_thread_id_t kind,
    ompd_size_t sizeof_thread_id,
    void *thread_id
  );

/* 4.3.4.6
 * Parallel Region Handles
 */

typedef ompd_rc_t (*ompd_get_curr_parallel_handle_fn_t) (
    ompd_thread_handle_t *thread_handle,
    ompd_parallel_handle_t **parallel_handle
  );

typedef ompd_rc_t (*ompd_get_enclosing_parallel_handle_fn_t) (
    ompd_parallel_handle_t *parallel_handle, /* IN: OpenMP parallel handle */
    ompd_parallel_handle_t **enclosing_parallel_handle /* OUT: OpenMP parallel handle */
  );

typedef ompd_rc_t (*ompd_get_task_parallel_handle_fn_t) (
    ompd_task_handle_t *task_handle,
    ompd_parallel_handle_t **task_parallel_handle
  );

typedef ompd_rc_t (*ompd_rel_parallel_handle_fn_t) (
    ompd_parallel_handle_t *parallel_handle
  );

typedef ompd_rc_t (*ompd_parallel_handle_compare_fn_t) (
    ompd_parallel_handle_t *parallel_handle_1,
    ompd_parallel_handle_t *parallel_handle_2,
    int                    *cmp_value
  );

/* 4.3.4.7
 * Task Handles
 */

typedef ompd_rc_t (*ompd_get_curr_task_handle_fn_t) (
    ompd_thread_handle_t *thread_handle,
    ompd_task_handle_t **task_handle
  );

typedef ompd_rc_t (*ompd_get_generating_task_handle_fn_t) (
    ompd_task_handle_t *task_handle,
    ompd_task_handle_t **generating_task_handle
  );

typedef ompd_rc_t (*ompd_get_scheduling_task_handle_fn_t) (
    ompd_task_handle_t *task_handle,
    ompd_task_handle_t **scheduling_task_handle
  );

typedef ompd_rc_t (*ompd_get_task_in_parallel_fn_t) (
    ompd_parallel_handle_t *parallel_handle,
    int thread_num,
    ompd_task_handle_t **task_handle
  );

typedef ompd_rc_t (*ompd_rel_task_handle_fn_t) (
    ompd_task_handle_t *task_handle
);

typedef ompd_rc_t (*ompd_task_handle_compare_fn_t) (
    ompd_task_handle_t *task_handle_1,
    ompd_task_handle_t *task_handle_2,
    int                *cmp_value
);

typedef ompd_rc_t (*ompd_get_task_function_fn_t) (
    ompd_task_handle_t *task_handle,
    ompd_address_t *entry_point
  );

typedef ompd_rc_t (*ompd_get_task_frame_fn_t) (
    ompd_task_handle_t *task_handle,
    ompd_address_t *exit_frame,
    ompd_address_t *enter_frame
  );

typedef ompd_rc_t (*ompd_enumerate_states_fn_t) (
    ompd_address_space_handle_t *address_space_handle,
    ompd_word_t current_state,
    ompd_word_t *next_state,
    const char **next_state_name,
    ompd_word_t *more_enums
  );

typedef ompd_rc_t (*ompd_get_state_fn_t) (
    ompd_thread_handle_t *thread_handle,     /* IN: OpenMP thread handle*/
    ompd_word_t *state,                    /* OUT: State of this thread */
    ompd_wait_id_t *wait_id                 /* OUT: Wait ID */
  );

typedef ompd_rc_t (*ompd_enumerate_icvs_fn_t) (
    ompd_address_space_handle_t *handle,
    ompd_icv_id_t current,
    ompd_icv_id_t *next_id,
    const char **next_icv_name,
    ompd_scope_t *next_scope,
    int *more
  );

typedef ompd_rc_t (*ompd_get_icv_from_scope_fn_t) (
    void *handle,
    ompd_scope_t scope,
    ompd_icv_id_t icv_id,
    ompd_word_t *icv_value
  );
