/*
 * ompd.h
 *
 *  Created on: Dec 22, 2014
 *      Author: Ignacio Laguna
 *              Joachim Protze
 *     Contact: ilaguna@llnl.gov
 *              protze@llnl.gov
 */
#ifndef SRC_OMPD_H_
#define SRC_OMPD_H_

/******************************************************************************
 * This header file defines the OMPD interface: an interface to help debuggers
 * to inspect state associated with OpenMP programming abstractions in a target
 * process. The interface is implemented in a dynamically loaded library (DLL)
 * that the debugger loads into its address space.
 *
 * Name conventions:
 *    - All named entities start with the prefix "ompd_" (for OpenMP debugging)
 *    - Type entities end with the suffix "_t" (for type)
 *    - Function types end with the suffix "_fn_t" (for function type)
 *    - Return code entities have "_rc_" in it
 *
 * Comment conventions:
 *    - Input function parameters denoted by "IN:"
 *    - Output function parameters denoted by "OUT:"
 */

#include <stdint.h>
//#include "omp.h"

#ifdef __cplusplus
extern "C" {
#endif

/******************************************************************************
 * General types and data structures
 */

/**
 * Basic types.
 */
typedef uint64_t ompd_addr_t; /* unsigned integer large enough */
                              /* to hold a target address or a */
                              /* target segment value          */
typedef int64_t ompd_word_t;  /* signed version of ompd_addr_t */
typedef uint64_t ompd_seg_t;
typedef uint64_t ompd_wait_id_t; /* identifies what a thread is   */
                                 /* waiting for                   */
typedef uint64_t ompd_size_t;    /* For sizes (e.g., size_t)      */

typedef struct ompd_address_t {
  ompd_seg_t segment;  /* target architecture specific  */
                       /* segment value                 */
  ompd_addr_t address; /* target address in the segment */
} ompd_address_t;

/* types for device and thread id KIND, not for the actual thread/device id */
typedef uint64_t omp_device_t;
typedef uint64_t ompd_thread_id_t;

/**
 * Context handle.
 * This is used by the debugger to identify a target process (or core file).
 * This will be cast to concrete types within the debugger. The callbacks use
 * context handles to specify the debugger where to look up (since the debugger
 * can be handling different contexts at the same time, e.g., processes and/or
 * core files). Without context handles the debugger would not know the target
 * of a callback request.
 */

typedef struct _ompd_address_space_context_s ompd_address_space_context_t;
typedef struct _ompd_thread_context_s ompd_thread_context_t;

/**
 * OpenMP abstractions handles.
 * Each operation in the OMPD interface must explicitly specify a handle for the
 * context of the operation. OMPD uses context handles for OpenMP entities, such
 * as threads, parallel regions, and tasks. A handle for an entity is constant
 * while the entity itself is live.
 */

typedef struct _ompd_device_handle_s ompd_device_handle_t;
typedef struct _ompd_thread_handle_s ompd_thread_handle_t;
typedef struct _ompd_parallel_handle_s ompd_parallel_handle_t;
typedef struct _ompd_task_handle_s ompd_task_handle_t;
typedef struct _ompd_address_space_handle_s ompd_address_space_handle_t;

/**
 * Scope for ICVs
 */
typedef enum ompd_scope_t {
  ompd_scope_global = 1,
  ompd_scope_address_space = 2,
  ompd_scope_thread = 3,
  ompd_scope_parallel = 4,
  ompd_scope_implicit_task = 5,
  ompd_scope_task = 6
} ompd_scope_t;

typedef uint64_t ompd_icv_id_t;
const uint64_t ompd_icv_undefined = 0;

/**
 * Return codes.
 * Each OMPD operation returns a code.
 */
typedef enum ompd_rc_t {
  ompd_rc_ok = 0,           /* operation was successful */
  ompd_rc_unavailable = 1,  /* info is not available (in this context) */
  ompd_rc_stale_handle = 2, /* handle is no longer valid */
  ompd_rc_bad_input = 3,    /* bad input parameters (other than handle) */
  ompd_rc_error = 4,        /* error */
  ompd_rc_unsupported = 5,  /* operation is not supported */
  ompd_rc_needs_state_tracking = 6, /* needs runtime state tracking enabled */
  ompd_rc_incompatible = 7,       /* target is not compatible with this OMPD */
  ompd_rc_target_read_error = 8,  /* error reading from the target */
  ompd_rc_target_write_error = 9, /* error writing from the target */
  ompd_rc_nomem = 10              /* unable to allocate memory */
} ompd_rc_t;

/**
 * Primitive type sizes.
 * These types are used by OMPD to interrogate the debugger about the size of
 * primitive types in the target.
 */
typedef struct ompd_device_type_sizes_t {
  uint8_t sizeof_char;
  uint8_t sizeof_short;
  uint8_t sizeof_int;
  uint8_t sizeof_long;
  uint8_t sizeof_long_long;
  uint8_t sizeof_pointer;
} ompd_device_type_sizes_t;

/******************************************************************************
 * Debugger callback signatures.
 * These callback function signatures are used by OMPD to obtain state
 * information of a target process, in particular to interrogate about info
 * that is dependent on a particular OpenMP runtime library. Typical queries are
 * sizes of primitive types in the target, symbols lookup, lookup of offsets of
 * fields in a type/structure, and read/write to memory in the target.
 */

/**
 * Allocate memory in the debugger's address space.
 */
typedef ompd_rc_t (*ompd_callback_memory_alloc_fn_t)(
    ompd_size_t bytes, /* IN: bytes of the primitive type */
    void **ptr         /* OUT: pointer of the allocated memory */
    );

/**
 * Free memory in the debugger's address space.
 */
typedef ompd_rc_t (*ompd_callback_memory_free_fn_t)(
    void *ptr /* IN: pointer of memory to deallocate */
    );

/**
 * Get thread specific context.
 */
typedef ompd_rc_t (*ompd_callback_get_thread_context_for_thread_id_fn_t)(
    ompd_address_space_context_t *context, ompd_thread_id_t kind,
    ompd_size_t sizeof_thread_id, const void *thread_id,
    ompd_thread_context_t **thread_context);

/**
 * Look up the sizes of primitive types in the target
 */
typedef ompd_rc_t (*ompd_callback_sizeof_fn_t)(
    ompd_address_space_context_t
        *context,                   /* IN: debugger handle for the target */
    ompd_device_type_sizes_t *sizes /* OUT: type sizes */
    );

/**
 * Look up the address of a global symbol in the target
 */
typedef ompd_rc_t (*ompd_callback_symbol_addr_fn_t)(
    ompd_address_space_context_t
        *context, /* IN: debugger handle for the target */
    ompd_thread_context_t
        *tcontext, /* IN: debugger handle for a target thread might be NULL */
    const char *symbol_name,    /* IN: global symbol name */
    ompd_address_t *symbol_addr /* OUT: symbol address */
    );

/**
 * Read memory from the target
 */
typedef ompd_rc_t (*ompd_callback_memory_read_fn_t)(
    ompd_address_space_context_t
        *context, /* IN: debugger handle for the target */
    ompd_thread_context_t
        *tcontext, /* IN: debugger handle for a target thread might be NULL */
    ompd_address_t addr, /* IN: address in the target */
    ompd_word_t nbytes,  /* IN: number of items to read */
    void *buffer         /* OUT: output buffer */
    );

/**
 * Write memory from the target
 */
typedef ompd_rc_t (*ompd_callback_memory_write_fn_t)(
    ompd_address_space_context_t
        *context, /* IN: debugger handle for the target */
    ompd_thread_context_t
        *tcontext, /* IN: debugger handle for a target thread might be NULL */
    ompd_address_t addr, /* IN: address in the target */
    ompd_word_t nbytes,  /* IN: number of items to write */
    const void *buffer   /* IN: output buffer */
    );

typedef ompd_rc_t (*ompd_callback_device_host_fn_t)(
    ompd_address_space_context_t *address_space_context, /* IN */
    const void *input,                                   /* IN */
    int unit_size,                                       /* IN */
    int count,   /* IN: number of primitive type */
                 /* items to process */
    void *output /* OUT */
    );

/**
 * This is used by the OMPD library to have the debugger print a string.
 * The OMPD should not print directly.
 */
typedef ompd_rc_t (*ompd_callback_print_string_fn_t)(
    const char *str /* IN: message to print */
    );

/**
 * Callbacks table.
 */
typedef struct ompd_callbacks_t {
  /* Debugger interface */
  ompd_callback_memory_alloc_fn_t memory_alloc;
  ompd_callback_memory_free_fn_t memory_free;
  ompd_callback_print_string_fn_t print_string;

  /* Target interface */
  ompd_callback_sizeof_fn_t sizeof_types;
  ompd_callback_symbol_addr_fn_t symbol_addr_lookup;
  ompd_callback_memory_read_fn_t read_memory;
  ompd_callback_memory_write_fn_t write_memory;

  ompd_callback_device_host_fn_t device_to_host;
  ompd_callback_device_host_fn_t host_to_device;

  ompd_callback_get_thread_context_for_thread_id_fn_t get_thread_context_for_thread_id;
} ompd_callbacks_t;

/******************************************************************************
 * Call signatures from the debugger to the OMPD DLL.
 */

/* --- 4.1 Initialization --------------------------------------------------- */

/**
 * The OMPD function ompd_get_version_string returns a descriptive string
 * describing an implementation of the OMPD library. The function
 * ompd_get_api_version returns an integer code used to indicate the
 * revision of the OMPD specification supported by an implementation of OMPD.
 */

ompd_rc_t ompd_get_api_version(ompd_word_t *version);

ompd_rc_t
ompd_get_api_version_string(const char **string /* OUT: OMPD version string */
                            );

/**
 * Initialize OMPD.
 * This provides the DLL the pointers to the debugger's functions to obtain
 * information about the OpenMP runtime library. The debugger promises to
 * maintain the functions valid for as long as needed.
 */
ompd_rc_t
ompd_initialize(ompd_word_t version,
                const ompd_callbacks_t *table /* IN: callbacks table */
              );

ompd_rc_t ompd_finalize(void);

/* --- 4.2 Per Process Initialization and Finalization ---------------------- */

ompd_rc_t
ompd_process_initialize(ompd_address_space_context_t
                            *context, /* IN: debugger handle for the target */
                        ompd_address_space_handle_t *
                            *addrhandle /* OUT: ompd handle for the target */
                        );

ompd_rc_t ompd_device_initialize(
    ompd_address_space_handle_t *addr_handle,    /* IN: handle for the address space */
    ompd_address_space_context_t *device_context,
    int kind,
    ompd_size_t sizeof_id,
    void *id,
    ompd_address_space_handle_t **device_handle
    );

ompd_rc_t ompd_release_address_space_handle(
    ompd_address_space_handle_t
        *addr_handle /* IN: handle for the address space */
    );

/* --- 4.4 Address Space Information ---------------------------------------- */

ompd_rc_t
ompd_get_omp_version(ompd_address_space_handle_t
                            *addr_handle, /* IN: handle for the address space */
                        ompd_word_t *version);

ompd_rc_t ompd_get_omp_version_string(
    ompd_address_space_handle_t
        *addr_handle, /* IN: handle for the address space */
    const char **string);

/* --- 4.5 Thread Handles --------------------------------------------------- */

/**
 * Retrieve handles for OpenMP threads in a parallel region.
 *
 * The ompd_get_thread_in_parallel operation enables the debugger to obtain
 * handles for all OpenMP threads associated with a parallel region. A
 * successful invocation of ompd_get_thread_in_parallel returns a pointer to a
 * vector of handles in thread_handle_array and returns the number of handles in
 * num_handles. This call yields meaningful results only if all OpenMP threads
 * in the parallel region are stopped; otherwise, the OpenMP runtime may be
 * creating and/or destroying threads during or after the call, rendering
 * useless the vector of handles returned.
 */
ompd_rc_t ompd_get_thread_in_parallel(
    ompd_parallel_handle_t *parallel_handle, /* IN */
    int nth_thread,                      /* IN: number of the thread in team */
    ompd_thread_handle_t **thread_handle /* OUT: handle */
    );

/**
 * Obtain an OpenMP thread handle and the internal OS thread handle for the
 * selected (context) thread.
 * If the function returns ompd_rc_ok then the operating system thread
 * corresponds to an OpenMP thread and the thread_handle is initialized. The
 * value of thread_handle ans os_thread is meaningful only to the OpenMP runtime
 * system.
 */
ompd_rc_t ompd_get_thread_handle(
    ompd_address_space_handle_t
        *addr_handle, /* IN: handle for the address space */
    ompd_thread_id_t kind,
    ompd_size_t sizeof_thread_id, const void *thread_id,
    ompd_thread_handle_t **thread_handle /* OUT: OpenMP thread handle*/
    );

ompd_rc_t ompd_release_thread_handle(ompd_thread_handle_t *thread_handle);

ompd_rc_t ompd_thread_handle_compare(ompd_thread_handle_t *thread_handle_1,
                                     ompd_thread_handle_t *thread_handle_2,
                                     int *cmp_value);

/**
 * Obtain the OS thread handle for an OpenMP thread handle.
 * this might change over time in case virtual openmp threads migrate between
 * OS threads.
 */
ompd_rc_t ompd_get_thread_id(
    ompd_thread_handle_t *thread_handle, /* IN: OpenMP thread handle*/
    ompd_thread_id_t kind, ompd_size_t sizeof_thread_id, void *thread_id);

/* --- 4.6 Parallel Region Handles------------------------------------------- */

/**
 * Retrieve the handle for the innermost patallel region for an OpenMP thread.
 *
 * The operation ompd_get_current_parallel_handle enables the debugger to obtain
 * the handle for the innermost parallel region associated with an OpenMP
 * thread. This call is meaningful only if the thread whose handle is provided
 * is stopped.
 */

ompd_rc_t ompd_get_current_parallel_handle(
    ompd_thread_handle_t *thread_handle,     /* IN: OpenMP thread handle*/
    ompd_parallel_handle_t **parallel_handle /* OUT: OpenMP parallel handle */
    );

/**
 * Retrieve the handle for an enclosing parallel region.
 *
 * The ompd_get_enclosing_parallel_handle operation enables the debugger to
 * obtain the handle for the parallel region enclosing the parallel region
 * specified by parallel_handle. This call is meaningful only if at least one
 * thread in the parallel region is stopped.
 */

ompd_rc_t ompd_get_enclosing_parallel_handle(
    ompd_parallel_handle_t *parallel_handle, /* IN: OpenMP parallel handle */
    ompd_parallel_handle_t *
        *enclosing_parallel_handle /* OUT: OpenMP parallel handle */
    );

/**
 * Retrieve the handle for the enclosing parallel region or a task region.
 *
 * The ompd_get_task_parallel_handle operation enables the debugger to
 * obtain the handle for the parallel region enclosing the task region
 * specified by task_handle. This call is meaningful only if at least one
 * thread in the parallel region is stopped.
 */

ompd_rc_t ompd_get_task_parallel_handle(
    ompd_task_handle_t *task_handle, /* IN: OpenMP task handle */
    ompd_parallel_handle_t *
        *enclosing_parallel_handle /* OUT: OpenMP parallel handle */
    );

ompd_rc_t ompd_release_parallel_handle(ompd_parallel_handle_t *parallel_handle);

ompd_rc_t
ompd_parallel_handle_compare(ompd_parallel_handle_t *parallel_handle_1,
                             ompd_parallel_handle_t *parallel_handle_2,
                             int *cmp_value);

/* --- 4.7 Task Handles ----------------------------------------------------- */

/**
 * Retrieve the handle for the innermost task for an OpenMP thread.
 *
 * The debugger uses the operation ompd_get_current_task__handle to obtain the
 * handle
 * for the innermost task region associated with an OpenMP thread. This call is
 * meaningful only if the thread whose handle is provided is stopped.
 */
ompd_rc_t ompd_get_current_task_handle(
    ompd_thread_handle_t *thread_handle, /* IN: OpenMP thread handle*/
    ompd_task_handle_t **task_handle     /* OUT: OpenMP task handle */
    );

/**
 * Retrieve the handle for an enclosing task.
 *
 * The debugger uses ompd_get_ancestor_task_handle to obtain the handle for the
 * task region enclosing the task region specified by task_handle. This call is
 * meaningful only if the thread executing the task specified by task_handle is
 * stopped.
 */

ompd_rc_t ompd_get_generating_task_handle(
    ompd_task_handle_t *task_handle,        /* IN: OpenMP task handle */
    ompd_task_handle_t **parent_task_handle /* OUT: OpenMP task handle */
    );

ompd_rc_t ompd_get_scheduling_task_handle(
    ompd_task_handle_t *task_handle,        /* IN: OpenMP task handle */
    ompd_task_handle_t **parent_task_handle /* OUT: OpenMP task handle */
    );

/**
 * Retrieve implicit task handle for a parallel region.
 *
 * The ompd_get_implicit_task_in_parallel operation enables the debugger to
 * obtain handles for implicit tasks associated with a parallel region. This
 * call is meaningful only if all threads associated with the parallel region
 * are stopped.
 */
ompd_rc_t ompd_get_task_in_parallel(
    ompd_parallel_handle_t *parallel_handle, /* IN: OpenMP parallel handle */
    int nth_handle,                          /* IN: number of the task handle */
    ompd_task_handle_t **task_handle         /* OUT: OpenMP task handle */
    );

ompd_rc_t ompd_release_task_handle(ompd_task_handle_t *task_handle);

ompd_rc_t ompd_task_handle_compare(ompd_task_handle_t *task_handle_1,
                                   ompd_task_handle_t *task_handle_2,
                                   int *cmp_value);

/*
ompd_rc_t ompd_get_task_function(
    ompd_task_handle_t *task_handle,
    ompd_address_t *entry_point);
*/

/**
 * The functions defined here are third-party versions of ompt_get_task_frame
 * and ompt_get_task_data. The only difference between the OMPD and OMPT
 * counterparts is that the OMPD version must supply a task handle to provide a
 * context for these inquiries.
 */

/**
 * sp_exit
 *
 * This value is set once, the first time that a task exits the runtime to begin
 * executing user code. This field points to the stack frame of the runtime
 * procedure that called the user code. This value is NULL until just before the
 * task exits the runtime.
 *
 * sp_reentry
 *
 * This value is set each time that current task re-enters the runtime to create
 * new (implicit or explicit) tasks. This field points to the stack frame of the
 * runtime procedure called by a task to re-enter the runtime. This value is
 * NULL
 * until just after the task re-enters the runtime.
 */

ompd_rc_t ompd_get_task_frame(
    ompd_task_handle_t *task_handle, /* IN: OpenMP task handle*/
    ompd_address_t *sp_exit,         /* OUT: next frame is user code */
    ompd_address_t *sp_reentry       /* OUT: previous frame is user code */
    );


/**
 * Get the state of a thread. This can use OMPT state data structure to define
 * different states of threads (e.g., idle, working, or barrier, etc) and what
 * entity cased this state (e.g., address of a lock);
 *
   * The function ompd_get_state is a third-party version of ompt_get_state. The
 * only difference between the OMPD and OMPT counterparts is that the OMPD
 * version must supply a thread handle to provide a context for this inquiry.
 */
ompd_rc_t ompd_enumerate_states (
    ompd_address_space_handle_t *address_space_handle,
    ompd_word_t current_state,
    ompd_word_t *next_state,
    const char **next_state_name,
    ompd_word_t *more_enums
    );

ompd_rc_t ompd_get_state(
    ompd_thread_handle_t *thread_handle, /* IN: OpenMP thread handle*/
    ompd_word_t *state,                  /* OUT: State of this thread */
    ompd_wait_id_t *wait_id              /* OUT: Wait ID */
    );

/* --- 4.8 Display Control Variables ---------------------------------------- */

/**
 * Using the ompd_display_control_vars function, the debugger can extract a
 * string that contains a sequence of name/value pairs of control variables
 * whose settings are (a) user controllable, and (b) important to the operation
 * or performance of an OpenMP runtime system. The control variables exposed
 * through this interface will include all of the OMP environment variables,
 * settings that may come from vendor or platform- specific environment
 * variables (e.g., the IBM XL compiler has an environment variable that
 * controls spinning vs. blocking behavior), and other settings that affect
 * the operation or functioning of an OpenMP runtime system (e.g., numactl
 * settings that cause threads to be bound to cores).
 */

ompd_rc_t
ompd_get_display_control_vars(ompd_address_space_handle_t *handle,   /* IN */
                              const char *const **control_var_values /* OUT */
                              );

ompd_rc_t ompd_release_display_control_vars(
    const char *const **control_var_values /* IN */
    );

/* --- 4.9 Internal Control Variables --------------------------------------- */

ompd_rc_t
ompd_enumerate_icvs(ompd_address_space_handle_t *handle, ompd_icv_id_t current,
                    ompd_icv_id_t *next_id, const char **next_icv_name,
                    ompd_scope_t *next_scope, int *more);

ompd_rc_t
ompd_get_icv_from_scope(void *handle, ompd_scope_t scope, ompd_icv_id_t icv_id,
                        ompd_word_t *icv_value);

ompd_rc_t
ompd_get_icv_string_from_scope(void *handle, ompd_scope_t scope,
                               ompd_icv_id_t icv_id, const char **icv_string);

#ifdef __cplusplus
}
#endif
#endif /* SRC_OMPD_H_ */
