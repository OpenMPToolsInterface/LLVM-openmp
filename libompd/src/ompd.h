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
 *    - Abstractions referring to the target have the prefix "t" (e.g.,
 *    "tmemory" for memory in the target, or "tsymbol" for symbol in the target)
 *    - Abstractions referring to the debugger have the prefix "d" (e.g.,
 *    "dmemory" for memory in the debugger)
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

#define OMPD_SEGMENT_UNSPECIFIED ((ompd_seg_t)0)
#define OMPD_SEGMENT_TEXT ((ompd_seg_t)1)
#define OMPD_SEGMENT_DATA ((ompd_seg_t)2)

/**
 * The following definitions match with ptx information stored in DWARF
 */
#define OMPD_SEGMENT_CUDA_PTX_UNSPECIFIED ((ompd_seg_t)0)
#define OMPD_SEGMENT_CUDA_PTX_CODE ((ompd_seg_t)1)
#define OMPD_SEGMENT_CUDA_PTX_REG ((ompd_seg_t)2)
#define OMPD_SEGMENT_CUDA_PTX_SREG ((ompd_seg_t)3)
#define OMPD_SEGMENT_CUDA_PTX_CONST ((ompd_seg_t)4)
#define OMPD_SEGMENT_CUDA_PTX_GLOBAL ((ompd_seg_t)5)
#define OMPD_SEGMENT_CUDA_PTX_LOCAL ((ompd_seg_t)6)
#define OMPD_SEGMENT_CUDA_PTX_PARAM ((ompd_seg_t)7)
#define OMPD_SEGMENT_CUDA_PTX_SHARED ((ompd_seg_t)8)
#define OMPD_SEGMENT_CUDA_PTX_SURF ((ompd_seg_t)9)
#define OMPD_SEGMENT_CUDA_PTX_TEX ((ompd_seg_t)10)
#define OMPD_SEGMENT_CUDA_PTX_TEXSAMPLER ((ompd_seg_t)11)
#define OMPD_SEGMENT_CUDA_PTX_GENERIC ((ompd_seg_t)12)
#define OMPD_SEGMENT_CUDA_PTX_IPARAM ((ompd_seg_t)13)
#define OMPD_SEGMENT_CUDA_PTX_OPARAM ((ompd_seg_t)14)
#define OMPD_SEGMENT_CUDA_PTX_FRAME ((ompd_seg_t)15)
#define OMPD_SEGMENT_CUDA_PTX_MAX ((ompd_seg_t)16)

//#if 0 // types removed in Austin F2F
/*
 * Definition of OMPD states, taken from OMPT
 */
#define FOREACH_OMPD_STATE(macro)                                              \
                                                                               \
  /* first */                                                                  \
  macro(ompd_state_first, 0x71) /* initial enumeration state */                \
                                                                               \
      /* work states (0..15) */                                                \
      macro(ompd_state_work_serial, 0x00)    /* working outside parallel */    \
      macro(ompd_state_work_parallel, 0x01)  /* working within parallel */     \
      macro(ompd_state_work_reduction, 0x02) /* performing a reduction */      \
                                                                               \
      /* idle (16..31) */                                                      \
      macro(ompd_state_idle, 0x10) /* waiting for work */                      \
                                                                               \
      /* overhead states (32..63) */                                           \
      macro(ompd_state_overhead, 0x20) /* overhead excluding wait states */    \
                                                                               \
      /* barrier wait states (64..79) */                                       \
      macro(ompd_state_wait_barrier, 0x40)          /* waiting at a barrier */ \
      macro(ompd_state_wait_barrier_implicit, 0x41) /* implicit barrier */     \
      macro(ompd_state_wait_barrier_explicit, 0x42) /* explicit barrier */     \
                                                                               \
      /* task wait states (80..95) */                                          \
      macro(ompd_state_wait_taskwait, 0x50)  /* waiting at a taskwait */       \
      macro(ompd_state_wait_taskgroup, 0x51) /* waiting at a taskgroup */      \
                                                                               \
      /* mutex wait states (96..111) */                                        \
      macro(ompd_state_wait_lock, 0x60)      /* waiting for lock */            \
      macro(ompd_state_wait_nest_lock, 0x61) /* waiting for nest lock */       \
      macro(ompd_state_wait_critical, 0x62)  /* waiting for critical */        \
      macro(ompd_state_wait_atomic, 0x63)    /* waiting for atomic */          \
      macro(ompd_state_wait_ordered, 0x64)   /* waiting for ordered */         \
      macro(ompd_state_wait_single,                                            \
            0x6F) /* waiting for single region (non-standard!) */              \
                                                                               \
      /* misc (112..127) */                                                    \
      macro(ompd_state_undefined, 0x70) /* undefined thread state */

typedef enum ompd_state_t {
#define ompd_state_macro(state, code) state = code,
  FOREACH_OMPD_STATE(ompd_state_macro)
#undef ompd_state_macro
} ompd_state_t;

#if 0
typedef enum ompd_sched_t {
  ompd_sched_static = 1,
  ompd_sched_dynamic = 2,
  ompd_sched_guided = 3,
  ompd_sched_auto = 4,
  ompd_sched_vendor_lo = 5,
  ompd_sched_vendor_hi = 0x7fffffff
} ompd_sched_t;

typedef enum ompd_proc_bind_t {
  ompd_proc_bind_false = 0,
  ompd_proc_bind_true = 1,
  ompd_proc_bind_master = 2,
  ompd_proc_bind_close = 3,
  ompd_proc_bind_spread = 4
} ompd_proc_bind_t;
#endif

typedef uint64_t ompd_device_identifier_t;

typedef enum ompd_device_kind_t {
  ompd_device_kind_host = 1,
  ompd_device_kind_cuda = 2
} ompd_device_kind_t;

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
 * Other handles.
 */
#define OMPD_THREAD_ID_PTHREAD 0
#define OMPD_THREAD_ID_LWP 1
#define OMPD_THREAD_ID_WINTHREAD 2
#define OMPD_THREAD_ID_CUDALOGICAL 3
#define OMPD_THREAD_ID_MAX 4

typedef enum ompd_thread_id_kind_t {
  ompd_thread_id_pthread = 0,
  ompd_thread_id_lwp = 1,
  ompd_thread_id_winthread = 2,
  ompd_thread_id_cudalogical = 3
} ompd_thread_id_kind_t;

/**
 * Logical coordinates of OMP target device threads
 */
typedef struct ompd_dim3_t {
  ompd_word_t x;
  ompd_word_t y;
  ompd_word_t z;
} ompd_dim3_t;

typedef struct ompd_cudathread_coord_t {
  ompd_addr_t cudaDevId;
  ompd_addr_t cudaContext;
  ompd_addr_t warpSize;
  ompd_addr_t gridId;
  ompd_addr_t kernelId; // TODO (MJM) - for some reason, cuda-gdb doesn't work
                        // with grids too well.
  ompd_dim3_t gridDim;
  ompd_dim3_t blockDim;
  ompd_dim3_t blockIdx;
  ompd_dim3_t threadIdx;
} ompd_cudathread_coord_t;

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
 * Primitive types.
 */
typedef enum ompd_target_prim_types_t {
  ompd_type_invalid = -1,
  ompd_type_char = 0,
  ompd_type_short = 1,
  ompd_type_int = 2,
  ompd_type_long = 3,
  ompd_type_long_long = 4,
  ompd_type_pointer = 5,
  ompd_type_max
} ompd_target_prim_types_t;

/**
 * Primitive type sizes.
 * These types are used by OMPD to interrogate the debugger about the size of
 * primitive types in the target.
 */
typedef struct ompd_target_type_sizes_t {
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
typedef ompd_rc_t (*ompd_dmemory_alloc_fn_t)(
    ompd_size_t bytes, /* IN: bytes of the primitive type */
    void **ptr         /* OUT: pointer of the allocated memory */
    );

/**
 * Free memory in the debugger's address space.
 */
typedef ompd_rc_t (*ompd_dmemory_free_fn_t)(
    void *ptr /* IN: pointer of memory to deallocate */
    );

/**
 * Get thread specific context.
 */
typedef ompd_rc_t (*ompd_get_thread_context_for_thread_id_fn_t)(
    ompd_address_space_context_t *context, ompd_thread_id_kind_t kind,
    ompd_size_t sizeof_thread_id, const void *thread_id,
    ompd_thread_context_t **thread_context);

#if 0
/**
 * Get containing (host) process context for address_space_context
 */
typedef ompd_rc_t (*ompd_get_process_context_for_context_fn_t) (
    ompd_address_space_context_t*
      address_space_context,           /* IN: OMP device/process addr space */
    ompd_address_space_context_t**
      containing_address_space_context /* OUT: Containing omp process addr space */
);
#endif

/**
 * Look up the sizes of primitive types in the target
 */
typedef ompd_rc_t (*ompd_tsizeof_prim_fn_t)(
    ompd_address_space_context_t
        *context,                   /* IN: debugger handle for the target */
    ompd_target_type_sizes_t *sizes /* OUT: type sizes */
    );

/**
 * Look up the address of a global symbol in the target
 */
typedef ompd_rc_t (*ompd_tsymbol_addr_fn_t)(
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
typedef ompd_rc_t (*ompd_tmemory_read_fn_t)(
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
typedef ompd_rc_t (*ompd_tmemory_write_fn_t)(
    ompd_address_space_context_t
        *context, /* IN: debugger handle for the target */
    ompd_thread_context_t
        *tcontext, /* IN: debugger handle for a target thread might be NULL */
    ompd_address_t addr, /* IN: address in the target */
    ompd_word_t nbytes,  /* IN: number of items to write */
    const void *buffer   /* IN: output buffer */
    );

typedef ompd_rc_t (*ompd_target_host_fn_t)(
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
typedef ompd_rc_t (*ompd_print_string_fn_t)(
    const char *str /* IN: message to print */
    );

/**
 * Callbacks table.
 */
typedef struct ompd_callbacks_t {
  /* Debugger interface */
  ompd_dmemory_alloc_fn_t dmemory_alloc;
  ompd_dmemory_free_fn_t dmemory_free;
  ompd_print_string_fn_t print_string;

  /* Target interface */
  ompd_tsizeof_prim_fn_t tsizeof_prim;
  ompd_tsymbol_addr_fn_t tsymbol_addr;
  ompd_tmemory_read_fn_t read_tmemory;
  ompd_tmemory_write_fn_t write_tmemory;

  ompd_target_host_fn_t target_to_host;
  ompd_target_host_fn_t host_to_target;

  ompd_get_thread_context_for_thread_id_fn_t get_thread_context_for_thread_id;
  //  ompd_get_process_context_for_context_fn_t  get_containing_process_context;

} ompd_callbacks_t;

/******************************************************************************
 * Call signatures from the debugger to the OMPD DLL.
 */

/* --- 4 Initialization ----------------------------------------------------- */

/**
 * The OMPD function ompd_get_version_string returns a descriptive string
 * describing an implementation of the OMPD library. The function
 * ompd_get_version_compatibility returns an integer code used to indicate the
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

ompd_rc_t
ompd_process_initialize(ompd_address_space_context_t
                            *context, /* IN: debugger handle for the target */
                        ompd_address_space_handle_t *
                            *addrhandle /* OUT: ompd handle for the target */
                        );

ompd_rc_t
ompd_get_openmp_version(ompd_address_space_handle_t
                            *addr_handle, /* IN: handle for the address space */
                        ompd_word_t *version);

ompd_rc_t ompd_get_openmp_version_string(
    ompd_address_space_handle_t
        *addr_handle, /* IN: handle for the address space */
    const char **string);

ompd_rc_t ompd_release_address_space_handle(
    ompd_address_space_handle_t
        *addr_handle /* IN: handle for the address space */
    );

ompd_rc_t ompd_device_initialize(
    ompd_address_space_handle_t *addr_handle,    /* IN: handle for the address space */
    ompd_address_space_context_t *device_context,
    int kind,
    ompd_size_t sizeof_id,
    void *id,
    ompd_address_space_handle_t **device_handle
    );

ompd_rc_t ompd_finalize(void);
/* --- 4 Handle Management -------------------------------------------------- */

/* --- 4.1 Thread Handles --------------------------------------------------- */

/**
 * Retrieve handles for all OpenMP threads.
 *
 * The ompd_get_threads operation enables the debugger to obtain handles for all
 * OpenMP threads. A successful invocation of ompd_get_threads returns a pointer
 * to a vector of handles in thread_handle_array and returns the number of
 * handles in num_handles. This call yields meaningful results only if all
 * OpenMP threads are stopped; otherwise, the OpenMP runtime may be creating
 * and/or destroying threads during or after the call, rendering useless the
 * vector of handles returned.
 */
 

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

#if 0
ompd_rc_t ompd_get_master_thread_in_parallel (
    ompd_parallel_handle_t *parallel_handle,    /* IN */
    ompd_thread_handle_t **thread_handle);
#endif

ompd_rc_t ompd_release_thread_handle(ompd_thread_handle_t *thread_handle);

ompd_rc_t ompd_thread_handle_compare(ompd_thread_handle_t *thread_handle_1,
                                     ompd_thread_handle_t *thread_handle_2,
                                     int *cmp_value);

#if 0
ompd_rc_t ompd_get_thread_handle_string_id (
    ompd_thread_handle_t *thread_handle,
    char **string_id
);
#endif

/* --- 4.2 Parallel Region Handles------------------------------------------- */

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

#if 0
ompd_rc_t ompd_get_parallel_handle_string_id (
    ompd_parallel_handle_t *parallel_handle,
    char **string_id
);
#endif

/* --- 4.3 Task Handles ----------------------------------------------------- */

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
#if 0
ompd_rc_t ompd_get_ancestor_task_handle(
    ompd_task_handle_t *task_handle,         /* IN: OpenMP task handle */
    ompd_task_handle_t **parent_task_handle  /* OUT: OpenMP task handle */
    );
#endif

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

#if 0
ompd_rc_t ompd_get_task_handle_string_id (
    ompd_task_handle_t *task_handle,
    char **string_id
);
#endif

/* --- 5o Process and Thread Settings ----------------------------------------
 */

/**
 * The functions ompd_get_num_procs and ompd_get_thread_limit are third-party
 * versions of the OpenMP runtime functions omp_get_num_procs and
 * omp_get_thread_limit.
 */

ompd_rc_t
ompd_get_num_procs(ompd_address_space_handle_t
                       *addr_handle, /* IN: handle for the address space */
                   ompd_word_t *val  /* OUT: number of processes */
                   );

ompd_rc_t
ompd_get_thread_limit(ompd_address_space_handle_t
                          *addr_handle, /* IN: handle for the address space */
                      ompd_word_t *val  /* OUT: max number of threads */
                      );

/* --- 6 Parallel Region Inqueries ------------------------------------------ */
/* --- 6.1 Settings --------------------------------------------------------- */

/**
 * Determine the number of threads associated with a parallel region.
 */
ompd_rc_t ompd_get_num_threads(
    ompd_parallel_handle_t *parallel_handle, /* IN: OpenMP parallel handle */
    ompd_word_t *val                         /* OUT: number of threads */
    );

/**
 * Determine the nesting depth of a particular parallel region instance.
 */
ompd_rc_t ompd_get_level(
    ompd_parallel_handle_t *parallel_handle, /* IN: OpenMP parallel handle */
    ompd_word_t *val                         /* OUT: nesting level */
    );

/**
 * Determine the number of enclosing active parallel regions.
 *
 * ompd_get_active_level returns the number of nested, active parallel regions
 * enclosing the parallel region specified by its handle.
 */
ompd_rc_t ompd_get_active_level(
    ompd_parallel_handle_t *parallel_handle, /* IN: OpenMP parallel handle */
    ompd_word_t *val                         /* OUT: active nesting level */
    );

/* --- 6.2 OMPT Parallel Region Inquiry Analogues ------------------------- */

/**
 * The functions ompd_get_parallel_id and ompd_get_parallel_function are
 * third-party variants of their OMPT counterparts. The only difference between
 * the OMPD and OMPT versions is that the OMPD must supply a parallel region
 * handle to provide a context for these inquiries.
 */
ompd_rc_t ompd_get_parallel_data(
    ompd_parallel_handle_t *parallel_handle, /* IN: OpenMP parallel handle */
    ompd_address_t *data                     /* OUT: OpenMP parallel id */
    );

#if 0
ompd_rc_t ompd_get_parallel_function(
    ompd_parallel_handle_t *parallel_handle, /* IN: OpenMP parallel handle */
    ompd_address_t *parallel_addr /* OUT: first instruction in the parallel region */
    );
#endif

/* --- 7 Thread Inquiry ----------------------------------------------------- */
/* --- 7.1 Operating System Thread Inquiry ---------------------------------- */

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
    ompd_thread_id_kind_t kind,
    ompd_size_t sizeof_thread_id, const void *thread_id,
    ompd_thread_handle_t **thread_handle /* OUT: OpenMP thread handle*/
    );

/**
 * Obtain the OS thread handle for an OpenMP thread handle.
 * this might change over time in case virtual openmp threads migrate between
 * OS threads.
 */
ompd_rc_t ompd_get_thread_id(
    ompd_thread_handle_t *thread_handle, /* IN: OpenMP thread handle*/
    ompd_thread_id_kind_t kind, ompd_size_t sizeof_thread_id, void *thread_id);

ompd_rc_t ompd_get_thread_data(
    ompd_thread_handle_t *thread_handle, /* IN: OpenMP thread handle*/
    ompd_address_t *data                 /* OUT: OpenMP thread data */
    );

ompd_rc_t ompd_get_thread_num(
    ompd_thread_handle_t *thread_handle, /* IN: OpenMP thread handle*/
    ompd_word_t *val /* OUT: number of the thread within the team */
    );

/* --- 7.2 OMPT Thread State Inquiry Analogue ------------------------------- */

/**
 * Get the state of a thread. This can use OMPT state data structure to define
 * different states of threads (e.g., idle, working, or barrier, etc) and what
 * entity cased this state (e.g., address of a lock);
 *
   * The function ompd_get_state is a third-party version of ompt_get_state. The
 * only difference between the OMPD and OMPT counterparts is that the OMPD
 * version must supply a thread handle to provide a context for this inquiry.
 */
ompd_rc_t ompd_get_state(
    ompd_thread_handle_t *thread_handle, /* IN: OpenMP thread handle*/
    ompd_word_t *state,                  /* OUT: State of this thread */
    ompd_wait_id_t *wait_id              /* OUT: Wait ID */
    );

/* --- 8 Task Inquiry ------------------------------------------------------- */

/* --- 8.1 Task Function Entry Point ---------------------------------------- */

/**
 * The ompd_get_task_function returns the entry point of the code that
 * corresponds to the body of code executed by the task.
 */

#if 0
ompd_rc_t ompd_get_task_function(
    ompd_task_handle_t *task_handle,         /* IN: OpenMP task handle*/
    ompd_address_t *entry_point /* OUT: first instruction in the task region */
    );
#endif

/* --- 8.2 Task Settings ---------------------------------------------------- */

/**
 * Retrieve information from OpenMP tasks. These inquiry functions have no
 * counterparts in the OMPT interface as a first-party tool can call OpenMP
 * runtime inquiry functions directly. The only difference between the OMPD
 * inquiry operations and their counterparts in the OpenMP runtime is that the
 * OMPD version must supply a task handle to provide a context for each inquiry.
 */

ompd_rc_t ompd_get_max_threads(
    ompd_task_handle_t *task_handle, /* IN: OpenMP task handle*/
    ompd_word_t *val                 /* OUT: max number of threads */
    );

ompd_rc_t
ompd_in_parallel(ompd_task_handle_t *task_handle, /* IN: OpenMP task handle*/
                 ompd_word_t *val /* OUT: Is OpenMP in parallel? */
                 );

ompd_rc_t
ompd_in_final(ompd_task_handle_t *task_handle, /* IN: OpenMP task handle*/
              ompd_word_t *val                 /* OUT: Is OpenMP in final? */
              );

ompd_rc_t
ompd_get_dynamic(ompd_task_handle_t *task_handle, /* IN: OpenMP task handle*/
                 ompd_word_t *val                 /* OUT: ? */
                 );

ompd_rc_t
ompd_get_nested(ompd_task_handle_t *task_handle, /* IN: OpenMP task handle */
                ompd_word_t *val                 /* OUT: Is this task nested? */
                );

ompd_rc_t ompd_get_max_active_levels(
    ompd_task_handle_t *task_handle, /* IN: OpenMP task handle */
    ompd_word_t *val                 /* OUT: max active levels */
    );

ompd_rc_t
ompd_get_schedule(ompd_task_handle_t *task_handle, /* IN: OpenMP task handle*/
                  ompd_word_t *kind,    /* OUT: Kind of OpenMP schedule*/
                  ompd_word_t *modifier /* OUT: Schedunling modifier */
                  );

ompd_rc_t
ompd_get_proc_bind(ompd_task_handle_t *task_handle, /* IN: OpenMP task handle*/
                   ompd_word_t *bind /* OUT: Kind of proc-binding */
                   );

ompd_rc_t
ompd_is_implicit(ompd_task_handle_t *task_handle, /* IN: OpenMP task handle*/
                 ompd_word_t *val /* OUT: implicit=1, explicit=0 */
                 );

/* --- 8.3 OMPT Task Inquiry Analogues -------------------------------------- */

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

ompd_rc_t
ompd_get_task_data(ompd_task_handle_t *task_handle, /* IN: OpenMP task handle */
                   ompd_address_t *task_data        /* OUT: OpenMP task ID */
                   );

/* --- 13 Display Control Variables ----------------------------------------- */

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

#ifdef __cplusplus
}
#endif
#endif /* SRC_OMPD_H_ */
