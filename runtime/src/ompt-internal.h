#ifndef __OMPT_INTERNAL_H__
#define __OMPT_INTERNAL_H__

#include "ompt.h"
#include "ompt-event-specific.h"

#define OMPT_VERSION 1

#define _OMP_EXTERN extern "C"

#define OMPT_INVOKER(x) \
  ((x == fork_context_gnu) ? ompt_invoker_program : ompt_invoker_runtime)


#define ompt_callback(e) e ## _callback


typedef struct ompt_callbacks_internal_s {
#define ompt_event_macro(event, callback, eventid) callback ompt_callback(event);

    FOREACH_OMPT_EVENT(ompt_event_macro)

#undef ompt_event_macro
} ompt_callbacks_internal_t;


typedef struct kmp_taskdata  kmp_taskdata_t;
                
typedef struct {
    ompt_frame_t            frame;
    void*                   function;
    ompt_task_data_t        task_data;
    kmp_taskdata_t *        scheduling_parent;
#if OMP_40_ENABLED
    int                     ndeps;
    ompt_task_dependence_t  *deps;
#endif /* OMP_40_ENABLED */
} ompt_task_info_t;


typedef struct {
    ompt_parallel_data_t  parallel_data;
    void                *microtask;
} ompt_team_info_t;


typedef struct ompt_lw_taskteam_s {
    ompt_team_info_t    ompt_team_info;
    ompt_task_info_t    ompt_task_info;
    int			heap;
    struct ompt_lw_taskteam_s *parent;
} ompt_lw_taskteam_t;


//typedef struct ompt_parallel_info_s {
//    ompt_task_data_t* parent_task_data;    /* data of parent task            */
//    ompt_parallel_data_t* parallel_data;   /* data of parallel region        */
//    ompt_frame_t *parent_task_frame;  /* frame data of parent task    */
//    void *parallel_function;          /* pointer to outlined function */
//} ompt_parallel_info_t;


typedef struct {
    ompt_thread_data_t    thread_data;
    ompt_parallel_data_t  parallel_data; /* stored here from implicit barrier-begin until implicit-task-end */
    ompt_task_data_t      task_data; /* stored here from implicit barrier-begin until implicit-task-end */
    void                  *kmp_return_address; /* stored here on entry of runtime */
    void                  *gomp_return_address; /* stored here on entry of runtime */
    omp_state_t           state;
    ompt_wait_id_t        wait_id;
    int                   ompt_task_yielded;
    void                  *idle_frame;
} ompt_thread_info_t;


extern ompt_callbacks_internal_t ompt_callbacks;

#if OMP_40_ENABLED && OMPT_SUPPORT && OMPT_OPTIONAL
#if USE_FAST_MEMORY
#  define KMP_OMPT_DEPS_ALLOC __kmp_fast_allocate
#  define KMP_OMPT_DEPS_FREE __kmp_fast_free
# else
#  define KMP_OMPT_DEPS_ALLOC __kmp_thread_malloc
#  define KMP_OMPT_DEPS_FREE __kmp_thread_free
# endif
#endif /* OMP_40_ENABLED && OMPT_SUPPORT && OMPT_OPTIONAL */

#ifdef __cplusplus
extern "C" {
#endif

void ompt_pre_init(void);
void ompt_post_init(void);
void ompt_fini(void);


#ifdef OMPT_USE_LIBUNWIND
  void* __ompt_get_return_address_internal(int level);
  #define OMPT_GET_RETURN_ADDRESS(level) __ompt_get_return_address_internal(level)
  void* __ompt_get_frame_address_internal(int level);
  #define OMPT_GET_FRAME_ADDRESS(level) __ompt_get_frame_address_internal(level)
#else
  void* __ompt_get_return_address_backtrace(int level);
  #define OMPT_GET_RETURN_ADDRESS(level) __builtin_return_address(level)
//  #define OMPT_GET_RETURN_ADDRESS(level) __ompt_get_return_address_backtrace(level)
  #define OMPT_GET_FRAME_ADDRESS(level) __builtin_frame_address(level)
#endif

int __kmp_control_tool(uint64_t command, uint64_t modifier, void *arg);

extern int ompt_enabled;

#ifdef __cplusplus
};
#endif

#endif
