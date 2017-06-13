#ifndef OMPT_SPECIFIC_H
#define OMPT_SPECIFIC_H

#include "kmp.h"

/*****************************************************************************
 * types
 ****************************************************************************/

typedef kmp_info_t ompt_thread_t;



/*****************************************************************************
 * forward declarations
 ****************************************************************************/

void __ompt_team_assign_id(kmp_team_t *team, ompt_parallel_data_t ompt_pid);
void __ompt_thread_assign_wait_id(void *variable);

void __ompt_lw_taskteam_init(ompt_lw_taskteam_t *lwt, ompt_thread_t *thr,
                             int gtid, void *microtask,
                             ompt_parallel_data_t* ompt_pid);

void __ompt_lw_taskteam_link(ompt_lw_taskteam_t *lwt,  ompt_thread_t *thr, int on_heap);

//ompt_lw_taskteam_t * 
void __ompt_lw_taskteam_unlink(ompt_thread_t *thr);

ompt_id_t __ompt_parallel_id_new(int gtid);
ompt_id_t __ompt_task_id_new(int gtid);

ompt_team_info_t *__ompt_get_teaminfo(int depth, int *size);

ompt_task_info_t *__ompt_get_task_info_object(int depth);

int __ompt_get_parallel_info_internal(int ancestor_level, ompt_data_t **parallel_data, int *team_size);

int __ompt_get_task_info_internal(
    int ancestor_level,
    ompt_task_type_t *type,
    ompt_data_t **task_data,
    ompt_frame_t **task_frame,
    ompt_data_t **parallel_data,
    int *thread_num);

ompt_thread_data_t *__ompt_get_thread_data_internal();

static uint64_t __ompt_get_get_unique_id_internal();

/*****************************************************************************
 * macros
 ****************************************************************************/

#define OMPT_CUR_TASK_INFO(thr) (&(thr->th.th_current_task->ompt_task_info))
#define OMPT_CUR_TASK_DATA(thr) (&(thr->th.th_current_task->ompt_task_info.task_data))
#define OMPT_CUR_TEAM_INFO(thr) (&(thr->th.th_team->t.ompt_team_info))
#define OMPT_CUR_TEAM_DATA(thr) (&(thr->th.th_team->t.ompt_team_info.parallel_data))

#define OMPT_PAR_TASK_INFO(thr) (&(thr->th.th_current_task->td_parent->ompt_task_info))
#define OMPT_PAR_TASK_DATA(thr) (&(thr->th.th_current_task->td_parent->ompt_task_info.task_data))
#define OMPT_PAR_TEAM_INFO(thr) (&(thr->th.th_team->t.t_parent->t.ompt_team_info))
#define OMPT_PAR_TEAM_DATA(thr) (&(thr->th.th_team->t.t_parent->t.ompt_team_info.parallel_data))

#define OMPT_HAVE_WEAK_ATTRIBUTE KMP_HAVE_WEAK_ATTRIBUTE
#define OMPT_HAVE_PSAPI KMP_HAVE_PSAPI
#define OMPT_STR_MATCH(haystack, needle) __kmp_str_match(haystack, 0, needle)

inline void* __ompt_load_return_address(int gtid)
{
  kmp_info_t* thr = __kmp_threads[gtid];
  void* return_address = thr->th.ompt_thread_info.gomp_return_address ? thr->th.ompt_thread_info.gomp_return_address : thr->th.ompt_thread_info.kmp_return_address;
  thr->th.ompt_thread_info.gomp_return_address=NULL;
  thr->th.ompt_thread_info.kmp_return_address=NULL;
  return return_address;
}

#define OMPT_STORE_GOMP_RETURN_ADDRESS(gtid) __kmp_threads[gtid]->th.ompt_thread_info.gomp_return_address = __builtin_return_address(0)
#define OMPT_STORE_KMP_RETURN_ADDRESS(gtid) __kmp_threads[gtid]->th.ompt_thread_info.kmp_return_address = __builtin_return_address(0)
#define OMPT_LOAD_RETURN_ADDRESS(gtid) __ompt_load_return_address(gtid)


//******************************************************************************
// inline functions
//******************************************************************************

inline ompt_thread_t *
ompt_get_thread_gtid(int gtid)
{
    return (gtid >= 0) ? __kmp_thread_from_gtid(gtid) : NULL;
}


inline ompt_thread_t *
ompt_get_thread()
{
    int gtid = __kmp_get_gtid();
    return ompt_get_thread_gtid(gtid);
}


inline void
ompt_set_thread_state(ompt_thread_t *thread, omp_state_t state)
{
    thread->th.ompt_thread_info.state = state;
}


inline const char *
ompt_get_runtime_version()
{
    return &__kmp_version_lib_ver[KMP_VERSION_MAGIC_LEN];
}

#endif
