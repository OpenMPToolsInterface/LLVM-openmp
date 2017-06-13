//******************************************************************************
// include files
//******************************************************************************


#include "kmp.h"
#include "ompt-internal.h"
#include "ompt-specific.h"

#ifdef OMPT_USE_LIBUNWIND
#define UNW_LOCAL_ONLY
#include <libunwind.h>
#endif

#include <dlfcn.h>
#include <execinfo.h>

//******************************************************************************
// macros
//******************************************************************************

#define LWT_FROM_TEAM(team) (team)->t.ompt_serialized_team_info

#define OMPT_THREAD_ID_BITS 16

// 2013 08 24 - John Mellor-Crummey
//   ideally, a thread should assign its own ids based on thread private data.
//   however, the way the intel runtime reinitializes thread data structures
//   when it creates teams makes it difficult to maintain persistent thread
//   data. using a shared variable instead is simple. I leave it to intel to
//   sort out how to implement a higher performance version in their runtime.

// when using fetch_and_add to generate the IDs, there isn't any reason to waste
// bits for thread id.
#if 0
#define NEXT_ID(id_ptr,tid) \
  ((KMP_TEST_THEN_INC64(id_ptr) << OMPT_THREAD_ID_BITS) | (tid))
#else
#define NEXT_ID(id_ptr,tid) (KMP_TEST_THEN_INC64((volatile kmp_int64 *)id_ptr))
#endif

//******************************************************************************
// private operations
//******************************************************************************

//----------------------------------------------------------
// traverse the team and task hierarchy
// note: __ompt_get_teaminfo and __ompt_get_task_info_object
//       traverse the hierarchy similarly and need to be
//       kept consistent
//----------------------------------------------------------

ompt_team_info_t *
__ompt_get_teaminfo(int depth, int *size)
{
    kmp_info_t *thr = ompt_get_thread();

    if (thr) {
        kmp_team *team = thr->th.th_team;
        if (team == NULL) return NULL;

        ompt_lw_taskteam_t *next_lwt = LWT_FROM_TEAM(team), *lwt=NULL;

        while(depth > 0) {
            // next lightweight team (if any)
            if (lwt) lwt = lwt->parent;

            // next heavyweight team (if any) after
            // lightweight teams are exhausted
            if (!lwt && team) {
                if(next_lwt) {
                    lwt = next_lwt;
                    next_lwt=NULL;
                } else {
                    team=team->t.t_parent;
                    if(team) {
                        next_lwt = LWT_FROM_TEAM(team);
                    }
                }
            }

            depth--;
        }

        if (lwt) {
            // lightweight teams have one task
            if (size) *size = 1;

            // return team info for lightweight team
            return &lwt->ompt_team_info;
        } else if (team) {
            // extract size from heavyweight team
            if (size) *size = team->t.t_nproc;

            // return team info for heavyweight team
            return &team->t.ompt_team_info;
        }
    }

    return NULL;
}


ompt_task_info_t *
__ompt_get_task_info_object(int depth)
{
    ompt_task_info_t *info = NULL;
    kmp_info_t *thr = ompt_get_thread();

    if (thr) {
        kmp_taskdata_t  *taskdata = thr->th.th_current_task;
        ompt_lw_taskteam_t *lwt=NULL, *next_lwt  = LWT_FROM_TEAM(taskdata->td_team);

        while (depth > 0) {
            // next lightweight team (if any)
            if (lwt) lwt = lwt->parent;

            // next heavyweight team (if any) after
            // lightweight teams are exhausted
            if (!lwt && taskdata) {
                if(next_lwt) {
                    lwt = next_lwt;
                    next_lwt=NULL;
                } else {
                    taskdata = taskdata->td_parent;
                    if (taskdata) {
                        next_lwt = LWT_FROM_TEAM(taskdata->td_team);
                    }
                }
            }
            depth--;
        }

        if (lwt) {
            info = &lwt->ompt_task_info;
        } else if (taskdata) {
            info = &taskdata->ompt_task_info;
        }
    }

    return info;
}

ompt_task_info_t *
__ompt_get_scheduling_taskinfo(int depth)
{
    ompt_task_info_t *info = NULL;
    kmp_info_t *thr = ompt_get_thread();

    if (thr) {
        kmp_taskdata_t  *taskdata = thr->th.th_current_task;

        ompt_lw_taskteam_t *lwt=NULL, *next_lwt  = LWT_FROM_TEAM(taskdata->td_team);

        while (depth > 0) {
            // next lightweight team (if any)
            if (lwt) lwt = lwt->parent;

            // next heavyweight team (if any) after
            // lightweight teams are exhausted
            if (!lwt && taskdata) {
                // first try scheduling parent (for explicit task scheduling)
                if (taskdata->ompt_task_info.scheduling_parent) {
                    taskdata = taskdata->ompt_task_info.scheduling_parent;
                }else if(next_lwt) {
                    lwt = next_lwt;
                    next_lwt=NULL;
                } else {
                    // then go for implicit tasks
                    taskdata = taskdata->td_parent;
                    if (taskdata) {
                        next_lwt = LWT_FROM_TEAM(taskdata->td_team);
                    }
                }
            }
            depth--;
        }

        if (lwt) {
            info = &lwt->ompt_task_info;
        } else if (taskdata) {
            info = &taskdata->ompt_task_info;
        }
    }

    return info;
}

/*ompt_task_info_t *
__ompt_get_scheduling_taskinfo(int depth)
{
    ompt_task_info_t *info = NULL;
    kmp_info_t *thr = ompt_get_thread();

    if (thr) {
        kmp_taskdata_t  *taskdata = thr->th.th_current_task;
//        ompt_lw_taskteam_t *lwt = LWT_FROM_TEAM(taskdata->td_team);

        while (depth > 0) {

            if (taskdata) {
                taskdata = taskdata->ompt_task_info.scheduling_parent;
            }else{
                return NULL;
            }
            depth--;
        }
        if (taskdata) {
            info = &taskdata->ompt_task_info;
        }
    }

    return info;
}*/



//******************************************************************************
// interface operations
//******************************************************************************

//----------------------------------------------------------
// thread support
//----------------------------------------------------------

ompt_id_t
__ompt_thread_id_new()
{
    static uint64_t ompt_thread_id = 1;
    return NEXT_ID(&ompt_thread_id, 0);
}

ompt_thread_data_t *
__ompt_get_thread_data_internal()
{
    kmp_info_t *thread = ompt_get_thread();
    return &(thread->th.ompt_thread_info.thread_data);
}


//----------------------------------------------------------
// state support
//----------------------------------------------------------

void
__ompt_thread_assign_wait_id(void *variable)
{
    kmp_info_t *ti = ompt_get_thread();

    ti->th.ompt_thread_info.wait_id = (ompt_wait_id_t) variable;
}

omp_state_t
__ompt_get_state_internal(ompt_wait_id_t *ompt_wait_id)
{
    kmp_info_t *ti = ompt_get_thread();

    if (ti) {
        if (ompt_wait_id)
            *ompt_wait_id = ti->th.ompt_thread_info.wait_id;
        return ti->th.ompt_thread_info.state;
    }
    return omp_state_undefined;
}

//----------------------------------------------------------
// parallel region support
//----------------------------------------------------------

ompt_id_t
__ompt_parallel_id_new(int gtid)
{
#ifdef OMPT_INIT_IDS
    static uint64_t ompt_parallel_id = 1;
    return gtid >= 0 ? NEXT_ID(&ompt_parallel_id, gtid) : 0;
#else
    return 0;
#endif
}


int
__ompt_get_parallel_info_internal(int ancestor_level, ompt_data_t **parallel_data, int *team_size)
{
    ompt_team_info_t *info;
    if(team_size)
    {
        info = __ompt_get_teaminfo(ancestor_level, team_size);
    }
    else
    {
        info = __ompt_get_teaminfo(ancestor_level, NULL);
    }
    if(parallel_data)
    {
        *parallel_data = info ? &(info->parallel_data) : NULL;
    }
    return info ? 1 : 0;
}

//----------------------------------------------------------
// lightweight task team support
//----------------------------------------------------------

void
__ompt_lw_taskteam_init(ompt_lw_taskteam_t *lwt, kmp_info_t *thr,
                        int gtid, void *microtask,
                        ompt_parallel_data_t* ompt_pid)
{
    // initialize parallel_data with input, return address to parallel_data on exit
    lwt->ompt_team_info.parallel_data = *ompt_pid;
    
    lwt->ompt_team_info.microtask = microtask;
    lwt->ompt_task_info.task_data.value = 0;
    lwt->ompt_task_info.frame.reenter_runtime_frame = NULL;
    lwt->ompt_task_info.frame.exit_runtime_frame = NULL;
    lwt->ompt_task_info.scheduling_parent = NULL;
    lwt->ompt_task_info.function = NULL;
    lwt->ompt_task_info.deps = NULL;
    lwt->ompt_task_info.ndeps = 0;
    lwt->heap=0;
    lwt->parent = 0;
}


void
__ompt_lw_taskteam_link(ompt_lw_taskteam_t *lwt,  kmp_info_t *thr, int on_heap)
{
    ompt_lw_taskteam_t *link_lwt = lwt;
    if (thr->th.th_team->t.t_serialized > 1) { // we already have a team, so link the new team and swap values
        if(on_heap) { // the lw_taskteam cannot stay on stack, allocate it on heap
            link_lwt = (ompt_lw_taskteam_t *) __kmp_allocate(sizeof(ompt_lw_taskteam_t));
        }
        link_lwt->heap = on_heap;

        // would be swap in the (on_stack) case.
        ompt_team_info_t tmp_team = lwt->ompt_team_info;
        link_lwt->ompt_team_info = thr->th.th_team->t.ompt_team_info;
        thr->th.th_team->t.ompt_team_info = tmp_team;

        ompt_task_info_t tmp_task = lwt->ompt_task_info;
        link_lwt->ompt_task_info = thr->th.th_current_task->ompt_task_info;
        thr->th.th_current_task->ompt_task_info = tmp_task;

        // link the taskteam into the list of taskteams:
        ompt_lw_taskteam_t *my_parent = thr->th.th_team->t.ompt_serialized_team_info;
        link_lwt->parent = my_parent;
        thr->th.th_team->t.ompt_serialized_team_info = link_lwt;
    }else{
        // this is the first serialized team, so we just store the values in the team and drop the taskteam-object
        thr->th.th_team->t.ompt_team_info = lwt->ompt_team_info;
        thr->th.th_current_task->ompt_task_info = lwt->ompt_task_info;
    }
}


void
__ompt_lw_taskteam_unlink(kmp_info_t *thr)
{
    ompt_lw_taskteam_t *lwtask = thr->th.th_team->t.ompt_serialized_team_info;
    if (lwtask) {
        thr->th.th_team->t.ompt_serialized_team_info = lwtask->parent;

//        std::swap(lwtask->ompt_team_info,thr->th.th_team->t.ompt_team_info);
        ompt_team_info_t tmp_team = lwtask->ompt_team_info;
        lwtask->ompt_team_info = thr->th.th_team->t.ompt_team_info;
        thr->th.th_team->t.ompt_team_info = tmp_team;

//        std::swap(lwtask->ompt_task_info,thr->th.th_current_task->ompt_task_info);
        ompt_task_info_t tmp_task = lwtask->ompt_task_info;
        lwtask->ompt_task_info = thr->th.th_current_task->ompt_task_info;
        thr->th.th_current_task->ompt_task_info = tmp_task;

        if(lwtask->heap) {
            __kmp_free(lwtask);
            lwtask=NULL;
        }
    }
//    return lwtask;
}


//----------------------------------------------------------
// task support
//----------------------------------------------------------

ompt_id_t
__ompt_task_id_new(int gtid)
{
#ifdef OMPT_INIT_IDS
    static uint64_t ompt_task_id = 1;
    return NEXT_ID(&ompt_task_id, gtid);
#else
    return 0;
#endif
}


int
__ompt_get_task_info_internal(
    int ancestor_level,
    ompt_task_type_t *type,
    ompt_data_t **task_data,
    ompt_frame_t **task_frame,
    ompt_data_t **parallel_data,
    int *thread_num)
{
    //copied from __ompt_get_scheduling_taskinfo
    ompt_task_info_t *info = NULL;
    ompt_team_info_t *team_info = NULL;
    kmp_info_t *thr = ompt_get_thread();

    if (thr) {
        kmp_taskdata_t  *taskdata = thr->th.th_current_task;
        kmp_team *team = thr->th.th_team;
        ompt_lw_taskteam_t *lwt=NULL, *next_lwt  = LWT_FROM_TEAM(taskdata->td_team);

        while (ancestor_level > 0) {
            // next lightweight team (if any)
            if (lwt) lwt = lwt->parent;

            // next heavyweight team (if any) after
            // lightweight teams are exhausted
            if (!lwt && taskdata) {
                // first try scheduling parent (for explicit task scheduling)
                if (taskdata->ompt_task_info.scheduling_parent) {
                    taskdata = taskdata->ompt_task_info.scheduling_parent;
                }else if(next_lwt) {
                    lwt = next_lwt;
                    next_lwt=NULL;
                } else {
                    // then go for implicit tasks
                    taskdata = taskdata->td_parent;
                    team = team->t.t_parent;
                    if (taskdata) {
                        next_lwt = LWT_FROM_TEAM(taskdata->td_team);
                    }
                }
            }
            ancestor_level--;
        }

        if (lwt) {
            info = &lwt->ompt_task_info;
            team_info = &lwt->ompt_team_info;
            if(type)
            {
                *type = ompt_task_implicit;
            }
        } else if (taskdata) {
            info = &taskdata->ompt_task_info;
            team_info = &team->t.ompt_team_info;
            if(type)
            {
                if(taskdata->td_parent)
                {
                    *type = taskdata->td_flags.tasktype ? ompt_task_explicit : ompt_task_implicit;
                }
                else
                {
                    *type = ompt_task_initial;
                }
            }
        }
        if(task_data)
        {
            *task_data = info ? &info->task_data : NULL;
        }
        if(task_frame)
        {
            // OpenMP spec asks for the scheduling task to be returned.
            *task_frame = info ? &info->frame : NULL;
        }
        if(parallel_data)
        {
            *parallel_data = team_info ? &(team_info->parallel_data) : NULL;
        }
        return info ? 1 : 0;
    }
    return 0;
}

//----------------------------------------------------------
// team support
//----------------------------------------------------------

void
__ompt_team_assign_id(kmp_team_t *team, ompt_parallel_data_t ompt_pid)
{
    team->t.ompt_team_info.parallel_data = ompt_pid;
}

//----------------------------------------------------------
// misc
//----------------------------------------------------------

static uint64_t __ompt_get_unique_id_internal()
{
    static uint64_t thread=1;
    static __thread uint64_t ID=0;
    if (ID == 0)
    {
      uint64_t new_thread = __sync_fetch_and_add(&thread,1);
      ID = new_thread << (sizeof(uint64_t)*8 - OMPT_THREAD_ID_BITS);
    }
    return ++ID;
}

void* __ompt_get_return_address_backtrace(int level)
{

    int real_level = level + 2;
    void *array[real_level];
    size_t size;
  
    size = backtrace (array, real_level);
    if(size == real_level)
      return array[real_level-1];
    else
      return NULL;
}

#ifdef OMPT_USE_LIBUNWIND

void* __ompt_get_return_address_internal(int level)
{
    //get info about runtime lib
    Dl_info lib_info;
    dladdr((void*)&__ompt_get_return_address_internal, &lib_info);

    unw_cursor_t cursor;
    unw_context_t uc;
    unw_word_t ip;
    Dl_info info;

    //search for return address that does not point into the runtime lib
    unw_getcontext(&uc);
    unw_init_local(&cursor, &uc);
    do
    {
        unw_get_reg(&cursor, UNW_REG_IP, &ip);
        dladdr((void*) ip, &info);
        if(info.dli_fbase != lib_info.dli_fbase)
            return (void*)ip;
    } while (unw_step(&cursor) > 0);
    
    /*
    printf("%p, %d\n", (void*)ip, dladdr((void*) ip, &info));
    printf("%s, %s\n", info.dli_fname, __FILE__);
    printf("%s\n", dlerror());
    while (level > 0 && unw_step(&cursor) > 0)
    {
        level--;
        unw_get_reg(&cursor, UNW_REG_IP, &ip);
        printf("%p, %d\n", (void*)ip, dladdr((void*) ip, &info));
        printf("%s, %s\n", info.dli_fname, __FILE__);
        printf("%s\n", dlerror());
    }
    unw_get_reg(&cursor, UNW_REG_IP, &ip);

    if(level == 0)
      return (void*)ip;
    else
    */
        return NULL;
}


void* __ompt_get_frame_address_internal(int level)
{
    level++;
    unw_cursor_t cursor;
    unw_context_t uc;
    unw_word_t fp;

    //printf("%p\n", (void*)cursor.opaque[0]);

    unw_getcontext(&uc);
    unw_init_local(&cursor, &uc);
    //unw_get_reg(&cursor, UNW_REG_SP, &fp);
    //printf("ompt %p\n", (void*)fp);
    while (level > 0 && unw_step(&cursor) > 0)
    {
        //unw_get_reg(&cursor, UNW_REG_SP, &fp);
        //printf("ompt %p\n", (void*)fp);
        level--;
    }
    unw_get_reg(&cursor, UNW_REG_SP, &fp);

    if(level == 0)
      return (void*)(fp);
    else
      return NULL;
}
#endif
