#include "omp-debug.h"
#include "ompd-private.h"
#include "TargetValue.h"

#define FOREACH_OMPD_ICV(macro)                                             \
    macro (levels_var, "levels-var", ompd_scope_parallel)                   \
    macro (active_levels_var, "active-levels-var", ompd_scope_parallel)     \
    macro (thread_limit_var, "thread-limit-var", ompd_scope_address_space)  \
    macro (max_active_levels_var, "max-active-levels-var", ompd_scope_task) \
    macro (bind_var, "bind-var", ompd_scope_task)                           \
    macro (num_procs_var, "ompd-num-procs-var", ompd_scope_address_space)   \
    macro (thread_num_var, "ompd-thread-num-var", ompd_scope_thread)        \
    macro (final_var, "ompd-final-var", ompd_scope_task)                    \
    macro (implicit_var, "ompd-implicit-var", ompd_scope_task)              \
    macro (team_size_var, "ompd-team-size-var", ompd_scope_parallel)        \

void __ompd_init_icvs(const ompd_callbacks_t *table) {
  callbacks = table;
}

enum ompd_icv {
  ompd_icv_undefined_marker = 0, // ompd_icv_undefined is already defined in ompd.h
#define ompd_icv_macro(v, n, s) ompd_icv_ ## v,
  FOREACH_OMPD_ICV(ompd_icv_macro)
#undef ompd_icv_macro
  ompd_icv_after_last_icv
};

static const char *ompd_icv_string_values[] = {
  "undefined",
#define ompd_icv_macro(v, n, s) n,
  FOREACH_OMPD_ICV(ompd_icv_macro)
#undef ompd_icv_macro
};

static const ompd_scope_t ompd_icv_scope_values[] = {
  ompd_scope_global,  // undefined marker
#define ompd_icv_macro(v, n, s) s,
  FOREACH_OMPD_ICV(ompd_icv_macro)
#undef ompd_icv_macro
};

ompd_rc_t ompd_enumerate_icvs(ompd_address_space_handle_t *handle,
                              ompd_icv_id_t current, ompd_icv_id_t *next_id,
                              const char **next_icv_name,
                              ompd_scope_t *next_scope,
                              int *more) {
  if (!handle) {
    return ompd_rc_stale_handle;
  }
  if (handle->kind == OMP_DEVICE_KIND_CUDA) {
    return ompd_rc_unsupported;
  }
  if (current + 1 >= ompd_icv_after_last_icv) {
    return ompd_rc_bad_input;
  }

  *next_id = current + 1;
  *next_icv_name = ompd_icv_string_values[*next_id];
  *next_scope = ompd_icv_scope_values[*next_id];

  if ((*next_id) + 1 >= ompd_icv_after_last_icv) {
    *more = 0;
  } else {
    *more = 1;
  }

  return ompd_rc_ok;
}


static ompd_rc_t ompd_get_level(
    ompd_parallel_handle_t *parallel_handle, /* IN: OpenMP parallel handle */
    ompd_word_t *val                         /* OUT: nesting level */
    ) {
  if (!parallel_handle->ah)
    return ompd_rc_stale_handle;
  ompd_address_space_context_t *context = parallel_handle->ah->context;
  if (!context)
    return ompd_rc_stale_handle;

  assert(callbacks && "Callback table not initialized!");

  uint32_t res;

  ompd_rc_t ret = TValue(context, parallel_handle->th)
                      .cast("kmp_base_team_t", 0) /*t*/
                      .access("t_level")          /*t.t_level*/
                      .castBase()
                      .getValue(res);
  *val = res;
  return ret;
}

static ompd_rc_t ompd_get_active_level(
    ompd_parallel_handle_t *parallel_handle, /* IN: OpenMP parallel handle */
    ompd_word_t *val                         /* OUT: active nesting level */
    ) {
  if (!parallel_handle->ah)
    return ompd_rc_stale_handle;
  ompd_address_space_context_t *context = parallel_handle->ah->context;
  if (!context)
    return ompd_rc_stale_handle;

  assert(callbacks && "Callback table not initialized!");

  uint32_t res;

  ompd_rc_t ret = TValue(context, parallel_handle->th)
                      .cast("kmp_base_team_t", 0) /*t*/
                      .access("t_active_level")   /*t.t_active_level*/
                      .castBase()
                      .getValue(res);
  *val = res;
  return ret;
}
 

static ompd_rc_t
ompd_get_num_procs(ompd_address_space_handle_t
                       *addr_handle, /* IN: handle for the address space */
                   ompd_word_t *val  /* OUT: number of processes */
                   ) {
  ompd_address_space_context_t *context = addr_handle->context;
  if (!context)
    return ompd_rc_stale_handle;
  ompd_rc_t ret;

  if (!context)
    return ompd_rc_stale_handle;

  assert(callbacks && "Callback table not initialized!");

  int nth;
  ret = TValue(context, "__kmp_avail_proc")
            .castBase("__kmp_avail_proc")
            .getValue(nth);
  *val = nth;
  return ret;
}

static ompd_rc_t
ompd_get_thread_limit(ompd_address_space_handle_t
                          *addr_handle, /* IN: handle for the address space */
                      ompd_word_t *val  /* OUT: max number of threads */
                      ) {
  if (!addr_handle)
    return ompd_rc_stale_handle;
  ompd_address_space_context_t *context = addr_handle->context;
  ompd_rc_t ret;

  if (!context)
    return ompd_rc_stale_handle;

  assert(callbacks && "Callback table not initialized!");

  int nth;
  ret =
      TValue(context, "__kmp_max_nth").castBase("__kmp_max_nth").getValue(nth);
  *val = nth;
  return ret;
} 

static ompd_rc_t ompd_get_thread_num(
    ompd_thread_handle_t *thread_handle, /* IN: OpenMP thread handle*/
    ompd_word_t *val /* OUT: number of the thread within the team */
    ) {
  // __kmp_threads[8]->th.th_info.ds.ds_tid
  if (!thread_handle)
    return ompd_rc_stale_handle;
  if (!thread_handle->ah)
    return ompd_rc_stale_handle;
  ompd_address_space_context_t *context = thread_handle->ah->context;
  if (!context)
    return ompd_rc_stale_handle;

  assert(callbacks && "Callback table not initialized!");

  ompd_rc_t ret =
      TValue(context, thread_handle->th) /*__kmp_threads[t]->th*/
          .cast("kmp_base_info_t")
          .access("th_info") /*__kmp_threads[t]->th.th_info*/
          .cast("kmp_desc_t")
          .access("ds") /*__kmp_threads[t]->th.th_info.ds*/
          .cast("kmp_desc_base_t")
          .access("ds_tid") /*__kmp_threads[t]->th.th_info.ds.ds_tid*/
          .castBase()
          .getValue(*val);
  return ret;
} 

static ompd_rc_t
ompd_in_final(ompd_task_handle_t *task_handle, /* IN: OpenMP task handle*/
              ompd_word_t *val                 /* OUT: max number of threads */
              ) {
  if (!task_handle->ah)
    return ompd_rc_stale_handle;
  ompd_address_space_context_t *context = task_handle->ah->context;
  if (!context)
    return ompd_rc_stale_handle;

  assert(callbacks && "Callback table not initialized!");

  ompd_rc_t ret = TValue(context, task_handle->th)
                      .cast("kmp_taskdata_t") // td
                      .access("td_flags")     // td->td_icvs
                      .cast("kmp_tasking_flags_t")
                      .check("final", val); // td->td_icvs.max_active_levels

  return ret;
}

static ompd_rc_t
ompd_get_max_active_levels(
    ompd_task_handle_t *task_handle, /* IN: OpenMP task handle*/
    ompd_word_t *val                 /* OUT: max number of threads */
    ) {
  if (!task_handle->ah)
    return ompd_rc_stale_handle;
  ompd_address_space_context_t *context = task_handle->ah->context;
  if (!context)
    return ompd_rc_stale_handle;

  assert(callbacks && "Callback table not initialized!");

  ompd_rc_t ret =
      TValue(context, task_handle->th)
          .cast("kmp_taskdata_t") // td
          .access("td_icvs")      // td->td_icvs
          .cast("kmp_internal_control_t", 0)
          .access("max_active_levels") // td->td_icvs.max_active_levels
          .castBase()
          .getValue(*val);

  return ret;
}

static ompd_rc_t
ompd_get_schedule(ompd_task_handle_t *task_handle, /* IN: OpenMP task handle*/
                  ompd_word_t *kind,    /* OUT: Kind of OpenMP schedule*/
                  ompd_word_t *modifier /* OUT: Schedunling modifier */
                  ) {
  if (!task_handle->ah)
    return ompd_rc_stale_handle;
  ompd_address_space_context_t *context = task_handle->ah->context;
  if (!context)
    return ompd_rc_stale_handle;

  assert(callbacks && "Callback table not initialized!");

  TValue sched = TValue(context, task_handle->th)
                     .cast("kmp_taskdata_t") // td
                     .access("td_icvs")      // td->td_icvs
                     .cast("kmp_internal_control_t", 0)
                     .access("sched") // td->td_icvs.sched
                     .cast("kmp_r_sched_t", 0);

  ompd_rc_t ret = sched
                      .access("r_sched_type") // td->td_icvs.sched.r_sched_type
                      .castBase()
                      .getValue(*kind);
  if (ret != ompd_rc_ok)
    return ret;
  ret = sched
            .access("chunk") // td->td_icvs.sched.r_sched_type
            .castBase()
            .getValue(*modifier);
  return ret;
}

static ompd_rc_t
ompd_get_proc_bind(ompd_task_handle_t *task_handle, /* IN: OpenMP task handle*/
                   ompd_word_t *bind /* OUT: Kind of proc-binding */
                   ) {
  if (!task_handle->ah)
    return ompd_rc_stale_handle;
  ompd_address_space_context_t *context = task_handle->ah->context;
  if (!context)
    return ompd_rc_stale_handle;

  assert(callbacks && "Callback table not initialized!");

  ompd_rc_t ret = TValue(context, task_handle->th)
                      .cast("kmp_taskdata_t") // td
                      .access("td_icvs")      // td->td_icvs
                      .cast("kmp_internal_control_t", 0)
                      .access("proc_bind") // td->td_icvs.proc_bind
                      .castBase()
                      .getValue(*bind);

  return ret;
}


static ompd_rc_t
ompd_is_implicit(ompd_task_handle_t *task_handle, /* IN: OpenMP task handle*/
                 ompd_word_t *val /* OUT: max number of threads */
                 ) {
  if (!task_handle)
    return ompd_rc_stale_handle;
  if (!task_handle->ah)
    return ompd_rc_stale_handle;
  ompd_address_space_context_t *context = task_handle->ah->context;
  if (!context)
    return ompd_rc_stale_handle;

  assert(callbacks && "Callback table not initialized!");

  ompd_rc_t ret = TValue(context, task_handle->th)
                      .cast("kmp_taskdata_t") // td
                      .access("td_flags")     // td->td_flags
                      .cast("kmp_tasking_flags_t")
                      .check("tasktype", val); // td->td_flags.tasktype
  *val ^= 1; // tasktype: explicit = 1, implicit = 0 => invert the value
  return ret;
} 

static ompd_rc_t
ompd_get_num_threads(ompd_parallel_handle_t
                        *parallel_handle, /* IN: OpenMP parallel handle */
                     ompd_word_t *val     /* OUT: number of threads */
                    ) {
  if (!parallel_handle->ah)
    return ompd_rc_stale_handle;
  ompd_address_space_context_t *context = parallel_handle->ah->context;
  if (!context)
    return ompd_rc_stale_handle;

  assert(callbacks && "Callback table not initialized!");

  ompd_rc_t ret = ompd_rc_ok;
  if (parallel_handle->lwt.address != 0) {
    *val = 1;
  } else {
    uint32_t res;
    ret = TValue(context, parallel_handle->th)
            .cast("kmp_base_team_t", 0) /*t*/
            .access("t_nproc")          /*t.t_nproc*/
            .castBase()
            .getValue(res);
   *val = res;
 }
 return ret;
} 

ompd_rc_t ompd_get_icv_from_scope(void *handle, ompd_scope_t scope,
                                  ompd_icv_id_t icv_id,
                                  ompd_word_t *icv_value) {
  if (!handle) {
    return ompd_rc_stale_handle;
  }
  if (icv_id >= ompd_icv_after_last_icv || icv_id == 0) {
    return ompd_rc_bad_input;
  }
  if (scope != ompd_icv_scope_values[icv_id]) {
    return ompd_rc_bad_input;
  }

  switch (icv_id) {
    case ompd_icv_levels_var:
      return ompd_get_level((ompd_parallel_handle_t *)handle, icv_value);
    case ompd_icv_active_levels_var:
      return ompd_get_active_level((ompd_parallel_handle_t *)handle, icv_value);
    case ompd_icv_thread_limit_var:
      return ompd_get_thread_limit((ompd_address_space_handle_t*)handle, icv_value);
    case ompd_icv_max_active_levels_var:
      return ompd_get_max_active_levels((ompd_task_handle_t*)handle, icv_value);
    case ompd_icv_bind_var:
      return ompd_get_proc_bind((ompd_task_handle_t*)handle, icv_value);
    case ompd_icv_num_procs_var:
      return ompd_get_num_procs((ompd_address_space_handle_t*)handle, icv_value);
    case ompd_icv_thread_num_var:
      return ompd_get_thread_num((ompd_thread_handle_t*)handle, icv_value);
    case ompd_icv_final_var:
      return ompd_in_final((ompd_task_handle_t*)handle, icv_value);
    case ompd_icv_implicit_var:
      return ompd_is_implicit((ompd_task_handle_t*)handle, icv_value);
    case ompd_icv_team_size_var:
      return ompd_get_num_threads((ompd_parallel_handle_t*)handle, icv_value);
    default:
      return ompd_rc_unsupported;
  }
}

ompd_rc_t
ompd_get_icv_string_from_scope(void *handle, ompd_scope_t scope,
                               ompd_icv_id_t icv_id,
                               const char **icv_string) {
  return ompd_rc_unsupported;
}
