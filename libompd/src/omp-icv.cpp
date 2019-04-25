#include <cstring>
#include "omp-debug.h"
#include "ompd-private.h"
#include "TargetValue.h"

#define FOREACH_OMPD_ICV(macro)                                                \
    macro (levels_var, "levels-var", ompd_scope_parallel, 1)                   \
    macro (active_levels_var, "active-levels-var", ompd_scope_parallel, 0)     \
    macro (thread_limit_var, "thread-limit-var", ompd_scope_address_space, 0)  \
    macro (max_active_levels_var, "max-active-levels-var", ompd_scope_task, 0) \
    macro (bind_var, "bind-var", ompd_scope_task, 0)                           \
    macro (num_procs_var, "ompd-num-procs-var", ompd_scope_address_space, 0)   \
    macro (thread_num_var, "ompd-thread-num-var", ompd_scope_thread, 1)        \
    macro (final_var, "ompd-final-var", ompd_scope_task, 0)                    \
    macro (implicit_var, "ompd-implicit-var", ompd_scope_task, 0)              \
    macro (team_size_var, "ompd-team-size-var", ompd_scope_parallel, 1)        \

void __ompd_init_icvs(const ompd_callbacks_t *table) {
  callbacks = table;
}

enum ompd_icv {
  ompd_icv_undefined_marker = 0, // ompd_icv_undefined is already defined in ompd.h
#define ompd_icv_macro(v, n, s, d) ompd_icv_ ## v,
  FOREACH_OMPD_ICV(ompd_icv_macro)
#undef ompd_icv_macro
  ompd_icv_after_last_icv
};

static const char *ompd_icv_string_values[] = {
  "undefined",
#define ompd_icv_macro(v, n, s, d) n,
  FOREACH_OMPD_ICV(ompd_icv_macro)
#undef ompd_icv_macro
};

static const ompd_scope_t ompd_icv_scope_values[] = {
  ompd_scope_global,  // undefined marker
#define ompd_icv_macro(v, n, s, d) s,
  FOREACH_OMPD_ICV(ompd_icv_macro)
#undef ompd_icv_macro
};

static const uint8_t ompd_icv_available_cuda[] = {
  1, // undefined marker
#define ompd_icv_macro(v, n, s, d) d,
  FOREACH_OMPD_ICV(ompd_icv_macro)
#undef ompd_icv_macro
 1, // icv after last icv marker
};


static ompd_rc_t ompd_enumerate_icvs_cuda(ompd_icv_id_t current,
                                          ompd_icv_id_t *next_id,
                                          const char **next_icv_name,
                                          ompd_scope_t *next_scope,
                                          int *more) {
  int next_possible_icv = current;
  ompd_rc_t ret;
  do {
    next_possible_icv++;
  } while (!ompd_icv_available_cuda[next_possible_icv]);

  if (next_possible_icv >= ompd_icv_after_last_icv) {
    return ompd_rc_bad_input;
  }

  *next_id = next_possible_icv;

  ret = callbacks->alloc_memory(
      std::strlen(ompd_icv_string_values[*next_id]) + 1,
      (void**) next_icv_name);
  if (ret != ompd_rc_ok) {
    return ret;
  }
  std::strcpy((char*)*next_icv_name, ompd_icv_string_values[*next_id]);

  *next_scope = ompd_icv_scope_values[*next_id];

  do {
    next_possible_icv++;
  } while (!ompd_icv_available_cuda[next_possible_icv]);

  if (next_possible_icv >= ompd_icv_after_last_icv) {
    *more = 0;
  } else {
    *more = 1;
  }
  return ompd_rc_ok;
}

ompd_rc_t ompd_enumerate_icvs(ompd_address_space_handle_t *handle,
                              ompd_icv_id_t current, ompd_icv_id_t *next_id,
                              const char **next_icv_name,
                              ompd_scope_t *next_scope,
                              int *more) {
  if (!handle) {
    return ompd_rc_stale_handle;
  }
  if (handle->kind == OMPD_DEVICE_KIND_CUDA) {
    return ompd_enumerate_icvs_cuda(current, next_id, next_icv_name,
                                    next_scope, more);
  }
  if (current + 1 >= ompd_icv_after_last_icv) {
    return ompd_rc_bad_input;
  }

  *next_id = current + 1;

  ompd_rc_t ret = callbacks->alloc_memory(
      std::strlen(ompd_icv_string_values[*next_id]) + 1,
      (void**)next_icv_name);
  if (ret != ompd_rc_ok) {
    return ret;
  }
  std::strcpy((char*)*next_icv_name, ompd_icv_string_values[*next_id]);

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


static ompd_rc_t ompd_get_level_cuda(
    ompd_parallel_handle_t *parallel_handle,
    ompd_word_t *val) {
  if (!parallel_handle->ah)
    return ompd_rc_stale_handle;
  ompd_address_space_context_t *context = parallel_handle->ah->context;
  if (!context)
    return ompd_rc_stale_handle;

  assert(callbacks && "Callback table not initialized");

  uint16_t res;
  ompd_rc_t ret = TValue(context, parallel_handle->th)
                      .cast("ompd_nvptx_parallel_info_t", 0,
                            OMPD_SEGMENT_CUDA_PTX_GLOBAL)
                      .access("level")
                      .castBase(ompd_type_short)
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

static ompd_rc_t
ompd_get_num_threads_cuda(ompd_parallel_handle_t *parallel_handle,
                          ompd_word_t *val) {
  if (!parallel_handle->ah)
    return ompd_rc_stale_handle;
  ompd_address_space_context_t *context = parallel_handle->ah->context;
  if (!context)
    return ompd_rc_stale_handle;

  assert(callbacks && "Callback table not initialized");

  uint16_t res;

  ompd_rc_t ret = TValue(context, parallel_handle->th)
                      .cast("ompd_nvptx_parallel_info_t", 0,
                            OMPD_SEGMENT_CUDA_PTX_GLOBAL)
                      .access("parallel_tasks")
                      .cast("omptarget_nvptx_TaskDescr", 1,
                            OMPD_SEGMENT_CUDA_PTX_GLOBAL)
                      .access("items__threadsInTeam")
                      .castBase()
                      .getValue(res);
  *val = res;
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

  ompd_device_t device_kind;

  switch (scope) {
    case ompd_scope_thread:
      device_kind = ((ompd_thread_handle_t *)handle)->ah->kind;
      break;
    case ompd_scope_parallel:
      device_kind = ((ompd_parallel_handle_t *)handle)->ah->kind;
      break;
    case ompd_scope_address_space:
      device_kind = ((ompd_address_space_handle_t *)handle)->kind;
      break;
    case ompd_scope_task:
      device_kind = ((ompd_task_handle_t *)handle)->ah->kind;
      break;
    default:
      return ompd_rc_bad_input;
  }


  if (device_kind == OMPD_DEVICE_KIND_HOST) {
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
  } else if (device_kind == OMPD_DEVICE_KIND_CUDA) {
    switch (icv_id) {
      case ompd_icv_levels_var:
        return ompd_get_level_cuda((ompd_parallel_handle_t *)handle, icv_value);
      case ompd_icv_team_size_var:
        return ompd_get_num_threads_cuda((ompd_parallel_handle_t*)handle, icv_value);
      default:
        return ompd_rc_unsupported;
    }
  }
  return ompd_rc_unsupported;
}

ompd_rc_t
ompd_get_icv_string_from_scope(void *handle, ompd_scope_t scope,
                               ompd_icv_id_t icv_id,
                               const char **icv_string) {
  return ompd_rc_unsupported;
}


static ompd_rc_t 
__ompd_get_tool_data(TValue& dataValue,
                     ompd_word_t *value,
                     ompd_address_t *ptr) {
  ompd_rc_t ret = dataValue.getError();
  if (ret != ompd_rc_ok)
    return ret;
  ret = dataValue
            .access("value")
            .castBase()
            .getValue(*value);
  if (ret != ompd_rc_ok)
    return ret;
  ptr->segment = OMPD_SEGMENT_UNSPECIFIED;
  ret = dataValue
            .access("ptr")
            .castBase()
            .getValue(ptr->address);
  return ret;
}

ompd_rc_t ompd_get_task_data (ompd_task_handle_t *task_handle,
                                  ompd_word_t *value,
                                  ompd_address_t *ptr) {
  ompd_address_space_context_t *context = task_handle->ah->context;
  if (!context)
    return ompd_rc_stale_handle;

  assert(callbacks && "Callback table not initialized!");
  TValue dataValue;
  if (task_handle->lwt.address) {
    dataValue = TValue(context, task_handle->lwt)
              .cast("ompt_lw_taskteam_t")   /*lwt*/
              .access("ompt_task_info") // lwt->ompt_task_info
              .cast("ompt_task_info_t")
              .access("task_data") // lwt->ompd_task_info.task_data
              .cast("ompt_data_t");
  } else {
    dataValue = TValue(context, task_handle->th)
              .cast("kmp_taskdata_t")   /*td*/
              .access("ompt_task_info") // td->ompt_task_info
              .cast("ompt_task_info_t")
              .access("task_data") // td->ompd_task_info.task_data
              .cast("ompt_data_t");
  }
  return __ompd_get_tool_data(dataValue, value, ptr);
} 


ompd_rc_t ompd_get_parallel_data (ompd_parallel_handle_t *parallel_handle,
                                  ompd_word_t *value,
                                  ompd_address_t *ptr) {
  ompd_address_space_context_t *context = parallel_handle->ah->context;
  if (!context)
    return ompd_rc_stale_handle;

  assert(callbacks && "Callback table not initialized!");

  TValue dataValue;
  if (parallel_handle->lwt.address) {
    dataValue = TValue(context, parallel_handle->lwt)
              .cast("ompt_lw_taskteam_t")   /*lwt*/
              .access("ompt_team_info") // lwt->ompt_team_info
              .cast("ompt_team_info_t")
              .access("parallel_data") // lwt->ompt_team_info.parallel_data
              .cast("ompt_data_t");
  } else {
    dataValue = TValue(context, parallel_handle->th)
              .cast("kmp_base_team_t")   /*t*/
              .access("ompt_team_info") // t->ompt_team_info
              .cast("ompt_team_info_t")
              .access("parallel_data") // t->ompt_team_info.parallel_data
              .cast("ompt_data_t");
  }
  return __ompd_get_tool_data(dataValue, value, ptr);
} 

ompd_rc_t ompd_get_thread_data (ompd_thread_handle_t *thread_handle,
                                  ompd_word_t *value,
                                  ompd_address_t *ptr) {
  ompd_address_space_context_t *context = thread_handle->ah->context;
  if (!context)
    return ompd_rc_stale_handle;

  assert(callbacks && "Callback table not initialized!");

  TValue dataValue = TValue(context, thread_handle->th)
              .cast("kmp_base_info_t")   /*th*/
              .access("ompt_thread_info") // th->ompt_thread_info
              .cast("ompt_thread_info_t")
              .access("thread_data") // th->ompt_thread_info.thread_data
              .cast("ompt_data_t");
  return __ompd_get_tool_data(dataValue, value, ptr);
} 



ompd_rc_t ompd_get_tool_data (void *handle, ompd_scope_t scope,
                                  ompd_word_t *value,
                                  ompd_address_t *ptr) {
  if (!handle) {
    return ompd_rc_stale_handle;
  }

  ompd_device_t device_kind;

  switch (scope) {
    case ompd_scope_thread:
      device_kind = ((ompd_thread_handle_t *)handle)->ah->kind;
      break;
    case ompd_scope_parallel:
      device_kind = ((ompd_parallel_handle_t *)handle)->ah->kind;
      break;
    case ompd_scope_task:
      device_kind = ((ompd_task_handle_t *)handle)->ah->kind;
      break;
    default:
      return ompd_rc_bad_input;
  }


  if (device_kind == OMPD_DEVICE_KIND_HOST) {
    switch (scope) {
      case ompd_scope_thread:
        return ompd_get_thread_data((ompd_thread_handle_t *)handle, value, ptr);
      case ompd_scope_parallel:
        return ompd_get_parallel_data((ompd_parallel_handle_t *)handle, value, ptr);
      case ompd_scope_task:
        return ompd_get_task_data((ompd_task_handle_t *)handle, value, ptr);
      default:
        return ompd_rc_unsupported;
    }
  } else if (device_kind == OMPD_DEVICE_KIND_CUDA) {
    return ompd_rc_unsupported;
  }
  return ompd_rc_unsupported;
}



