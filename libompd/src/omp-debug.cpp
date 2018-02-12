/*
 * omp-debug.cpp
 *
 *  Created on: Jan 14, 2015
 *      Author: Ignacio Laguna
 *              Joachim Protze
 *     Contact: ilaguna@llnl.gov
 *              protze@llnl.gov
 */
/*******************************************************************************
 * This implements an OMPD DLL for the LLVM OpenMP runtime library.
 */

#define NDEBUG 1

#include "omp-debug.h"
#include "omp.h"
#include "ompd.h"
// #include <stdio.h>
#include "TargetValue.h"
#include <assert.h>
#include <cstdio>
#include <inttypes.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>

ompd_target_type_sizes_t type_sizes;
uint64_t ompd_state;

/* --- OMPD functions ------------------------------------------------------- */

/* --- 3 Initialization ----------------------------------------------------- */

ompd_rc_t ompd_initialize(const ompd_callbacks_t *table, ompd_word_t version) {
  ompd_rc_t ret = table ? ompd_rc_ok : ompd_rc_bad_input;
  callbacks = table;
  TValue::callbacks = table;

  return ret;
}

ompd_rc_t ompd_finalize(void) { return ompd_rc_ok; }

ompd_rc_t
ompd_process_initialize(ompd_address_space_context_t
                            *context, /* IN: debugger handle for the target */
                        ompd_address_space_handle_t *
                            *addrhandle /* OUT: ompd handle for the target */
                        ) {
  if (!context)
    return ompd_rc_bad_input;
  if (!addrhandle)
    return ompd_rc_bad_input;

  int rtl_version;
  ompd_rc_t ret = initTypeSizes(context);
  if (ret != ompd_rc_ok)
    return ret;
  ret = TValue(context, "ompd_rtl_version")
            .castBase(ompd_type_int)
            .getValue(rtl_version);
  if ((ret == ompd_rc_ok && rtl_version < 5) ||
      ret == ompd_rc_target_read_error)
    return ompd_rc_incompatible;
  if (ret != ompd_rc_ok)
    return ret;
  ret = TValue(context, "ompd_state")
            .castBase(ompd_type_long_long)
            .getValue(ompd_state);
  if (ret != ompd_rc_ok)
    return ret;
  *addrhandle = new ompd_address_space_handle_t;
  if (!addrhandle)
    return ompd_rc_error;
  (*addrhandle)->context = context;
  (*addrhandle)->kind = ompd_device_kind_host;

  return ompd_rc_ok;
}

ompd_rc_t
ompd_get_openmp_version(ompd_address_space_handle_t
                            *addr_handle, /* IN: handle for the address space */
                        ompd_word_t *version) {
  if (!addr_handle)
    return ompd_rc_stale_handle;
  ompd_address_space_context_t *context = addr_handle->context;
  ompd_rc_t ret;

  if (!context)
    return ompd_rc_stale_handle;

  assert(callbacks && "Callback table not initialized!");

  ret = TValue(context, "__kmp_openmp_version")
            .castBase(ompd_type_int)
            .getValue(*version);
  return ret;
}

ompd_rc_t ompd_get_openmp_version_string(
    ompd_address_space_handle_t
        *addr_handle, /* IN: handle for the address space */
    const char **string) {
  if (!addr_handle)
    return ompd_rc_bad_input;
  static const char *omp_version = "";
  *string = omp_version;
  return ompd_rc_ok;
}

ompd_rc_t ompd_release_address_space_handle(
    ompd_address_space_handle_t
        *addr_handle /* IN: handle for the address space */
    ) {
  if (!addr_handle)
    return ompd_rc_bad_input;

  delete addr_handle;
  return ompd_rc_ok;
}

#if 0  // no device support yet
ompd_rc_t ompd_device_initialize (
    ompd_address_space_context_t *context,  /* IN: */
    ompd_device_identifier_t id,            /* IN: object defined by native device API */
    ompd_device_kind_t kind,                /* IN: */
    ompd_address_space_handle_t **addrhandle  /* OUT: ompd handle for the device */
    )
{
  if (!context)
    return ompd_rc_bad_input;

  ompd_rc_t ret = initTypeSizes(context);
  if (ret != ompd_rc_ok)
    return ret;

  uint64_t ompd_num_cuda_devices;

  ompd_address_space_context_t *process_context;
  ret = callbacks->get_containing_process_context(context, &process_context);
  if ( ret != ompd_rc_ok )
    return ret;

  ret = TValue(process_context, "ompd_num_cuda_devices").
        castBase(ompd_type_long_long).
        getValue(ompd_num_cuda_devices);
  if (ret != ompd_rc_ok) {
    return ret;
  }

  for (uint64_t i = 0; i < ompd_num_cuda_devices; i++) {
    uint64_t cuda_ctx;

    /* TODO(mjm) - Hack!  Currently using ompt_parallel_id_t.  Need to find a
     * place to define ID type information for CUDA contexts
     */
    ret = TValue(process_context, "ompd_CudaContextArray").
          cast("ompt_parallel_id_t",1).           
          getArrayElement(i).
          castBase(ompd_type_long_long).
          getValue(cuda_ctx);

    if ( ret != ompd_rc_ok )
      continue;

    if (cuda_ctx == id) {
      *addrhandle = new ompd_address_space_handle_t;
      if (!addrhandle)
        return ompd_rc_error;
      (*addrhandle)->context = context;

      return ompd_rc_ok;
    }
  }

  /* TODO(mjm) - Find appropriate error return result for not finding a match */
  return ompd_rc_ok;
}
#endif // no device support

/* --- 4 Handle Management -------------------------------------------------- */

/* --- 4.1 Thread Handles --------------------------------------------------- */

/* thread_handle is of type (kmp_base_info_t) */

ompd_rc_t ompd_get_thread_in_parallel(
    ompd_parallel_handle_t *parallel_handle, /* IN: OpenMP parallel handle */
    int nth_handle, /* OUT: number of handles in the array */
    ompd_thread_handle_t **thread_handle /* OUT: handle */
    ) {
  if (!parallel_handle)
    return ompd_rc_stale_handle;
  if (!parallel_handle->ah)
    return ompd_rc_stale_handle;
  ompd_address_space_context_t *context = parallel_handle->ah->context;
  ompd_rc_t ret;
  int i;

  if (!context)
    return ompd_rc_stale_handle;

  assert(callbacks && "Callback table not initialized!");

  ompd_address_t taddr;

  ret = TValue(context, parallel_handle->th) /* t */
            .cast("kmp_base_team_t", 0)
            .access("t_threads") /*t.t_threads*/
            .cast("kmp_info_t", 2)
            .getArrayElement(nth_handle) /*t.t_threads[nth_handle]*/
            .access("th")                /*t.t_threads[i]->th*/
            .getAddress(&taddr);

  if (ret != ompd_rc_ok)
    return ret;
  ret = callbacks->dmemory_alloc(sizeof(ompd_thread_handle_t),
                                 (void **)(thread_handle));
  if (ret != ompd_rc_ok)
    return ret;

  (*thread_handle)->th = taddr;
  (*thread_handle)->ah = parallel_handle->ah;
  return ret;
}

ompd_rc_t ompd_release_thread_handle(
    ompd_thread_handle_t *thread_handle /* IN: OpenMP parallel handle */
    ) {
  if (!thread_handle)
    return ompd_rc_stale_handle;
  ompd_rc_t ret = callbacks->dmemory_free((void *)(thread_handle));
  if (ret != ompd_rc_ok)
    return ret;
  return ompd_rc_ok;
}

ompd_rc_t ompd_thread_handle_compare(ompd_thread_handle_t *thread_handle_1,
                                     ompd_thread_handle_t *thread_handle_2,
                                     int *cmp_value) {
  if (!thread_handle_1)
    return ompd_rc_stale_handle;
  if (!thread_handle_2)
    return ompd_rc_stale_handle;
  *cmp_value = thread_handle_1->th.address - thread_handle_2->th.address;
  return ompd_rc_ok;
}

#if 0
ompd_rc_t ompd_get_thread_handle_string_id (
    ompd_thread_handle_t *thread_handle,
    char **string_id
    )
{
  pthread_t thread_id;
  ompd_rc_t ret;
  ret = ompd_get_thread_id(thread_handle, ompd_thread_id_pthread, sizeof(pthread_t), &thread_id);
  if (ret!=ompd_rc_ok)
    return ret;
  ret = callbacks->dmemory_alloc(sizeof(void*)*2+3, (void**)string_id);
  if (ret!=ompd_rc_ok)
    return ret;
  sprintf(*string_id, "0x%llx", (long long)thread_id);
  return ompd_rc_ok;
}
#endif

/* --- 4.2 Parallel Region Handles------------------------------------------- */

/* parallel_handle is of type (kmp_base_team_t)*/

ompd_rc_t ompd_get_current_parallel_handle(
    ompd_thread_handle_t *thread_handle,     /* IN: OpenMP thread handle*/
    ompd_parallel_handle_t **parallel_handle /* OUT: OpenMP parallel handle */
    ) {
  if (!thread_handle)
    return ompd_rc_stale_handle;
  if (!thread_handle->ah)
    return ompd_rc_stale_handle;
  ompd_address_space_context_t *context = thread_handle->ah->context;
  if (!context)
    return ompd_rc_stale_handle;

  assert(callbacks && "Callback table not initialized!");
  ompd_address_t taddr, lwt;

  TValue teamdata = TValue(context, thread_handle->th) /*__kmp_threads[t]->th*/
                        .cast("kmp_base_info_t")
                        .access("th_team") /*__kmp_threads[t]->th.th_team*/
                        .cast("kmp_team_p", 1)
                        .access("t"); /*__kmp_threads[t]->th.th_team->t*/

  ompd_rc_t ret = teamdata.getAddress(&taddr);
  if (ret != ompd_rc_ok)
    return ret;

  lwt.segment = OMPD_SEGMENT_UNSPECIFIED;
  ret = teamdata.cast("kmp_base_team_t", 0)
            .access("ompt_serialized_team_info")
            .castBase()
            .getValue(lwt.address);
  if (ret != ompd_rc_ok)
    return ret;

  ret = callbacks->dmemory_alloc(sizeof(ompd_parallel_handle_t),
                                 (void **)(parallel_handle));
  if (ret != ompd_rc_ok)
    return ret;

  (*parallel_handle)->ah = thread_handle->ah;
  (*parallel_handle)->th = taddr;
  (*parallel_handle)->lwt = lwt;
  return ompd_rc_ok;
}

ompd_rc_t ompd_get_enclosing_parallel_handle(
    ompd_parallel_handle_t *parallel_handle, /* IN: OpenMP parallel handle */
    ompd_parallel_handle_t *
        *enclosing_parallel_handle /* OUT: OpenMP parallel handle */
    ) {
  if (!parallel_handle)
    return ompd_rc_stale_handle;
  if (!parallel_handle->ah)
    return ompd_rc_stale_handle;
  ompd_address_space_context_t *context = parallel_handle->ah->context;

  if (!context)
    return ompd_rc_stale_handle;

  assert(callbacks && "Callback table not initialized!");
  ompd_address_t taddr = parallel_handle->th, lwt;

  ompd_rc_t ret = ompd_rc_stale_handle;
  TValue lwtValue = TValue(context, parallel_handle->lwt);
  if (lwtValue.getError() == ompd_rc_ok) // lwt == 0x0
  {                                      // if we are in lwt, get parent
    ret = lwtValue.cast("ompt_lw_taskteam_t", 0)
              .access("parent")
              .cast("ompt_lw_taskteam_t", 1)
              .dereference()
              .getAddress(&lwt);
  }
  if (ret != ompd_rc_ok) { // no lwt or parent==0x0

    TValue teamdata =
        TValue(context, parallel_handle->th) /*__kmp_threads[t]->th*/
            .cast("kmp_base_team_t", 0)      /*t*/
            .access("t_parent")              /*t.t_parent*/
            .cast("kmp_team_p", 1)
            .access("t"); /*t.t_parent->t*/

    ret = teamdata.getAddress(&taddr);
    if (ret != ompd_rc_ok)
      return ret;

    lwt.segment = OMPD_SEGMENT_UNSPECIFIED;
    ret = teamdata.cast("kmp_base_team_t", 0)
              .access("ompt_serialized_team_info")
              .castBase()
              .getValue(lwt.address);
    if (ret != ompd_rc_ok)
      return ret;
  }

  ret = callbacks->dmemory_alloc(sizeof(ompd_parallel_handle_t),
                                 (void **)(enclosing_parallel_handle));
  if (ret != ompd_rc_ok)
    return ret;
  (*enclosing_parallel_handle)->th = taddr;
  (*enclosing_parallel_handle)->lwt = lwt;
  (*enclosing_parallel_handle)->ah = parallel_handle->ah;
  return ompd_rc_ok;
}

ompd_rc_t ompd_get_task_parallel_handle(
    ompd_task_handle_t *task_handle, /* IN: OpenMP task handle */
    ompd_parallel_handle_t *
        *enclosing_parallel_handle /* OUT: OpenMP parallel handle */
    ) {
  if (!task_handle)
    return ompd_rc_stale_handle;
  if (!task_handle->ah)
    return ompd_rc_stale_handle;
  ompd_address_space_context_t *context = task_handle->ah->context;

  if (!context)
    return ompd_rc_stale_handle;

  assert(callbacks && "Callback table not initialized!");
  ompd_address_t taddr;

  ompd_rc_t ret;
  ret = TValue(context, task_handle->th)
            .cast("kmp_taskdata_t") /*td*/
            .access("td_team")      /*td.td_team*/
            .cast("kmp_team_p", 1)
            .access("t") /*td.td_team->t*/
            .getAddress(&taddr);

  if (ret != ompd_rc_ok)
    return ret;

  ret = callbacks->dmemory_alloc(sizeof(ompd_parallel_handle_t),
                                 (void **)(enclosing_parallel_handle));
  if (ret != ompd_rc_ok)
    return ret;

  (*enclosing_parallel_handle)->ah = task_handle->ah;
  (*enclosing_parallel_handle)->lwt = task_handle->lwt;
  (*enclosing_parallel_handle)->th = taddr;
  return ompd_rc_ok;
}

ompd_rc_t ompd_release_parallel_handle(
    ompd_parallel_handle_t *parallel_handle /* IN: OpenMP parallel handle */
    ) {
  if (!parallel_handle)
    return ompd_rc_stale_handle;
  ompd_rc_t ret = callbacks->dmemory_free((void *)(parallel_handle));
  if (ret != ompd_rc_ok)
    return ret;
  return ompd_rc_ok;
}

ompd_rc_t
ompd_parallel_handle_compare(ompd_parallel_handle_t *parallel_handle_1,
                             ompd_parallel_handle_t *parallel_handle_2,
                             int *cmp_value) {
  if (!parallel_handle_1)
    return ompd_rc_stale_handle;
  if (!parallel_handle_2)
    return ompd_rc_stale_handle;
  if (parallel_handle_1->th.address - parallel_handle_2->th.address)
    *cmp_value = parallel_handle_1->th.address - parallel_handle_2->th.address;
  else
    *cmp_value =
        parallel_handle_1->lwt.address - parallel_handle_2->lwt.address;
  return ompd_rc_ok;
}

#if 0 // parallel-id is initialized to zero
ompd_rc_t ompd_get_parallel_handle_string_id (
    ompd_parallel_handle_t *parallel_handle,
    char **string_id
    )
{
    ompd_parallel_id_t id;
    ompd_rc_t ret;
    ret = ompd_get_parallel_id(parallel_handle, &id);
    if (ret!=ompd_rc_ok)
      return ret;
    ret = callbacks->dmemory_alloc(sizeof(void*)*2+3, (void**)string_id);
    if (ret!=ompd_rc_ok)
      return ret;
    sprintf(*string_id, "0x%llx", (long long)id);
    return ompd_rc_ok;
}
#endif

/* --- 4.3 Task Handles ----------------------------------------------------- */

/* task_handle is of type (kmp_taskdata_t) */

ompd_rc_t ompd_get_current_task__handle(
    ompd_thread_handle_t *thread_handle, /* IN: OpenMP thread handle*/
    ompd_task_handle_t **task_handle     /* OUT: OpenMP task handle */
    ) {
  if (!thread_handle)
    return ompd_rc_stale_handle;
  if (!thread_handle->ah)
    return ompd_rc_stale_handle;
  ompd_address_space_context_t *context = thread_handle->ah->context;
  if (!context)
    return ompd_rc_stale_handle;

  assert(callbacks && "Callback table not initialized!");
  ompd_address_t taddr, lwt;

  TValue taskdata =
      TValue(context, thread_handle->th) /*__kmp_threads[t]->th*/
          .cast("kmp_base_info_t")
          .access("th_current_task") /*__kmp_threads[t]->th.th_current_task*/
          .cast("kmp_taskdata_t", 1);

  ompd_rc_t ret = taskdata.dereference().getAddress(&taddr);
  if (ret != ompd_rc_ok)
    return ret;

  lwt.segment = OMPD_SEGMENT_UNSPECIFIED;
  ret = taskdata
            .access("td_team") /*td.td_team*/
            .cast("kmp_team_p", 1)
            .access("t") /*td.td_team->t*/
            .cast("kmp_base_team_t", 0)
            .access("ompt_serialized_team_info")
            .castBase()
            .getValue(lwt.address);
  if (ret != ompd_rc_ok)
    return ret;

  ret = callbacks->dmemory_alloc(sizeof(ompd_task_handle_t),
                                 (void **)(task_handle));
  if (ret != ompd_rc_ok)
    return ret;

  (*task_handle)->th = taddr;
  (*task_handle)->lwt = lwt;
  (*task_handle)->ah = thread_handle->ah;
  return ompd_rc_ok;
}

ompd_rc_t ompd_get_generating_ancestor_task_handle(
    ompd_task_handle_t *task_handle,        /* IN: OpenMP task handle */
    ompd_task_handle_t **parent_task_handle /* OUT: OpenMP task handle */
    ) {
  if (!task_handle)
    return ompd_rc_stale_handle;
  if (!task_handle->ah)
    return ompd_rc_stale_handle;
  ompd_address_space_context_t *context = task_handle->ah->context;
  if (!context)
    return ompd_rc_stale_handle;

  assert(callbacks && "Callback table not initialized!");
  ompd_address_t taddr = task_handle->th, lwt;

  ompd_rc_t ret = ompd_rc_stale_handle;
  TValue lwtValue = TValue(context, task_handle->lwt);
  if (lwtValue.getError() == ompd_rc_ok) // lwt == 0x0
  {                                      // if we are in lwt, get parent
    ret = lwtValue.cast("ompt_lw_taskteam_t", 0)
              .access("parent")
              .cast("ompt_lw_taskteam_t", 1)
              .dereference()
              .getAddress(&lwt);
  }
  if (ret != ompd_rc_ok) { // no lwt or parent==0x0

    TValue taskdata = TValue(context, task_handle->th) /*__kmp_threads[t]->th*/
                          .cast("kmp_taskdata_t")      /*td*/
                          .access("td_parent")         /*td->td_parent*/
                          .cast("kmp_taskdata_t", 1);

    ret = taskdata.dereference().getAddress(&taddr);
    if (ret != ompd_rc_ok)
      return ret;

    lwt.segment = OMPD_SEGMENT_UNSPECIFIED;
    ret = taskdata
              .access("td_team") /*td.td_team*/
              .cast("kmp_team_p", 1)
              .access("t") /*td.td_team->t*/
              .cast("kmp_base_team_t", 0)
              .access("ompt_serialized_team_info")
              .castBase()
              .getValue(lwt.address);
    if (ret != ompd_rc_ok)
      return ret;
  }

  ret = callbacks->dmemory_alloc(sizeof(ompd_task_handle_t),
                                 (void **)(parent_task_handle));
  if (ret != ompd_rc_ok)
    return ret;

  (*parent_task_handle)->th = taddr;
  (*parent_task_handle)->lwt = lwt;
  (*parent_task_handle)->ah = task_handle->ah;
  return ret;
}

ompd_rc_t ompd_get_scheduling_ancestor_task_handle(
    ompd_task_handle_t *task_handle,        /* IN: OpenMP task handle */
    ompd_task_handle_t **parent_task_handle /* OUT: OpenMP task handle */
    ) {
  if (!task_handle)
    return ompd_rc_stale_handle;
  if (!task_handle->ah)
    return ompd_rc_stale_handle;
  ompd_address_space_context_t *context = task_handle->ah->context;
  if (!context)
    return ompd_rc_stale_handle;

  assert(callbacks && "Callback table not initialized!");
  ompd_address_t taddr;

  ompd_rc_t ret =
      TValue(context, task_handle->th)
          .cast("kmp_taskdata_t")   /*td*/
          .access("ompt_task_info") // td->ompt_task_info
          .cast("ompt_task_info_t")
          .access("scheduling_parent") // td->ompd_task_info.scheduling_parent
          .cast("kmp_taskdata_t", 1)
          .dereference()
          .getAddress(&taddr);

  if (ret != ompd_rc_ok)
    return ret;
  ret = callbacks->dmemory_alloc(sizeof(ompd_task_handle_t),
                                 (void **)(parent_task_handle));
  if (ret != ompd_rc_ok)
    return ret;

  (*parent_task_handle)->th = taddr;
  (*parent_task_handle)->ah = task_handle->ah;
  return ret;
}

ompd_rc_t ompd_get_task_in_parallel(
    ompd_parallel_handle_t *parallel_handle, /* IN: OpenMP parallel handle */
    int nth_handle,                  /* OUT: number of the task handle */
    ompd_task_handle_t **task_handle /* OUT: OpenMP task handle */
    ) {
  int i;
  if (!parallel_handle)
    return ompd_rc_stale_handle;
  if (!parallel_handle->ah)
    return ompd_rc_stale_handle;
  ompd_address_space_context_t *context = parallel_handle->ah->context;
  if (!context)
    return ompd_rc_stale_handle;

  assert(callbacks && "Callback table not initialized!");

  ompd_rc_t ret;
  ompd_address_t taddr;
  ret = TValue(context, parallel_handle->th) /* t */
            .cast("kmp_base_team_t", 0)
            .access("t_implicit_task_taskdata") /*t.t_implicit_task_taskdata*/
            .cast("kmp_taskdata_t", 1)
            .getArrayElement(
                nth_handle) /*t.t_implicit_task_taskdata[nth_handle]*/
            .getAddress(&taddr);

  if (ret != ompd_rc_ok)
    return ret;
  ret = callbacks->dmemory_alloc(sizeof(ompd_task_handle_t),
                                 (void **)(task_handle));
  if (ret != ompd_rc_ok)
    return ret;

  (*task_handle)->th = taddr;
  (*task_handle)->ah = parallel_handle->ah;
  return ret;
}

ompd_rc_t ompd_release_task_handle(
    ompd_task_handle_t *task_handle /* IN: OpenMP task handle */
    ) {
  if (!task_handle)
    return ompd_rc_stale_handle;
  ompd_rc_t ret = callbacks->dmemory_free((void *)(task_handle));
  if (ret != ompd_rc_ok)
    return ret;
  return ompd_rc_ok;
}

ompd_rc_t ompd_task_handle_compare(ompd_task_handle_t *task_handle_1,
                                   ompd_task_handle_t *task_handle_2,
                                   int *cmp_value) {
  if (!task_handle_1)
    return ompd_rc_stale_handle;
  if (!task_handle_2)
    return ompd_rc_stale_handle;
  if (task_handle_1->th.address - task_handle_2->th.address)
    *cmp_value = task_handle_1->th.address - task_handle_2->th.address;
  else
    *cmp_value = task_handle_1->lwt.address - task_handle_2->lwt.address;
  return ompd_rc_ok;
}

#if 0 // all task ids are initialized to zero
ompd_rc_t ompd_get_task_handle_string_id (
    ompd_task_handle_t *task_handle,
    char **string_id
    )
{
    ompd_task_id_t id;
    ompd_rc_t ret = ompd_get_task_id(task_handle, &id);
    if (ret!=ompd_rc_ok)
      return ret;
    ret = callbacks->dmemory_alloc(sizeof(void*)*2+3, (void**)string_id);
    if (ret!=ompd_rc_ok)
      return ret;
    sprintf(*string_id, "0x%llx", (long long)id);
    return ompd_rc_ok;
}
#endif

/* --- 5 Process and Thread Settings ---------------------------------------- */

ompd_rc_t
ompd_get_num_procs(ompd_address_space_handle_t
                       *addr_handle, /* IN: handle for the address space */
                   ompd_word_t *val  /* OUT: number of processes */
                   ) {
  if (!addr_handle)
    return ompd_rc_stale_handle;
  ompd_address_space_context_t *context = addr_handle->context;
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

ompd_rc_t
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

/* --- 6 Parallel Region Inqueries ------------------------------------------ */
/* --- 6.1 Settings --------------------------------------------------------- */

ompd_rc_t ompd_get_num_threads(
    ompd_parallel_handle_t *parallel_handle, /* IN: OpenMP parallel handle */
    ompd_word_t *val                         /* OUT: number of threads */
    ) {
  if (!parallel_handle)
    return ompd_rc_stale_handle;
  if (!parallel_handle->ah)
    return ompd_rc_stale_handle;
  ompd_address_space_context_t *context = parallel_handle->ah->context;
  if (!context)
    return ompd_rc_stale_handle;

  assert(callbacks && "Callback table not initialized!");

  ompd_rc_t ret = ompd_rc_ok;
  if (parallel_handle->lwt.address != 0)
    *val = 1;
  else {
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

ompd_rc_t ompd_get_level(
    ompd_parallel_handle_t *parallel_handle, /* IN: OpenMP parallel handle */
    ompd_word_t *val                         /* OUT: nesting level */
    ) {
  if (!parallel_handle)
    return ompd_rc_stale_handle;
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

ompd_rc_t ompd_get_active_level(
    ompd_parallel_handle_t *parallel_handle, /* IN: OpenMP parallel handle */
    ompd_word_t *val                         /* OUT: active nesting level */
    ) {
  if (!parallel_handle)
    return ompd_rc_stale_handle;
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

/* --- 6.2 OMPT Parallel Region Inquiry Analogues ------------------------- */

ompd_rc_t ompd_get_parallel_data(
    ompd_parallel_handle_t *parallel_handle, /* IN: OpenMP parallel handle */
    ompd_address_t *data                     /* OUT: OpenMP parallel id */
    ) {
  if (!parallel_handle)
    return ompd_rc_stale_handle;
  if (!parallel_handle->ah)
    return ompd_rc_stale_handle;
  ompd_address_space_context_t *context = parallel_handle->ah->context;
  if (!context)
    return ompd_rc_stale_handle;
  if (!ompd_state)
    return ompd_rc_needs_state_tracking;

  assert(callbacks && "Callback table not initialized!");

  TValue teamInfo;
  if (parallel_handle->lwt.address != 0)
    teamInfo = TValue(context, parallel_handle->lwt)
                   .cast("ompt_lw_taskteam_t", 0); /*lwt*/
  else
    teamInfo =
        TValue(context, parallel_handle->th).cast("kmp_base_team_t", 0); /*t*/
  ompd_rc_t ret = teamInfo
                      .access("ompt_team_info") /*t.ompt_team_info*/
                      .cast("ompt_team_info_t", 0)
                      .access("parallel_data") /*t.ompt_team_info.parallel_id*/
                      .getAddress(data);
  return ret;
}

#if 0  // there is no such thing as a parallel function
ompd_rc_t ompd_get_parallel_function(
    ompd_parallel_handle_t *parallel_handle, /* IN: OpenMP parallel handle */
    ompd_address_t *parallel_addr /* OUT: first instruction in the parallel region */
    )
{
  if (!parallel_handle)
    return ompd_rc_stale_handle;
  if (!parallel_handle->ah)
    return ompd_rc_stale_handle;
  ompd_address_space_context_t *context = parallel_handle->ah->context;
  if (!context)
    return ompd_rc_stale_handle;
  if (!ompd_state)
    return ompd_rc_needs_state_tracking;

  assert(callbacks && "Callback table not initialized!");
  parallel_addr->segment = OMPD_SEGMENT_UNSPECIFIED;

  TValue teamInfo;
  if(parallel_handle->lwt.address!=0)
    teamInfo = TValue(context, parallel_handle->lwt).
          cast("ompt_lw_taskteam_t",0);		/*lwt*/
  else
    teamInfo = TValue(context, parallel_handle->th).
          cast("kmp_base_team_t",0);		/*t*/
  ompd_rc_t ret = teamInfo.
        access("ompt_team_info").             /*t.ompt_team_info*/
        cast("ompt_team_info_t",0).
        access("microtask").                /*t.ompt_team_info.microtask*/
        castBase().    
        getValue(parallel_addr->address);
  return ret;
}
#endif // no parallel function

/* --- 7 Thread Inquiry ----------------------------------------------------- */

/* --- 7.1 Operating System Thread Inquiry ---------------------------------- */

ompd_rc_t
ompd_get_thread_handle(ompd_address_space_handle_t
                           *addr_handle, /* IN: handle for the address space */
                       ompd_thread_id_kind_t kind,
                       ompd_size_t sizeof_thread_id, const void *thread_id,
                       ompd_thread_handle_t **thread_handle) {
  if (!addr_handle)
    return ompd_rc_stale_handle;
  ompd_address_space_context_t *context = addr_handle->context;
  ompd_rc_t ret;

  if (!context)
    return ompd_rc_stale_handle;

  assert(callbacks && "Callback table not initialized!");
  ompd_thread_context_t *tcontext;
  ret = callbacks->get_thread_context_for_thread_id(
      context, kind, sizeof_thread_id, thread_id, &tcontext);
  if (ret != ompd_rc_ok)
    return ret;

  int tId;

  if (kind == ompd_thread_id_cudalogical) {
    ompd_cudathread_coord_t *p = (ompd_cudathread_coord_t *)thread_id;

    // omptarget_nvptx_threadPrivateContext->topTaskDescr[p->threadIdx.x]->data.items.threadId

    ret =
        TValue(context, tcontext, "omptarget_nvptx_threadPrivateContext",
               OMPD_SEGMENT_CUDA_PTX_SHARED)
            .cast("omptarget_nvptx_ThreadPrivateContext", 1,
                  OMPD_SEGMENT_CUDA_PTX_SHARED)
            .access("topTaskDescr")
            .cast("omptarget_nvptx_TaskDescr", 1, OMPD_SEGMENT_CUDA_PTX_GLOBAL)
            .getArrayElement(p->threadIdx.x)
            .access("data__items__threadId")
            .castBase(ompd_type_short)
            .getValue(tId);

    if (ret != ompd_rc_ok)
      return ret;

    if (tId != p->threadIdx.x)
      return ompd_rc_stale_handle;
  } else {
    ret = TValue(context, tcontext, "__kmp_gtid")
              .castBase("__kmp_gtid")
              .getValue(tId);
    if (ret != ompd_rc_ok)
      return ret;

    if (tId < 0) // thread is no omp worker
      return ompd_rc_unavailable;

    TValue th = TValue(context, "__kmp_threads") // __kmp_threads
                    .cast("kmp_info_t", 2)
                    .getArrayElement(tId) /*__kmp_threads[t]*/
                    .access("th");        /*__kmp_threads[t]->th*/

    ompd_address_t taddr;
    ret = th.getAddress(&taddr);
    if (ret != ompd_rc_ok)
      return ret;
    ret = callbacks->dmemory_alloc(sizeof(ompd_thread_handle_t),
                                   (void **)(thread_handle));
    if (ret != ompd_rc_ok)
      return ret;
    (*thread_handle)->ah = addr_handle;
    (*thread_handle)->th = taddr;

#ifndef NDEBUG
    if (ret != ompd_rc_ok)
      return ret;

    pthread_t oshandle;
    TBaseValue ds_handle =
        th.cast("kmp_base_info_t")
            .access("th_info") /*__kmp_threads[t]->th.th_info*/
            .cast("kmp_desc_t")
            .access("ds") /*__kmp_threads[t]->th.th_info.ds*/
            .cast("kmp_desc_base_t")
            .access("ds_thread") /*__kmp_threads[t]->th.th_info.ds.ds_thread*/
            .castBase();

    assert(ompd_rc_ok == ds_handle.getValue(oshandle) &&
           oshandle == *(pthread_t *)(thread_id) &&
           "Callback table not initialized!");
#endif
  }
  return ret;
}

ompd_rc_t ompd_get_thread_id(
    ompd_thread_handle_t *thread_handle, /* IN: OpenMP thread handle*/
    ompd_thread_id_kind_t kind, ompd_size_t sizeof_thread_id, void *thread_id) {
  if (kind != ompd_thread_id_pthread)
    return ompd_rc_bad_input;
  if (!thread_handle)
    return ompd_rc_stale_handle;
  if (!thread_handle->ah)
    return ompd_rc_stale_handle;
  ompd_address_space_context_t *context = thread_handle->ah->context;
  if (!context)
    return ompd_rc_stale_handle;
  ompd_size_t size;
  ompd_rc_t ret = tf.getType(context, "kmp_thread_t").getSize(&size);
  if (ret != ompd_rc_ok)
    return ret;
  if (sizeof_thread_id != size)
    return ompd_rc_bad_input;

  assert(callbacks && "Callback table not initialized!");

  ret = TValue(context, thread_handle->th) /*__kmp_threads[t]->th*/
            .cast("kmp_base_info_t")
            .access("th_info") /*__kmp_threads[t]->th.th_info*/
            .cast("kmp_desc_t")
            .access("ds") /*__kmp_threads[t]->th.th_info.ds*/
            .cast("kmp_desc_base_t")
            .access("ds_thread") /*__kmp_threads[t]->th.th_info.ds.ds_thread*/
            .cast("kmp_thread_t")
            .getRawValue(thread_id, 1);
  return ret;
}

ompd_rc_t ompd_get_thread_num(
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

/* --- 7.2 OMPT Thread State Inquiry Analogue ------------------------------- */

ompd_rc_t ompd_get_state(
    ompd_thread_handle_t *thread_handle, /* IN: OpenMP thread handle*/
    ompd_word_t *state,                  /* OUT: State of this thread */
    ompd_wait_id_t *wait_id              /* OUT: Wait ID */
    ) {
  if (!thread_handle)
    return ompd_rc_stale_handle;
  if (!thread_handle->ah)
    return ompd_rc_stale_handle;
  ompd_address_space_context_t *context = thread_handle->ah->context;
  if (!context)
    return ompd_rc_stale_handle;
  if (!ompd_state)
    return ompd_rc_needs_state_tracking;

  assert(callbacks && "Callback table not initialized!");

  TValue ompt_thread_info =
      TValue(context, thread_handle->th) /*__kmp_threads[t]->th*/
          .cast("kmp_base_info_t")
          .access("ompt_thread_info") /*__kmp_threads[t]->th.ompt_thread_info*/
          .cast("ompt_thread_info_t");
  if (ompt_thread_info.gotError())
    return ompt_thread_info.getError();
  ompd_rc_t ret =
      ompt_thread_info
          .access("state") /*__kmp_threads[t]->th.ompt_thread_info.state*/
          .castBase()
          .getValue(*state);
  if (ret != ompd_rc_ok)
    return ret;
  ret = ompt_thread_info
            .access("wait_id") /*__kmp_threads[t]->th.ompt_thread_info.state*/
            .castBase()
            .getValue(*wait_id);
  return ret;
}

/* --- 8 Task Inquiry ------------------------------------------------------- */

/* --- 8.1 Task Settings ---------------------------------------------------- */

ompd_rc_t ompd_get_max_threads(
    ompd_task_handle_t *task_handle, /* IN: OpenMP task handle*/
    ompd_word_t *val                 /* OUT: max number of threads */
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
                      .access("td_icvs")      // td->td_icvs
                      .cast("kmp_internal_control_t", 0)
                      .access("nproc") // td->td_icvs.dynamic
                      .castBase()
                      .getValue(*val);

  return ret;
}

ompd_rc_t ompd_in_parallel( // Why do we need a task context for _in_parallel?
    ompd_task_handle_t *task_handle, /* IN: OpenMP task handle*/
    ompd_word_t *val                 /* OUT: max number of threads */
    ) {
  if (!task_handle)
    return ompd_rc_stale_handle;
  if (!task_handle->ah)
    return ompd_rc_stale_handle;
  ompd_address_space_context_t *context = task_handle->ah->context;
  if (!context)
    return ompd_rc_stale_handle;
  ompd_rc_t ret;

  assert(callbacks && "Callback table not initialized!");

  ret = TValue(context, "__kmp_root") // __kmp_root
            .cast("kmp_root_t", 2)
            .dereference() // (*__kmp_root)
            .access("r")   // (*__kmp_root)->r
            .cast("kmp_base_root_t")
            .access("r_in_parallel") // (*__kmp_root)->r.r_in_parallel
            .castBase()
            .getValue(*val);
  if (ret != ompd_rc_ok)
    return ret;
  if (*val)
    *val = 1;

  return ret;
}

ompd_rc_t
ompd_in_final(ompd_task_handle_t *task_handle, /* IN: OpenMP task handle*/
              ompd_word_t *val                 /* OUT: max number of threads */
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
                      .access("td_flags")     // td->td_icvs
                      .cast("kmp_tasking_flags_t")
                      .check("final", val); // td->td_icvs.max_active_levels

  return ret;
}

ompd_rc_t
ompd_get_dynamic(ompd_task_handle_t *task_handle, /* IN: OpenMP task handle*/
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
                      .access("td_icvs")      // td->td_icvs
                      .cast("kmp_internal_control_t", 0)
                      .access("dynamic") // td->td_icvs.dynamic
                      .castBase()
                      .getValue(*val);

  return ret;
}

ompd_rc_t
ompd_get_nested(ompd_task_handle_t *task_handle, /* IN: OpenMP task handle*/
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
                      .access("td_icvs")      // td->td_icvs
                      .cast("kmp_internal_control_t", 0)
                      .access("nested") // td->td_icvs.nested
                      .castBase()
                      .getValue(*val);

  return ret;
}

ompd_rc_t ompd_get_max_active_levels(
    ompd_task_handle_t *task_handle, /* IN: OpenMP task handle*/
    ompd_word_t *val                 /* OUT: max number of threads */
    ) {
  if (!task_handle)
    return ompd_rc_stale_handle;
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

ompd_rc_t
ompd_get_schedule(ompd_task_handle_t *task_handle, /* IN: OpenMP task handle*/
                  ompd_word_t *kind,    /* OUT: Kind of OpenMP schedule*/
                  ompd_word_t *modifier /* OUT: Schedunling modifier */
                  ) {
  if (!task_handle)
    return ompd_rc_stale_handle;
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

ompd_rc_t
ompd_get_proc_bind(ompd_task_handle_t *task_handle, /* IN: OpenMP task handle*/
                   ompd_word_t *bind /* OUT: Kind of proc-binding */
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
                      .access("td_icvs")      // td->td_icvs
                      .cast("kmp_internal_control_t", 0)
                      .access("proc_bind") // td->td_icvs.proc_bind
                      .castBase()
                      .getValue(*bind);

  return ret;
}

ompd_rc_t
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

/* --- 8.2 OMPT Task Inquiry Analogues -------------------------------------- */

ompd_rc_t ompd_get_task_frame(
    ompd_task_handle_t *task_handle, /* IN: OpenMP task handle*/
    ompd_address_t *sp_exit,         /* OUT: next frame is user code */
    ompd_address_t *sp_reentry       /* OUT: previous frame is user code */
    ) {
  if (!task_handle)
    return ompd_rc_stale_handle;
  if (!task_handle->ah)
    return ompd_rc_stale_handle;
  ompd_address_space_context_t *context = task_handle->ah->context;
  if (!context)
    return ompd_rc_stale_handle;
  if (!ompd_state)
    return ompd_rc_needs_state_tracking;

  assert(callbacks && "Callback table not initialized!");

  TValue taskInfo;
  if (task_handle->lwt.address != 0)
    taskInfo =
        TValue(context, task_handle->lwt).cast("ompt_lw_taskteam_t", 0); /*lwt*/
  else
    taskInfo = TValue(context, task_handle->th).cast("kmp_taskdata_t", 0); /*t*/
  TValue frame = taskInfo
                     .access("ompt_task_info") // td->ompt_task_info
                     .cast("ompt_task_info_t")
                     .access("frame") // td->ompd_task_info.frame
                     .cast("ompt_frame_t", 0);
  sp_reentry->segment = OMPD_SEGMENT_UNSPECIFIED;
  ompd_rc_t ret =
      frame
          .access("enter_frame") // td->ompt_task_info.frame.enter_frame
          .castBase()
          .getValue(sp_reentry->address);

  if (ret != ompd_rc_ok)
    return ret;

  sp_exit->segment = OMPD_SEGMENT_UNSPECIFIED;
  ret = frame
            .access("exit_frame") // td->ompt_task_info.frame.exit_frame
            .castBase()
            .getValue(sp_exit->address);

  return ret;
}

ompd_rc_t
ompd_get_task_data(ompd_task_handle_t *task_handle, /* IN: OpenMP task handle*/
                   ompd_address_t *task_data        /* OUT: OpenMP task ID */
                   ) {
  if (!task_handle)
    return ompd_rc_stale_handle;
  if (!task_handle->ah)
    return ompd_rc_stale_handle;
  ompd_address_space_context_t *context = task_handle->ah->context;
  if (!context)
    return ompd_rc_stale_handle;
  if (!ompd_state)
    return ompd_rc_needs_state_tracking;

  assert(callbacks && "Callback table not initialized!");

  TValue taskInfo;
  if (task_handle->lwt.address != 0)
    taskInfo =
        TValue(context, task_handle->lwt).cast("ompt_lw_taskteam_t", 0); /*lwt*/
  else
    taskInfo = TValue(context, task_handle->th).cast("kmp_taskdata_t", 0); /*t*/
  ompd_rc_t ret = taskInfo
                      .access("ompt_task_info") // td->ompt_task_info
                      .cast("ompt_task_info_t")
                      .access("task_data") // td->ompt_task_info.task_data
                      .getAddress(task_data);

  return ret;
}

#if 0 // the runtime currently does not have task function information
ompd_rc_t ompd_get_task_function(
    ompd_task_handle_t *task_handle, /* IN: OpenMP task handle */
    ompd_address_t *task_addr /* OUT: first instruction in the task region */
    )
{
  if (!task_handle)
    return ompd_rc_stale_handle;
  if (!task_handle->ah)
    return ompd_rc_stale_handle;
  ompd_address_space_context_t *context = task_handle->ah->context;
  if (!context)
    return ompd_rc_stale_handle;
  if (!ompd_state)
    return ompd_rc_needs_state_tracking;

  assert(callbacks && "Callback table not initialized!");

#if 0
  /* We don't have a task function for implicit tasks */
  ompd_word_t implicit;
  ompd_rc_t ret = ompd_is_implicit (task_handle, &implicit);
  if (ret != ompd_rc_ok)
    return ret;
  if (implicit)
    return ompd_rc_bad_input;
#else
  ompd_rc_t ret;
#endif  
  task_addr->segment = OMPD_SEGMENT_UNSPECIFIED;
  TValue taskInfo;
  if(task_handle->lwt.address!=0)
    taskInfo = TValue(context, task_handle->lwt).
          cast("ompt_lw_taskteam_t",0);		/*lwt*/
  else
    taskInfo = TValue(context, task_handle->th).
          cast("kmp_taskdata_t",0);		/*t*/
  ret = taskInfo.
        access("ompt_task_info").             /*td->ompt_task_info*/
        cast("ompt_task_info_t").
        access("function").                /*td->ompt_task_info.function*/
        castBase().    
        getValue(task_addr->address);
  return ret;
}
#endif

/* --- 9 OMPD Version and Compatibility Information ------------------------- */

ompd_rc_t ompd_get_api_version(ompd_word_t *version) {
  *version = OMPD_VERSION;
  return ompd_rc_ok;
}

ompd_rc_t
ompd_get_api_version_string(const char **string /* OUT: OMPD version string */
                            ) {
  static const char version_string[] =
      "LLVM OpenMP " STR(OMPD_IMPLEMENTS_OPENMP) "." STR(
          OMPD_IMPLEMENTS_OPENMP_SUBVERSION) " Debugging Library implmenting "
                                             "TR " STR(OMPD_TR_VERSION) "" STR(OMPD_TR_SUBVERSION);
  *string = version_string;
  return ompd_rc_ok;
}

/* --- 12 Display Control Variables ----------------------------------------- */

ompd_rc_t
ompd_get_display_control_vars(ompd_address_space_handle_t *handle,
                              const char *const **control_var_values) {
  static const char *const control_vars[] = {NULL};
  *control_var_values = control_vars;
  return ompd_rc_ok;
}

ompd_rc_t
ompd_release_display_control_vars(const char *const **control_var_values) {
  return ompd_rc_ok;
}

/* --- Helper functions ----------------------------------------------------- */

ompd_rc_t initTypeSizes(ompd_address_space_context_t *context) {
  static int inited = 0;
  static ompd_rc_t ret;
  if (inited)
    return ret;
  ret = callbacks->tsizeof_prim(context, &type_sizes);
  if (ret != ompd_rc_ok)
    return ret;
  if (!(type_sizes.sizeof_pointer > 0))
    return ompd_rc_error;
  ret = callbacks->tsizeof_prim(context, &TValue::type_sizes);
  if (ret != ompd_rc_ok)
    return ret;
  inited = 1;
  return ret;
}
