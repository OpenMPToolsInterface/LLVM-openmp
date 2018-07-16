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
#include "ompd-private.h"
#include "TargetValue.h"
#include <assert.h>
#include <cstdio>
#include <inttypes.h>
#include <pthread.h>
#include <stdint.h>

const ompd_callbacks_t *callbacks = nullptr;
ompd_device_type_sizes_t type_sizes;
uint64_t ompd_state;

/* --- OMPD functions ------------------------------------------------------- */

/* --- 3 Initialization ----------------------------------------------------- */

ompd_rc_t ompd_initialize(ompd_word_t version, const ompd_callbacks_t *table) {
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
  *addrhandle = new ompd_address_space_handle_t;
  if (!addrhandle)
    return ompd_rc_error;
  (*addrhandle)->context = context;
  (*addrhandle)->kind = OMP_DEVICE_KIND_HOST;

  return ompd_rc_ok;
}

ompd_rc_t
ompd_get_omp_version(ompd_address_space_handle_t
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

ompd_rc_t ompd_get_omp_version_string(
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

ompd_rc_t ompd_device_initialize(
    ompd_address_space_handle_t *process_handle,
    ompd_address_space_context_t *device_context,
    int kind,
    ompd_size_t sizeof_id,
    void *id,
    ompd_address_space_handle_t **device_handle
    )
{
  if (!device_context)
    return ompd_rc_bad_input;

  // TODO:(mr) primitive type sizes can be different on devices? Think about implementing that

  ompd_rc_t ret;
  uint64_t ompd_num_cuda_devices;

  ret = TValue(process_handle->context, "ompd_num_cuda_devices").
        castBase(ompd_type_long_long).
        getValue(ompd_num_cuda_devices);
  if (ret != ompd_rc_ok)
    return ret;


  for (uint64_t i = 0; i < ompd_num_cuda_devices; i++) {
    uint64_t cuda_ctx;

    // TODO: (mr) think of a better way to cast contexts
    ret = TValue(process_handle->context, "ompd_CudaContextArray").
          cast("ompd_cuda_context_ptr_t",1).
          getArrayElement(i).
          castBase(ompd_type_long_long).
          getValue(cuda_ctx);

    if ( ret != ompd_rc_ok )
      continue;

    if (cuda_ctx == (*((uint64_t *)id))) {
      *device_handle = new ompd_address_space_handle_t;
      if (!device_handle)
        return ompd_rc_error;
      (*device_handle)->context = device_context;
      (*device_handle)->kind = OMP_DEVICE_KIND_CUDA;
      (*device_handle)->id = (uint64_t)id;
      return ompd_rc_ok;
    }
  }

  return ompd_rc_unavailable;
}


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
  ret = callbacks->memory_alloc(sizeof(ompd_thread_handle_t),
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
  ompd_rc_t ret = callbacks->memory_free((void *)(thread_handle));
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
  ompd_thread_context_t *thread_context = thread_handle->thread_context;
  if (!context || !thread_context)
    return ompd_rc_stale_handle;

  assert(callbacks && "Callback table not initialized!");

  ompd_rc_t ret;

  if (thread_handle->ah->kind == OMP_DEVICE_KIND_CUDA) {
    ompd_address_t taddr;
    TValue ph = TValue(context, thread_context,
                       "omptarget_nvptx_threadPrivateContext",
                       OMPD_SEGMENT_CUDA_PTX_SHARED);
    ret = ph.getAddress(&taddr);
    if (ret != ompd_rc_ok)
      return ret;

      ret = callbacks->memory_alloc(sizeof(ompd_parallel_handle_t),
                                   (void **)(parallel_handle));
    if (ret != ompd_rc_ok)
      return ret;

    (*parallel_handle)->ah = thread_handle->ah;
    (*parallel_handle)->th = taddr;
  } else {
    ompd_address_t taddr, lwt;

    TValue teamdata = TValue(context, thread_handle->th) /*__kmp_threads[t]->th*/
                          .cast("kmp_base_info_t")
                          .access("th_team") /*__kmp_threads[t]->th.th_team*/
                          .cast("kmp_team_p", 1)
                          .access("t"); /*__kmp_threads[t]->th.th_team->t*/

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

    ret = callbacks->memory_alloc(sizeof(ompd_parallel_handle_t),
                                   (void **)(parallel_handle));
    if (ret != ompd_rc_ok)
      return ret;

    (*parallel_handle)->ah = thread_handle->ah;
    (*parallel_handle)->th = taddr;
    (*parallel_handle)->lwt = lwt;
  }
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

  ret = callbacks->memory_alloc(sizeof(ompd_parallel_handle_t),
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

  ret = callbacks->memory_alloc(sizeof(ompd_parallel_handle_t),
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
  ompd_rc_t ret = callbacks->memory_free((void *)(parallel_handle));
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

/* --- 4.3 Task Handles ----------------------------------------------------- */

/* task_handle is of type (kmp_taskdata_t) */

ompd_rc_t ompd_get_current_task_handle(
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

  ret = callbacks->memory_alloc(sizeof(ompd_task_handle_t),
                                 (void **)(task_handle));
  if (ret != ompd_rc_ok)
    return ret;

  (*task_handle)->th = taddr;
  (*task_handle)->lwt = lwt;
  (*task_handle)->ah = thread_handle->ah;
  return ompd_rc_ok;
}

ompd_rc_t ompd_get_generating_task_handle(
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

  ret = callbacks->memory_alloc(sizeof(ompd_task_handle_t),
                                 (void **)(parent_task_handle));
  if (ret != ompd_rc_ok)
    return ret;

  (*parent_task_handle)->th = taddr;
  (*parent_task_handle)->lwt = lwt;
  (*parent_task_handle)->ah = task_handle->ah;
  return ret;
}

ompd_rc_t ompd_get_scheduling_task_handle(
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
  ret = callbacks->memory_alloc(sizeof(ompd_task_handle_t),
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
  ret = callbacks->memory_alloc(sizeof(ompd_task_handle_t),
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
  ompd_rc_t ret = callbacks->memory_free((void *)(task_handle));
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
#if 0
    if (!ompd_state)
      return ompd_rc_needs_state_tracking;
#endif

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


/* --- 7 Thread Inquiry ----------------------------------------------------- */

/* --- 7.1 Operating System Thread Inquiry ---------------------------------- */

ompd_rc_t
ompd_get_thread_handle(ompd_address_space_handle_t
                           *addr_handle, /* IN: handle for the address space */
                       ompd_thread_id_t kind,
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

  if (kind == OMPD_THREAD_ID_CUDALOGICAL) {
    ompd_cudathread_coord_t *p = (ompd_cudathread_coord_t *)thread_id;

    // omptarget_nvptx_threadPrivateContext->topTaskDescr[p->threadIdx.x]->items.threadId
    TValue th = TValue(context, tcontext,
                       "omptarget_nvptx_threadPrivateContext",
                       OMPD_SEGMENT_CUDA_PTX_SHARED)
                .cast("omptarget_nvptx_ThreadPrivateContext", 1,
                      OMPD_SEGMENT_CUDA_PTX_SHARED)
                .access("topTaskDescr")
                .cast("omptarget_nvptx_TaskDescr", 1,
                      OMPD_SEGMENT_CUDA_PTX_GLOBAL)
                .getArrayElement(p->threadIdx.x);

    ompd_address_t taddr;
    ret = th.getAddress(&taddr);

    if (ret != ompd_rc_ok)
      return ret;

    ret = th.access("items__threadId")
            .castBase(ompd_type_short)
            .getValue(tId);

    if (ret != ompd_rc_ok)
      return ret;

    if (tId != p->threadIdx.x)
      return ompd_rc_stale_handle;

    ret = callbacks->memory_alloc(sizeof(ompd_thread_handle_t),
                                   (void **)(thread_handle));
    if (ret != ompd_rc_ok)
      return ret;

    (*thread_handle)->ah = addr_handle;
    (*thread_handle)->th = taddr;
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
    ret = callbacks->memory_alloc(sizeof(ompd_thread_handle_t),
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
  (*thread_handle)->thread_context = tcontext;
  return ret;
}

ompd_rc_t ompd_get_thread_id(
    ompd_thread_handle_t *thread_handle, /* IN: OpenMP thread handle*/
    ompd_thread_id_t kind, ompd_size_t sizeof_thread_id, void *thread_id) {
  if (kind != OMPD_THREAD_ID_PTHREAD && kind != OMPD_THREAD_ID_CUDALOGICAL)
    return ompd_rc_bad_input;
  if (!thread_handle)
    return ompd_rc_stale_handle;
  if (!thread_handle->ah)
    return ompd_rc_stale_handle;
  ompd_address_space_context_t *context = thread_handle->ah->context;
  if (!context)
    return ompd_rc_stale_handle;
  ompd_rc_t ret;

  if (kind != OMPD_THREAD_ID_CUDALOGICAL) {
    ret = ompd_rc_unsupported;
  } else {
    ompd_size_t size;
    ret = tf.getType(context, "kmp_thread_t").getSize(&size);
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
  }
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
#if 0
  if (!ompd_state)
    return ompd_rc_needs_state_tracking;
#endif

  ompd_rc_t ret;
  assert(callbacks && "Callback table not initialized!");

  if (thread_handle->ah->kind == OMP_DEVICE_KIND_HOST) {
    if (wait_id)
      *wait_id = 0; //TODO: (mr) implement wait_ids in nvptx device rtl
    ret  = TValue(context, thread_handle->th)
            .cast("omptarget_nvptx_TaskDescr", 0, OMPD_SEGMENT_CUDA_PTX_SHARED)
            .access("ompd_thread_info")
            .cast("ompd_nvptx_thread_info_t", 0, OMPD_SEGMENT_CUDA_PTX_GLOBAL)
            .access("state")
            .castBase(ompd_type_long_long)
            .getValue(*state);
  } else {
    TValue ompt_thread_info =
        TValue(context, thread_handle->th) /*__kmp_threads[t]->th*/
            .cast("kmp_base_info_t")
            .access("ompt_thread_info") /*__kmp_threads[t]->th.ompt_thread_info*/
            .cast("ompt_thread_info_t");
    if (ompt_thread_info.gotError())
      return ompt_thread_info.getError();
    ret = ompt_thread_info
            .access("state") /*__kmp_threads[t]->th.ompt_thread_info.state*/
            .castBase()
            .getValue(*state);
    if (ret != ompd_rc_ok)
      return ret;
    ret = ompt_thread_info
              .access("wait_id") /*__kmp_threads[t]->th.ompt_thread_info.state*/
              .castBase()
              .getValue(*wait_id);
  }
  return ret;
}

/* --- 8 Task Inquiry ------------------------------------------------------- */

/* --- 8.1 Task Settings ---------------------------------------------------- */

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
#if 0
  if (!ompd_state)
    return ompd_rc_needs_state_tracking;
#endif

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
#if 0
  if (!ompd_state)
    return ompd_rc_needs_state_tracking;
#endif

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
  ret = callbacks->sizeof_types(context, &type_sizes);
  if (ret != ompd_rc_ok)
    return ret;
  if (!(type_sizes.sizeof_pointer > 0))
    return ompd_rc_error;
  ret = callbacks->sizeof_types(context, &TValue::type_sizes);
  if (ret != ompd_rc_ok)
    return ret;
  inited = 1;
  return ret;
}
