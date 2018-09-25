/*
 * ompd_intel.h
 *
 *  Created on: Jan 14, 2015
 *      Author: Ignacio Laguna
 *              Joachim Protze
 *     Contact: ilaguna@llnl.gov
 *              protze@llnl.gov
 */
#ifndef SRC_OMP_DEBUG_H_
#define SRC_OMP_DEBUG_H_

#ifdef __cplusplus

#include <cstdlib>

#define OMPD_DLL_VERSION 201811;

extern "C" {
#endif

#define OMPD_IMPLEMENTS_OPENMP 5
#define OMPD_IMPLEMENTS_OPENMP_SUBVERSION 0
#define OMPD_TR_VERSION 6
#define OMPD_TR_SUBVERSION 2
#define OMPD_VERSION                                                           \
  (OMPD_IMPLEMENTS_OPENMP << 24) + (OMPD_IMPLEMENTS_OPENMP_SUBVERSION << 16) + \
      (OMPD_TR_VERSION << 8) + OMPD_TR_SUBVERSION

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

#include "ompd.h"

/******************************************************************************
 * General helper functions
   */
  ompd_rc_t initTypeSizes(ompd_address_space_context_t *context);

#ifdef __cplusplus
  }


static const ompd_callbacks_t *callbacks = nullptr;


// Information shared by all threads in a kernel
// Used to map thread handles to native cuda thread ids
typedef struct _ompd_cuda_thread_kernel_info_s {
  ompd_addr_t cudaDevId;
  ompd_addr_t cudaContext;
  ompd_addr_t warpSize;
  ompd_addr_t gridId;
} ompd_cuda_thread_kernel_info_t;

typedef struct _ompd_address_space_context_s ompd_address_space_context_t;

typedef struct _ompd_process_handle_s {
  ompd_address_space_context_t *context;
} ompd_process_handle_t;

typedef struct _ompd_address_space_handle_s {
  ompd_address_space_context_t *context;
  ompd_device_t kind;
  uint64_t id;
} ompd_address_space_handle_t;

typedef struct _ompd_thread_handle_s {
  ompd_address_space_handle_t *ah;
  ompd_thread_context_t *thread_context;
  ompd_address_t th; /* target handle */
  ompd_cuda_thread_kernel_info_t *cuda_kernel_info; /* only valid for cuda */
} ompd_thread_handle_t;

typedef struct _ompd_parallel_handle_s {
  ompd_address_space_handle_t *ah;
  ompd_address_t th;  /* target handle */
  ompd_address_t lwt; /* lwt handle */
  ompd_cuda_thread_kernel_info_t *cuda_kernel_info; /* copied from the thread
                                                       used to retrieve this
                                                       parallel region handle
                                                     */
} ompd_parallel_handle_t;

typedef struct _ompd_task_handle_s {
  ompd_address_space_handle_t *ah;
  ompd_address_t th;  /* target handle */
  ompd_address_t lwt; /* lwt handle */
  ompd_cuda_thread_kernel_info_t *cuda_kernel_info; /* copied from the thread
                                                       used to retrieve this
                                                       parallel region handle
                                                     */
} ompd_task_handle_t;

#endif

// TODO (mr) this is ugly, but better then a global symbol (?)
void __ompd_init_icvs(const ompd_callbacks_t *table);
void __ompd_init_states(const ompd_callbacks_t *table);

#endif /* SRC_OMP_DEBUG_H_ */
