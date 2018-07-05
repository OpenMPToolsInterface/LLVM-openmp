#ifdef OMPD_SUPPORT
#include "ompd-specific.h"
#include "omptarget-nvptx.h"
/**
   * Declaration of symbols to hold struct size and member offset information
    */

__device__ __shared__ static int ompd_target_initialized;

#define ompd_target_declare_access(t,m) __device__ __shared__ uint64_t ompd_access__##t##__##m; 
OMPD_FOREACH_ACCESS(ompd_target_declare_access)
#undef ompd_target_declare_access

#define ompd_target_declare_sizeof_member(t,m) __device__ __shared__ uint64_t ompd_sizeof__##t##__##m; 
    OMPD_FOREACH_ACCESS(ompd_target_declare_sizeof_member)
#undef ompd_target_declare_sizeof_member

#define ompd_target_declare_sizeof(t) __device__ __shared__ uint64_t ompd_sizeof__##t; 
    OMPD_FOREACH_SIZEOF(ompd_target_declare_sizeof)
#undef ompd_target_declare_sizeof

__device__ __shared__
  uint64_t ompd_access__omptarget_nvptx_TaskDescr__items__threadId;

__device__ __shared__
  uint64_t ompd_sizeof__omptarget_nvptx_TaskDescr__items__threadId;


__device__ void ompd_init ( void )
{
  getMyTopTaskDescriptor()->ompd_thread_info.state = omp_state_undefined;
  getMyTopTaskDescriptor()->ompd_thread_info.blockIdx_x = blockIdx.x;
  getMyTopTaskDescriptor()->ompd_thread_info.threadIdx_x = threadIdx.x;

  if (ompd_target_initialized)
    return;

#define ompd_target_init_access(t,m) ompd_access__##t##__##m = (uint64_t)&(((t*)0)->m); 
  OMPD_FOREACH_ACCESS(ompd_target_init_access)
#undef ompd_target_init_access

  ompd_access__omptarget_nvptx_TaskDescr__items__threadId = 
          (uint64_t)&(((omptarget_nvptx_TaskDescr*)0)->items.threadId);

#define ompd_target_init_sizeof_member(t,m) ompd_sizeof__##t##__##m = sizeof(((t*)0)->m); 
  OMPD_FOREACH_ACCESS(ompd_target_init_sizeof_member)
#undef ompd_target_init_sizeof_member

  ompd_sizeof__omptarget_nvptx_TaskDescr__items__threadId =
    (uint64_t)sizeof(((omptarget_nvptx_TaskDescr*)0)->items.threadId);

#define ompd_target_init_sizeof(t) ompd_sizeof__##t = sizeof(t); 
  OMPD_FOREACH_SIZEOF(ompd_target_init_sizeof)
#undef ompd_target_init_sizeof

  ompd_target_initialized = 1;
}

__device__ void ompd_set_device_thread_state(omp_state_t state) {
  getMyTopTaskDescriptor()->ompd_thread_info.state = state;
}

__device__ void ompd_bp_parallel_begin (){ asm (""); }
__device__ void ompd_bp_parallel_end (){ asm (""); }
__device__ void ompd_bp_task_begin (){ asm (""); }
__device__ void ompd_bp_task_end (){ asm (""); }
#endif /* OMPD_SUPPORT */
