//TODO: (mr) move this to cmake file
#define OMPD_SUPPORT 1

#ifdef OMPD_SUPPORT

#include "omptarget-nvptx.h"
#include "state-queue.h"
#include "option.h"
#include <stdint.h>

#ifndef __OMPD_SPECIFIC_H__
#define __OMPD_SPECIFIC_H__


__device__ void ompd_init( void );
extern "C" __device__ void ompd_bp_parallel_begin ( void );
extern "C" __device__ void ompd_bp_parallel_end ( void );
extern "C" __device__ void ompd_bp_task_begin ( void );
extern "C" __device__ void ompd_bp_task_end ( void );


#define OMPD_FOREACH_ACCESS(OMPD_ACCESS) \
  OMPD_ACCESS(omptarget_nvptx_ThreadPrivateContext, topTaskDescr) \

#define OMPD_FOREACH_SIZEOF(OMPD_SIZEOF) \
  OMPD_SIZEOF(omptarget_nvptx_ThreadPrivateContext)\
  OMPD_SIZEOF(omptarget_nvptx_TaskDescr)

#endif /* OMPD_SUPPORT */
#endif
