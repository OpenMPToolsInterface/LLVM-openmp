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

#ifdef  __cplusplus

#include <cstdlib>
#include <new>
//void* operator new(std::size_t size) /*throw (std::bad_alloc)*/;
//void* operator new[](std::size_t size) /*throw (std::bad_alloc)*/;
//void operator delete(void* addr) throw ();
//void operator delete[](void* addr) throw ();

  #define OMPD_DLL_VERSION 201811;

extern "C" {
#endif

  
#define OMPD_IMPLEMENTS_OPENMP 3
#define OMPD_IMPLEMENTS_OPENMP_SUBVERSION 1
#define OMPD_TR_VERSION 6
#define OMPD_TR_SUBVERSION 'j'
#define OMPD_VERSION (OMPD_IMPLEMENTS_OPENMP << 24) + (OMPD_IMPLEMENTS_OPENMP_SUBVERSION << 16) + (OMPD_TR_VERSION << 8) + OMPD_TR_SUBVERSION

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)
  
#include "ompd.h"

/******************************************************************************
 * General helper functions
 */
ompd_rc_t initTypeSizes(ompd_address_space_context_t *context);


#ifdef  __cplusplus
}

static const ompd_callbacks_t *callbacks = NULL;

class ompdAllocatable {
public:
  static void* operator new(std::size_t sz)
  {
    void* res;
    ompd_rc_t ret = callbacks->dmemory_alloc(sz, &res);
    if (ret==ompd_rc_ok)
      return res;
    throw std::bad_alloc();
  }
  static void* operator new[](std::size_t sz)
  {
    void* res;
    ompd_rc_t ret = callbacks->dmemory_alloc(sz, &res);
    if (ret==ompd_rc_ok)
      return res;
    throw std::bad_alloc();
  }
  void operator delete(void* addr) throw ()
  {
    ompd_rc_t ret = callbacks->dmemory_free(addr);
    if (ret!=ompd_rc_ok)
      throw std::bad_alloc();
  }
  void operator delete[](void* addr) throw ()
  {
    ompd_rc_t ret = callbacks->dmemory_free(addr);
    if (ret!=ompd_rc_ok)
      throw std::bad_alloc();
  }
};


typedef struct _ompd_address_space_context_s ompd_address_space_context_t;

typedef struct _ompd_process_handle_s : public ompdAllocatable
{
  ompd_address_space_context_t* context;
} ompd_process_handle_t;

typedef struct _ompd_address_space_handle_s : public ompdAllocatable
{
  ompd_address_space_context_t* context;
  ompd_device_kind_t kind;
  ompd_device_identifier_t id;
} ompd_address_space_handle_t;

typedef struct _ompd_device_handle_s : public ompdAllocatable 
{
  ompd_address_space_handle_t* ah;
  ompd_address_t th; /* target handle */
} ompd_device_handle_t;

typedef struct _ompd_thread_handle_s : public ompdAllocatable
{
  ompd_address_space_handle_t* ah;
  ompd_address_t th; /* target handle */
} ompd_thread_handle_t;

typedef struct _ompd_parallel_handle_s  : public ompdAllocatable
{
  ompd_address_space_handle_t* ah;
  ompd_address_t th; /* target handle */
  ompd_address_t lwt; /* lwt handle */
} ompd_parallel_handle_t;

typedef struct _ompd_task_handle_s  : public ompdAllocatable
{
  ompd_address_space_handle_t* ah;
  ompd_address_t th; /* target handle */
  ompd_address_t lwt; /* lwt handle */
} ompd_task_handle_t;

#endif

#endif /* SRC_OMP_DEBUG_H_ */
