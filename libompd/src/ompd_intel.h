/*
 * ompd_intel.h
 *
 *  Created on: Jan 14, 2015
 *      Author: Ignacio Laguna
 *              Joachim Protze
 *     Contact: ilaguna@llnl.gov
 *              protze@llnl.gov
 */
#ifndef SRC_OMPD_INTEL_H_
#define SRC_OMPD_INTEL_H_

#ifdef  __cplusplus

#include <cstdlib>
//#include <new>
//void* operator new(std::size_t size) /*throw (std::bad_alloc)*/;
//void* operator new[](std::size_t size) /*throw (std::bad_alloc)*/;
//void operator delete(void* addr) throw ();
//void operator delete[](void* addr) throw ();

extern "C" {
#endif

  
#define OMPD_IMPLEMENTS_OPENMP 3
#define OMPD_IMPLEMENTS_OPENMP_SUBVERSION 1
#define OMPD_TR_VERSION 6
#define OMPD_TR_SUBVERSION 'j'
#define OMPD_VERSION (OMPD_IMPLEMENTS_OPENMP << 24) + (OMPD_IMPLEMENTS_OPENMP_SUBVERSION << 16) + (OMPD_TR_VERSION << 8) + OMPD_TR_SUBVERSION
  
#include "ompd.h"
#include "ompd_typedefs.h"

/******************************************************************************
 * General helper functions
 */
ompd_rc_t initTypeSizes(ompd_address_space_context_t *context);
//uint32_t getThreadLevel(ompd_context_t *context, int t);
/*int getNumberOfOMPThreads(ompd_context_t *context);
uint64_t getSystemThreadID(ompd_context_t *context, int t);
int64_t getOmpThreadID(ompd_context_t *context);*/


#ifdef  __cplusplus
}
#endif


typedef struct _ompd_address_space_context_s ompd_address_space_context_t;

typedef struct _ompd_process_handle_s
{
  ompd_address_space_context_t* context;
} ompd_process_handle_t;

typedef struct _ompd_address_space_handle_s 
{
  ompd_address_space_context_t* context;
  ompd_device_kind_t kind;
  ompd_device_identifier_t id;
} ompd_address_space_handle_t;

typedef struct _ompd_device_handle_s {
  ompd_address_space_handle_t* ah;
  ompd_address_t th; /* target handle */
} ompd_device_handle_t;

typedef struct _ompd_thread_handle_s
{
  ompd_address_space_handle_t* ah;
  ompd_address_t th; /* target handle */
} ompd_thread_handle_t;

typedef struct _ompd_parallel_handle_s 
{
  ompd_address_space_handle_t* ah;
  ompd_address_t th; /* target handle */
  ompd_address_t lwt; /* lwt handle */
} ompd_parallel_handle_t;

typedef struct _ompd_task_handle_s 
{
  ompd_address_space_handle_t* ah;
  ompd_address_t th; /* target handle */
  ompd_address_t lwt; /* lwt handle */
} ompd_task_handle_t;


#endif /* SRC_OMPD_INTEL_H_ */
