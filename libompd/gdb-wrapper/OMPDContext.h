/*
 * OMPDContext.h
 *
 *  Created on: Apr 24, 2015
 *      Author: Joachim Protze
 *     Contact: protze1@llnl.gov
 */
#ifndef GDB_OMPDCONTEXT_H_
#define GDB_OMPDCONTEXT_H_

/*******************************************************************************
 * This class implements the ompd context handle for GDB.
 * The context provides information about the process, the selected thread
 *   and other information that reflects the current state of the debuggers
 *   context.
 */

#include "omp-tools.h"
//#include "ompd_test.h"
#include "GdbProcess.h"
#include "Callbacks.h"
#include "CudaGdb.h"

#include <cstring>
#include <cassert>
#include <memory>
#include <vector>
#include <set>
#include <pthread.h>

typedef uint32_t gdb_thread_id;

namespace ompd_gdb {

class OMPDContext;
class OMPDHostContext;
class OMPDCudaContext;

class OMPDContextPool
{
public:
  static GdbProcessPtr gdb;
  virtual ompd_address_space_context_t* getGlobalOmpdContext() = 0;
};

class OMPDContext
{
friend class OMPDHostContextPool;
friend class OMPDCudaContextPool;
public:
  virtual bool setThisGdbContext() = 0;

  virtual ompd_thread_context_t* getContextForThread(gdb_thread_id& thr_handle) { return nullptr; }
  virtual ompd_thread_context_t* getContextForThread(CudaThread* cuda_thr) { return nullptr; }
  virtual ompd_thread_context_t* getContextForThread(pthread_t* osthread) { return nullptr; }
};

class OMPDHostContextPool: public OMPDContextPool
{
private:
  std::vector<OMPDHostContext*> contexts;
  OMPDContext* cachedthread; // Arbitrarily picked first thread
  
public:
  OMPDContext* getThreadContext(gdb_thread_id& thr_handle);
  OMPDContext* getThreadContext(pthread_t* osthread);
  ompd_thread_context_t* getThreadOmpdContext(gdb_thread_id& thr_handle);
  ompd_thread_context_t* getThreadOmpdContext(pthread_t* osthread);
  ompd_address_space_context_t* getGlobalOmpdContext();
  ompd_thread_context_t* getCurrentOmpdContext();
  OMPDContext* getFirstThreadContext(void);
  OMPDHostContextPool(GdbProcessPtr gdb);
};

class OMPDHostContext: public OMPDContext
{
friend class OMPDHostContextPool;

private:
  gdb_thread_id thread;
  
public:
  static OMPDHostContextPool* cp;

  OMPDHostContext(gdb_thread_id t): thread(t) {}

  bool setThisGdbContext();

/**
 * Get a context for given os thread handle
 */
  ompd_thread_context_t* getContextForThread(gdb_thread_id& thr_handle);
  ompd_thread_context_t* getContextForThread(pthread_t* osthread);
};

class OMPDCudaContext;

// We allocate a separate pool per Cuda Device (CUDA Context)
class OMPDCudaContextPool: public OMPDContextPool
{
private:
  std::map<CudaThread*, OMPDCudaContext*> contexts;
  
public:
  ompd_address_space_handle_t *ompd_device_handle;
  OMPDContext* getThreadContext(CudaThread* cuda_thread);
  ompd_thread_context_t* getThreadOmpdContext(CudaThread* cuda_thread);
  ompd_address_space_context_t* getGlobalOmpdContext();
  OMPDCudaContextPool(CudaThread* cuda_thread);
};

class OMPDCudaContext: public OMPDContext
{
friend class OMPDCudaContextPool;
private:
  OMPDCudaContext(OMPDCudaContextPool* _cp, CudaThread* cuda_thread): 
    cp(_cp), cudathread(cuda_thread) {}

public:
  static OMPDHostContextPool* host_cp;   // Only one Host Context Pool
  OMPDCudaContextPool* cp;               // One per cuda device
  CudaThread *cudathread;

  bool setThisGdbContext(); /* Make this context active in gdb */
  ompd_thread_context_t* getContextForThread(CudaThread* cuda_thr);
};
}

#endif /* GDB_OMPDCONTEXT_H_ */
