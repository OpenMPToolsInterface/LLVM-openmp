/*
 * OMPDContext.cpp
 *
 *  Created on: Apr 24, 2015
 *      Author: Joachim Protze
 *     Contact: protze1@llnl.gov
 */

#include "OMPDContext.h"
#include "CudaGdb.h"
#include <string>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <sstream>

using namespace ompd_gdb;
using namespace std;

static StringParser parser;
//
// Host context
OMPDHostContextPool* OMPDHostContext::cp=NULL;
OMPDHostContextPool* OMPDCudaContext::host_cp=NULL;
GdbProcessPtr OMPDContextPool::gdb=NULL;

OMPDHostContextPool::OMPDHostContextPool(GdbProcessPtr gdb)
{
  contexts.resize( 1 );
  contexts[0] = new OMPDHostContext(0);
  OMPDHostContext::cp = this;
  OMPDContextPool::gdb = gdb;
  cachedthread = nullptr;
}

OMPDContext* OMPDHostContextPool::getThreadContext(gdb_thread_id& thr_id)
{
  if ( contexts.size() < thr_id + 1 ) {
    contexts.resize( thr_id + 1, nullptr );
    contexts[thr_id] = new OMPDHostContext{thr_id};
  } 
  else if (contexts[thr_id] == nullptr) {
    contexts[thr_id] = new OMPDHostContext{thr_id};
  }
  cachedthread = contexts[thr_id]; 
  return contexts[thr_id]; 
}

OMPDContext* OMPDHostContextPool::getFirstThreadContext()
{
  if (cachedthread == nullptr) {
    auto threads = getThreadIDsFromDebugger();
    cachedthread = getThreadContext(threads[0].first);
  }
  return cachedthread;
}

OMPDContext* OMPDHostContextPool::getThreadContext(pthread_t* osthread)
{
  for(auto threads : getThreadIDsFromDebugger())
    if (threads.second == *(uint64_t *)(osthread))
      return getThreadContext(threads.first);
  return NULL;
}

ompd_thread_context_t* OMPDHostContextPool::getThreadOmpdContext(gdb_thread_id& thr_id)
{
  return (ompd_thread_context_t*)getThreadContext(thr_id);
}

ompd_thread_context_t* OMPDHostContextPool::getThreadOmpdContext(pthread_t* osthread)
{
  return (ompd_thread_context_t*)getThreadContext(osthread);
}

ompd_address_space_context_t* OMPDHostContextPool::getGlobalOmpdContext()
{
  return (ompd_address_space_context_t*)contexts[0];
}

ompd_thread_context_t* OMPDHostContextPool::getCurrentOmpdContext()
{
  OMPDContextPool::gdb->writeInput("thread");
  string gdbOut = OMPDContextPool::gdb->readOutput();
  int thread_id = parser.matchThreadID(gdbOut.c_str());
  if ((unsigned int)thread_id >= contexts.size())
    return (ompd_thread_context_t*)contexts[0];
  return (ompd_thread_context_t*)contexts[thread_id];
}

// Cuda specialization
OMPDCudaContextPool::OMPDCudaContextPool(CudaThread* cthread)
{
  OMPDCudaContext::host_cp = OMPDHostContext::cp; // Assumes Host Context always there first

  contexts.insert( pair<CudaThread*, OMPDCudaContext*> (0, new OMPDCudaContext{this, cthread}) );
}

OMPDContext* OMPDCudaContextPool::getThreadContext(CudaThread* thr_id)
{
  if (contexts.find(thr_id) == contexts.end())
    contexts.insert ( pair<CudaThread*, OMPDCudaContext*> (thr_id, new OMPDCudaContext{this, thr_id}) );
  return contexts.find(thr_id)->second;
}

ompd_thread_context_t* OMPDCudaContextPool::getThreadOmpdContext(CudaThread* cuda_thread)
{
  return (ompd_thread_context_t*)getThreadContext(cuda_thread);
}

ompd_address_space_context_t* OMPDCudaContextPool::getGlobalOmpdContext()
{
  return (ompd_address_space_context_t*)contexts.find(0)->second;
}

bool OMPDHostContext::setThisGdbContext()
{
  bool ret = false;

  stringstream command;
  command << "thread " << (this->thread);
  OMPDContextPool::gdb->writeInput(command.str().c_str());
  string gdbOut = OMPDContextPool::gdb->readOutput();
  if (gdbOut.find("not known")==0)
    ret = true;
  return ret;
}

ompd_thread_context_t* OMPDHostContext::getContextForThread(pthread_t* _osthread)
{
  return cp->getThreadOmpdContext(_osthread);
}
  

ompd_thread_context_t * OMPDHostContext::getContextForThread(gdb_thread_id& thr_id)
{
  return cp->getThreadOmpdContext(thr_id);
}

bool OMPDCudaContext::setThisGdbContext()
{
  bool ret = true;
  stringstream device_command;
  stringstream coord_command;
  device_command << "cuda device " << this->cudathread->coord.cudaDevId;
  coord_command << "cuda grid " << this->cudathread->coord.gridId
                << " block " << this->cudathread->coord.blockIdx.x
                << " thread " << this->cudathread->coord.threadIdx.x;
  OMPDContextPool::gdb->writeInput(device_command.str().c_str());
  string gdbOut = OMPDContextPool::gdb->readOutput();
  if (gdbOut.find("cannot be satisfied") != 0)
    ret = false;

  OMPDContextPool::gdb->writeInput(coord_command.str().c_str());
  gdbOut = OMPDContextPool::gdb->readOutput();
  if (gdbOut.find("cannot be satisfied") != 0)
    ret = false;

#if 0
  stringstream command;
  command 
#ifdef HACK_FOR_CUDA_GDB
    << "cuda device " << this->cudathread->coord.cudaDevId
    << " grid " << this->cudathread->coord.gridId
#else
    << "cuda kernel " << this->cudathread->coord.kernelId
#endif
    << " block " << this->cudathread->coord.blockIdx.x
    << " thread " << this->cudathread->coord.threadIdx.x;
  OMPDContextPool::gdb->writeInput(command.str().c_str());
  string gdbOut = OMPDContextPool::gdb->readOutput();
  if (gdbOut.find("not known")==0)
    ret = true;
#endif
  return ret;
}

ompd_thread_context_t * OMPDCudaContext::getContextForThread(CudaThread* cthread_id)
{
  return cp->getThreadOmpdContext(cthread_id);
}
