/*
 * OMPDCommand.cpp
 *
 *  Created on: Dec 28, 2014
 *      Author: Ignacio Laguna
 *     Contact: ilaguna@llnl.gov
 */
#include <config.h>
#include "OMPDCommand.h"
#include "OMPDContext.h"
#include "Callbacks.h"
#include "OutputString.h"
#include "Debug.h"
#include "ompd.h"
#include "ompd_test.h"
#include "CudaGdb.h"

#include <cstdlib>
#include <dlfcn.h>
#include <iostream>
#include <sstream>

using namespace ompd_gdb;
using namespace std;

const char * ompd_state_names[256];
extern OMPDHostContextPool * host_contextPool;

/* --- OMPDCommandFactory --------------------------------------------------- */

OMPDCommandFactory::OMPDCommandFactory()
{
  functions = OMPDFunctionsPtr(new OMPDFunctions);

#define ompd_state_macro(state, code) ompd_state_names[code] = #state;
    FOREACH_OMPD_STATE(ompd_state_macro)
#undef ompd_state_macro

  // Load OMPD DLL and get a handle
#ifdef ODB_LINUX
  functions->ompdLibHandle = dlopen("libompd_intel.so", RTLD_LAZY);
#elif defined(ODB_MACOS)
  functions->ompdLibHandle = dlopen("libompd_intel.dylib", RTLD_LAZY);
#else
#error Unsupported platform!
#endif
  if (!functions->ompdLibHandle)
  {
    stringstream msg;
    msg << "ERROR: could not open OMPD library.\n" << dlerror() << "\n";
    out << msg.str().c_str();
    functions->ompdLibHandle = nullptr;
    exit(1);
    return;
  }
  else
  {
    cerr << "OMPD library loaded\n";
  }
  dlerror(); // Clear any existing error

  /* Look up OMPD API function in the library
   * The Macro generates function pointer lookups for all implemented API function listed in OMPDCommand.h:41
   */
#define OMPD_FIND_API_FUNCTION(FN) functions-> FN        = \
    (FN##_fn_t) findFunctionInLibrary(#FN);\

FOREACH_OMPD_API_FN(OMPD_FIND_API_FUNCTION)
#undef OMPD_FIND_API_FUNCTION

  //functions->test_CB_tsizeof_prim   =
  //   (void (*)()) findFunctionInLibrary("test_CB_tsizeof_prim");
  //functions->test_CB_dmemory_alloc  =
  //   (void (*)()) findFunctionInLibrary("test_CB_dmemory_alloc");

  // Initialize OMPD library
  ompd_callbacks_t *table = getCallbacksTable();
  assert(table && "Invalid callbacks table");
  ompd_rc_t ret = functions->ompd_initialize(table);
  if (ret != ompd_rc_ok)
  {
    out << "ERROR: could not initialize OMPD\n";
  }

  ret = functions->ompd_process_initialize(host_contextPool->getGlobalOmpdContext(),
                                          /*&prochandle, */&addrhandle);
  if (ret != ompd_rc_ok)
  {
    out << "ERROR: could not initialize target process\n";
  }
}

OMPDCommandFactory::~OMPDCommandFactory()
{
  ompd_rc_t ret;
//   ret = functions->ompd_process_finalize(prochandle);
//   if (ret != ompd_rc_ok)
//   {
//     out << "ERROR: could not finalize target process\n";
//   }
  ret = functions->ompd_release_address_space_handle(addrhandle);
  if (ret != ompd_rc_ok)
  {
    out << "ERROR: could not finalize target address space\n";
  }
}

void * OMPDCommandFactory::findFunctionInLibrary(const char *fun) const
{
  if (!functions->ompdLibHandle)
    return nullptr;

  void *ret = dlsym(functions->ompdLibHandle, fun);
  char *err = dlerror();
  if (err)
  {
    stringstream msg;
    msg << "ERROR: could not find ompd_initialize: " << err << "\n";
    out << msg.str().c_str();
    return nullptr;
  }
  return ret;
}

OMPDCommand* OMPDCommandFactory::create(const char *str, const vector<string>& extraArgs) const
{
  if (strcmp(str, "test") == 0)
    return new OMPDTestCallbacks(functions, addrhandle, extraArgs);
  else if (strcmp(str, "threads") == 0)
    return new OMPDThreads(functions, addrhandle, extraArgs);
  else if (strcmp(str, "levels") == 0)
    return new OMPDLevels(functions, addrhandle, extraArgs);
  else if (strcmp(str, "callback") == 0)
    return new OMPDCallback(functions, addrhandle, extraArgs);
  else if (strcmp(str, "api") == 0)
    return new OMPDApi(functions, addrhandle, extraArgs);
  else if (strcmp(str, "testapi") == 0)
    return new OMPDTest(functions, addrhandle, extraArgs);

  return new OMPDNull;
}

/* --- OMPDNull ------------------------------------------------------------- */

void OMPDNull::staticExecute()
{
  hout << "Null odb command\n";
}

void OMPDNull::execute() const
{
  staticExecute();
}

const char* OMPDNull::toString() const
{
  return "NULL";
}

/* --- OMPDTestCallbacks ---------------------------------------------------- */

void OMPDTestCallbacks::execute() const
{
  // If any function is null, execute a null command
  if (!functions->test_CB_tsizeof_prim ||
      !functions->test_CB_dmemory_alloc)
  {
    OMPDNull::staticExecute();
    return;
  }

  // Call all test functions in OMPD
  functions->test_CB_tsizeof_prim();
  functions->test_CB_dmemory_alloc();
}

const char* OMPDTestCallbacks::toString() const
{
  return "odb test";
}

/* --- OMPDThreads ---------------------------------------------------------- */

void OMPDThreads::execute() const
{
  printf("\nHOST THREADS\n");
  printf("Debugger_handle    Thread_handle     System_thread\n");
  printf("--------------------------------------------------\n");

  auto thread_ids = getThreadIDsFromDebugger();
  for(auto i: thread_ids) {
    ompd_thread_handle_t* thread_handle;
    ompd_rc_t ret = functions->ompd_get_thread_handle(
        addrhandle, ompd_osthread_pthread, sizeof(i.second),
        &(i.second), &thread_handle);
    if (ret == ompd_rc_ok)
    {
      ompd_state_t state;
      ompd_wait_id_t wait_id;
      ret = functions->ompd_get_state(thread_handle, &state, &wait_id);
      printf("  %-12u     %p     0x%lx\t%i\t%lx\n", 
          (unsigned int)i.first, thread_handle, i.second, state, wait_id);
    }
    else
    {
      printf("  %-12u     %-12s     %-12s\n", (unsigned int)i.first, "-", "-");
    }
  }

  CudaGdb cuda;
  int omp_cuda_threads = 0;
  vector<OMPDCudaContextPool*> cuda_ContextPools;
  map<uint64_t, bool> device_initialized;
  map<ompd_taddr_t, ompd_address_space_handle_t*> address_spaces;

  for(auto i: cuda.threads) {
    if (!device_initialized[i.coord.cudaContext]) {
      OMPDCudaContextPool* cpool;
      cpool = new OMPDCudaContextPool(&i);
      ompd_rc_t result;

      device_initialized[i.coord.cudaContext] = true;
      result = functions->ompd_device_initialize(
           cpool->getGlobalOmpdContext(),
           i.coord.cudaContext,
           ompd_device_kind_cuda, 
          &cpool->ompd_device_handle);

        if (result != ompd_rc_ok)
          continue;

        address_spaces[i.coord.cudaContext] = cpool->ompd_device_handle;
    }

    ompd_thread_handle_t* thread_handle;
    ompd_rc_t ret = functions->ompd_get_thread_handle(
                                    address_spaces[i.coord.cudaContext],
                                    ompd_osthread_cudalogical,
                                    sizeof(i.coord), &i.coord,
                                    &thread_handle);

    if (ret == ompd_rc_ok)
      omp_cuda_threads++;
  }

  if (cuda.threads.size() != 0) {
    cout << cuda.threads.size() << " CUDA Threads Found, "
      << omp_cuda_threads << " OMP Threads\n";
  }
}

const char* OMPDThreads::toString() const
{
  return "odb threads";
}


/* --- OMPDLevels ----------------------------------------------------------- */

void OMPDLevels::execute() const
{
/*  ompd_size_t num_os_threads;
  ompd_rc_t ret = CB_num_os_threads(contextPool->getGlobalOmpdContext(), &num_os_threads);
  assert(ret==ompd_rc_ok && "Error calling OMPD!");
  ompd_osthread_t* osThreads =  (ompd_osthread_t*)
      malloc(sizeof(ompd_osthread_t)*num_os_threads);
  ret = CB_get_os_threads (contextPool->getGlobalOmpdContext(),  &num_os_threads, &osThreads);
  assert(ret==ompd_rc_ok && "Error calling OMPD!");

  printf("\n");
  printf("Thread_handle     Nesting_level\n");
  printf("-------------------------------\n");
  for (size_t i=0; i < num_os_threads; ++i)
  {
    ompd_thread_handle_t thread_handle;
    ret = functions->ompd_get_thread_handle(
            contextPool->getGlobalOmpdContext(), &(osThreads[i]), &thread_handle);
    if (ret == ompd_rc_ok)
    {
      ompd_tword_t level=0;
      ret = functions->ompd_nesting_level(
          contextPool->getGlobalOmpdContext(), &thread_handle, &level);
      printf("%-12u      %ld\n", (unsigned int)thread_handle, level);
    }
  }*/
}

const char* OMPDLevels::toString() const
{
  return "odb levels";
}


/* --- OMPDCallback ----------------------------------------------------------- */

ompd_target_prim_types_t get_prim_type_from_string(const string& str)
{
  const char * names[ompd_type_max] = { 
    "CHAR", 
    "SHORT",
    "INT",
    "LONG",
    "LONG_LONG",
    "POINTER"
    };
  for (int i = 0; 0<ompd_type_max; i++)
    if (str == names[i])
      return (ompd_target_prim_types_t) i;
}

void OMPDCallback::execute() const
{ 
  ompd_rc_t ret;

  if (extraArgs.empty() || extraArgs[0] == "help")
  {
    hout << "callbacks available: read_tmemory, ttype, ttype_sizeof, ttype_offset, tsymbol_addr" << endl
         << "Use \"odb callback <callback_name>\" to get more help on the usage" << endl;
    return;
  }     

/*ompd_rc_t CB_read_tmemory (
    ompd_context_t *context,
    ompd_taddr_t addr,
    ompd_tword_t bufsize,
    void *buffer
    );*/
  if (extraArgs[0] == "read_tmemory")
  {
    if(extraArgs.size()<4)
    {
      hout << "Usage: odb callback read_tmemory <addr> <length-in-bytes> " << endl;
      return;
    }
    long long temp=0;
    ompd_taddr_t addr = (ompd_taddr_t)strtoll(extraArgs[1].c_str(), NULL, 0);
    int cnt = atoi(extraArgs[2].c_str());
    ret = CB_read_tmemory(
            host_contextPool->getGlobalOmpdContext(), NULL, {0,addr}, cnt, &temp);
    if (ret != ompd_rc_ok)
      return;
    sout << "0x" << hex << temp << endl;
  }

/*ompd_rc_t CB_tsymbol_addr (
    ompd_context_t *context,
    const char *symbol_name,
    ompd_taddr_t *symbol_addr);*/

  if (extraArgs[0] == "tsymbol_addr")
  {
    if(extraArgs.size()<2)
    {
      hout << "Usage: odb callback tsymbol_addr <symbol_name>" << endl;
      return;
    }
    ompd_address_t temp={0,0};
    ret = CB_tsymbol_addr(
            host_contextPool->getGlobalOmpdContext(), NULL, extraArgs[1].c_str(), &temp);
    if (ret != ompd_rc_ok)
      return;
    sout << "0x" << hex << temp.address << endl;
  }

}

const char* OMPDCallback ::toString() const
{
  return "odb callback";
}

void OMPDApi::execute() const
{ 
  ompd_rc_t ret;

  if (extraArgs.empty() || extraArgs[0] == "help")
  {
    hout << "API functions available: read_tmemory, ttype, ttype_sizeof, ttype_offset, tsymbol_addr" << endl
         << "Use \"odb api <function_name>\" to get more help on the usage" << endl;
    return;
  }     

//ompd_rc_t ompd_get_threads (
//    ompd_context_t *context,    /* IN: debugger handle for the target */
//    ompd_thread_handle_t **thread_handle_array, /* OUT: array of handles */
//    ompd_size_t *num_handles    /* OUT: number of handles in the array */
//    );

  if (extraArgs[0] == "get_threads")
  {
    if(extraArgs.size()>1)
    {
      hout << "Usage: odb api get_threads" << endl;
      return;
    }
    ompd_thread_handle_t ** thread_handle_array;
    int num_handles;
    
  
    ret = functions->ompd_get_threads (
          addrhandle, &thread_handle_array, &num_handles);
    if (ret != ompd_rc_ok)
      return;
    sout << num_handles << " OpenMP threads:" << endl;
    for (int i=0; i<num_handles; i++){
      sout << "0x" << hex << thread_handle_array[i] << ", ";
    }    
    sout << endl << "";
  }

}

const char* OMPDApi ::toString() const
{
  return "odb api";
}

vector<ompd_thread_handle_t*> odbGetThreadHandles(ompd_address_space_handle_t* addrhandle, OMPDFunctionsPtr functions)
{
  ompd_rc_t ret;
  auto thread_ids = getThreadIDsFromDebugger();
  vector<ompd_thread_handle_t*> thread_handles;
  for(auto i: thread_ids)
  {
    ompd_thread_handle_t* thread_handle;
    ret = functions->ompd_get_thread_handle(
        addrhandle, ompd_osthread_pthread, sizeof(i.second) ,&(i.second), &thread_handle);
    if (ret!=ompd_rc_ok)
      continue;
    thread_handles.push_back(thread_handle);
  }
  return thread_handles;
}

vector<ompd_parallel_handle_t*> odbGetParallelRegions(OMPDFunctionsPtr functions, ompd_thread_handle_t* &th)
{
  ompd_rc_t ret;
  ompd_parallel_handle_t * parallel_handle;
  vector<ompd_parallel_handle_t*> parallel_handles;
  ret = functions->ompd_get_top_parallel_region(
          th, &parallel_handle);  
  while(ret == ompd_rc_ok)
  {
    parallel_handles.push_back(parallel_handle);
    ret = functions->ompd_get_enclosing_parallel_handle(
          parallel_handle, &parallel_handle);  
  }
  return parallel_handles;
}

bool odbCheckParallelIDs(OMPDFunctionsPtr functions, vector<ompd_parallel_handle_t*> phs)
{
  bool res=true;
//  ompd_rc_t ret;
  int i=0;
  uint64_t ompt_res, ompd_res;
//   ((OMPDContext*)context)->setThisGdbContext();
  for (auto ph : phs)
  {
    stringstream ss;
    ss << "p ompt_get_parallel_id(" << i << ")";
    ompt_res = evalGdbExpression(ss.str());
    /*ret = */functions->ompd_get_parallel_id(ph, &ompd_res);
    sout << "  parallelid ompt: " << ompt_res << " ompd: " << ompd_res << endl;
    i++;
    if (ompt_res != ompd_res) res=false;
  }
  return res;
}

bool odbCheckParallelNumThreads(OMPDFunctionsPtr functions, vector<ompd_parallel_handle_t*> phs)
{
  bool res=true;
//  ompd_rc_t ret;
  int i=0;
  uint64_t ompt_res, ompd_res;
//   ((OMPDContext*)context)->setThisGdbContext();
  for (auto ph : phs)
  {
    stringstream ss;
    ss << "p ompt_get_num_threads(" << i << ")";
    ompt_res = evalGdbExpression(ss.str());
    /*ret = */functions->ompd_get_parallel_id(ph, &ompd_res);
    sout << "  parallelid ompt: " << ompt_res << " ompd: " << ompd_res << endl;
    i++;
    if (ompt_res != ompd_res) res=false;
  }
  return res;
}

bool odbCheckTaskIDs(OMPDFunctionsPtr functions, vector<ompd_task_handle_t*> ths)
{
  bool res=true;
//  ompd_rc_t ret;
  int i=0;
  uint64_t ompt_res, ompd_res;
//   ((OMPDContext*)context)->setThisGdbContext();
  for (auto th : ths)
  {
    stringstream ss;
    ss << "p ompt_get_task_id(" << i << ")";
    ompt_res = evalGdbExpression(ss.str());
    /*ret =*/ functions->ompd_get_task_id(th, &ompd_res);
    sout << "  taskid ompt: " << ompt_res << " ompd: " << ompd_res << endl;
    i++;
    if (ompt_res != ompd_res) res=false;
  }
  return res;
}

vector<ompd_task_handle_t*> odbGetTaskRegions(OMPDFunctionsPtr functions, ompd_thread_handle_t* th)
{
  ompd_rc_t ret;
  ompd_task_handle_t * task_handle;
  vector<ompd_task_handle_t*> task_handles;
  ret = functions->ompd_get_top_task_region(
          th, &task_handle);  
  while(ret == ompd_rc_ok)
  {
    task_handles.push_back(task_handle);
    ret = functions->ompd_get_ancestor_task_region(
          task_handle, &task_handle);  
  }
  return task_handles;
}

vector<ompd_task_handle_t*> odbGetImplicitTasks(OMPDFunctionsPtr functions, ompd_parallel_handle_t* ph)
{
//  ompd_rc_t ret;
  ompd_task_handle_t** task_handles;
  int num_tasks;
  vector<ompd_task_handle_t*> return_handles;
  /*ret = */functions->ompd_get_implicit_task_in_parallel(
          ph, &task_handles, &num_tasks);  
  for(int i=0; i<num_tasks; i++)
  {
    return_handles.push_back(task_handles[i]);
  }
  free(task_handles);
  return return_handles;
}

void OMPDTest::execute() const
{ 
//  ompd_rc_t ret;

  if (extraArgs.empty() || extraArgs[0] == "help")
  {
    hout << "Test suites available: threads, parallel, tasks" << endl;
    return;
  }     

  if (extraArgs[0] == "threads")
  {
    if(extraArgs.size()>1)
    {
      hout << "Usage: odb testapi threads" << endl;
      return;
    }


    auto thread_handles = odbGetThreadHandles(addrhandle, functions);
    for(auto thr_h: thread_handles)
    {
      auto parallel_h = odbGetParallelRegions(functions, thr_h);
      auto task_h = odbGetTaskRegions(functions, thr_h);
      
      sout << "Thread handle: 0x" << hex << thr_h << endl << "Parallel: ";
      for(auto ph: parallel_h)
      {
        sout << "Parallel handle: 0x" << hex << ph << endl;
        sout << "implicit Tasks: ";
        auto implicit_task_h = odbGetImplicitTasks(functions, ph);
        for(auto ith: implicit_task_h)
        {
          uint64_t tid;
          functions->ompd_get_task_id(
               ith, &tid);
          sout << "0x" << hex << ith << " (" << tid << "), ";
          functions->ompd_release_task_handle(ith);
        }
        sout << endl;
      }
      sout << endl << "Tasks: ";
      for(auto th: task_h){
        sout << "0x" << hex << th << ", ";
      }
      sout << endl;
      pthread_t            osthread;
      functions->ompd_get_osthread(thr_h, ompd_osthread_pthread, sizeof(pthread_t), &osthread);
      host_contextPool->getThreadContext(&osthread)->setThisGdbContext();
      odbCheckParallelIDs(functions, parallel_h);
      odbCheckTaskIDs(functions, task_h);
      for(auto ph: parallel_h)
        functions->ompd_release_parallel_handle(ph);
      for(auto th: task_h)
        functions->ompd_release_task_handle(th);
      functions->ompd_release_thread_handle(thr_h);
    }
  }


}

const char* OMPDTest::toString() const
{
  return "odb api";
}
