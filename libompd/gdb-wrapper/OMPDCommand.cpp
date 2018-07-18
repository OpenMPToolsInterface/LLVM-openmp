/*
 * OMPDCommand.cpp
 *
 *  Created on: Dec 28, 2014
 *      Author: Ignacio Laguna
 *     Contact: ilaguna@llnl.gov
 */
//#include <config.h>
#include "OMPDCommand.h"
#include "OMPDContext.h"
#include "Callbacks.h"
#include "OutputString.h"
#include "Debug.h"
#include "omp.h"
#include "ompd.h"
//#include "ompd_test.h"
#define ODB_LINUX
#include "CudaGdb.h"

#include <cstdlib>
#include <dlfcn.h>
#include <iostream>
#include <sstream>

using namespace ompd_gdb;
using namespace std;

extern OMPDHostContextPool * host_contextPool;

/* --- OMPDIcvs ------------------------------------------------------------- */

OMPDIcvs::OMPDIcvs(OMPDFunctionsPtr functions,
                   ompd_address_space_handle_t *addrhandle)
    : functions(functions) {
  ompd_icv_id_t next_icv_id = ompd_icv_undefined;
  int more = 1;
  const char *next_icv_name_str;
  ompd_scope_t next_scope;
  ompd_rc_t ret = ompd_rc_ok;
  while (more && ret == ompd_rc_ok) {
    ret = functions->ompd_enumerate_icvs(addrhandle,
                                         next_icv_id,
                                         &next_icv_id,
                                         &next_icv_name_str,
                                         &next_scope,
                                         &more);
    if (ret == ompd_rc_ok) {
      availableIcvs[next_icv_name_str] =
          std::pair<ompd_icv_id_t, ompd_scope_t>(next_icv_id, next_scope);
    }
  }
}


ompd_rc_t OMPDIcvs::get(ompd_parallel_handle_t *handle, const char *name,
                        ompd_word_t *value) {
  ompd_icv_id_t icv;
  ompd_scope_t scope;

  auto &p = availableIcvs.at(name);
  icv = p.first;
  scope = p.second;

  if (scope != ompd_scope_parallel) {
    return ompd_rc_bad_input;
  }

  return functions->ompd_get_icv_from_scope((void *)handle, scope, icv, value);
}

/* --- OMPDCommandFactory --------------------------------------------------- */

OMPDCommandFactory::OMPDCommandFactory()
{
  functions = OMPDFunctionsPtr(new OMPDFunctions);

  // Load OMPD DLL and get a handle
#ifdef ODB_LINUX
  functions->ompdLibHandle = dlopen("libompd.so", RTLD_LAZY);
#elif defined(ODB_MACOS)
  functions->ompdLibHandle = dlopen("libompd.dylib", RTLD_LAZY);
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

  // Initialize OMPD library
  ompd_callbacks_t *table = getCallbacksTable();
  assert(table && "Invalid callbacks table");
  ompd_rc_t ret = functions->ompd_initialize(0, table);
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

  icvs = OMPDIcvsPtr(new OMPDIcvs(functions, addrhandle));
}

OMPDCommandFactory::~OMPDCommandFactory()
{
  ompd_rc_t ret;
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
    return new OMPDLevels(functions, addrhandle, icvs, extraArgs);
  else if (strcmp(str, "callback") == 0)
    return new OMPDCallback(functions, addrhandle, extraArgs);
  else if (strcmp(str, "api") == 0)
    return new OMPDApi(functions, addrhandle, extraArgs);
  else if (strcmp(str, "testapi") == 0)
    return new OMPDTest(functions, addrhandle, extraArgs);
  else if (strcmp(str, "parallel") == 0)
    return new OMPDParallelRegions(functions, addrhandle, icvs, extraArgs);
  else if (strcmp(str, "tasks") == 0)
    return new OMPDTasks(functions, addrhandle, icvs, extraArgs);
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
  // get state names
  map<ompd_word_t, const char*> host_state_names;
  ompd_word_t more_states = 1;
  ompd_word_t next_state = omp_state_undefined;
  host_state_names[next_state] = "ompd_state_undefined";
  while (more_states) {
    const char *state_name;
    functions->ompd_enumerate_states(addrhandle, next_state, &next_state, &state_name, &more_states);
    host_state_names[next_state] = state_name;
  }

  printf("\nHOST THREADS\n");
  printf("Debugger_handle    Thread_handle     System_thread\n");
  printf("--------------------------------------------------\n");

  auto thread_ids = getThreadIDsFromDebugger();
  for(auto i: thread_ids) {
    ompd_thread_handle_t* thread_handle;
    ompd_rc_t ret = functions->ompd_get_thread_handle(
        addrhandle, ompd_thread_id_pthread, sizeof(i.second),
        &(i.second), &thread_handle);
    if (ret == ompd_rc_ok)
    {
      ompd_word_t state;
      ompd_wait_id_t wait_id;
      ret = functions->ompd_get_state(thread_handle, &state, &wait_id);
      printf("  %-12u     %p     0x%lx\t%s\t%lx\n",
          (unsigned int)i.first, thread_handle, i.second, host_state_names[state], wait_id);
      functions->ompd_release_thread_handle(thread_handle);
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
  map<ompd_addr_t, ompd_address_space_handle_t*> address_spaces;
  ompd_word_t last_state = -1;
  ompd_cudathread_coord_t last_coords;
  vector<ompd_thread_handle_t *> device_thread_handles;

  // get cuda states
  map<ompd_word_t, const char*> cuda_state_names;
  more_states = 1;
  next_state = omp_state_undefined;
  cuda_state_names[next_state] = "omp_state_undefined";

  printf("\nCUDA THREADS\n");
  printf("Cuda block  from Thread  to Thread  state\n");
  printf("------------------------------------------\n");

  for(auto i: cuda.threads) {
    if (!device_initialized[i.coord.cudaContext]) {
      OMPDCudaContextPool* cpool;
      cpool = new OMPDCudaContextPool(&i);
      ompd_rc_t result;

      device_initialized[i.coord.cudaContext] = true;
      result = functions->ompd_device_initialize(
          addrhandle,
          cpool->getGlobalOmpdContext(),
          ompd_device_kind_cuda,
          sizeof(i.coord.cudaContext),
          &i.coord.cudaContext,
          &cpool->ompd_device_handle);

        if (result != ompd_rc_ok)
        {
          continue;
        }

        address_spaces[i.coord.cudaContext] = cpool->ompd_device_handle;
        while (more_states) {
          const char *state_name;
          functions->ompd_enumerate_states(cpool->ompd_device_handle,
                                           next_state, &next_state,
                                           &state_name, &more_states);
          cuda_state_names[next_state] = state_name;
        }
    }

    ompd_thread_handle_t* thread_handle;
    ompd_rc_t ret = functions->ompd_get_thread_handle(
                                    address_spaces[i.coord.cudaContext],
                                    ompd_thread_id_cudalogical,
                                    sizeof(i.coord), &i.coord,
                                    &thread_handle);

    if (ret == ompd_rc_ok)
    {
      ompd_word_t state;
      device_thread_handles.push_back(thread_handle);
      functions->ompd_get_state(thread_handle, &state, NULL);
      if (last_state == -1) {
        last_state = state;
        last_coords = i.coord;
        printf("(%li,0,0)  (%li,0,0)", i.coord.blockIdx.x, i.coord.threadIdx.x);
      } else if (state != last_state || i.coord.blockIdx.x != last_coords.blockIdx.x) {
        printf("  (%li,0,0)  %s\n", last_coords.threadIdx.x, cuda_state_names[last_state]);
        last_coords = i.coord;
        last_state = state;
        printf("(%li,0,0)  (%li,0,0)", i.coord.blockIdx.x, i.coord.threadIdx.x);
      } else { /* state == last_state*/
        last_coords = i.coord;
      }
      omp_cuda_threads++;
    }
  }
  // Check for non-unique handles
  for (auto i: device_thread_handles) {
    for (auto j: device_thread_handles) {
      int value;
      if (i == j) {
        continue;
      }
      ompd_rc_t ret = functions->ompd_thread_handle_compare(i, j, &value);
      if (!value) {
        printf("FOUND NON-UNIQUE THREAD HANDLES FOR DIFFERENT THREADS\n");
      }
    }
  }

  // release thread handles
  for (auto i: device_thread_handles) {
    functions->ompd_release_thread_handle(i);
  }

  if (last_state != -1) {
    printf("  (%li,0,0)  %s\n", last_coords.threadIdx.x, cuda_state_names[last_state]);
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
  ompd_rc_t ret;
  printf("\n");
  printf("Thread_handle     Nesting_level\n");
  printf("-------------------------------\n");
  for (auto i: getThreadIDsFromDebugger())
  {
    ompd_thread_handle_t *thread_handle;
    ompd_parallel_handle_t *parallel_handle;
    ret = functions->ompd_get_thread_handle(
        addrhandle, ompd_thread_id_pthread, sizeof(i.second) ,&(i.second), &thread_handle);
    if (ret != ompd_rc_ok) {
      continue;
    }
    ret = functions->ompd_get_current_parallel_handle(thread_handle,
                                                      &parallel_handle);
    if (ret == ompd_rc_ok)
    {
      ompd_word_t level=0;
      icvs->get(parallel_handle, "levels-var", &level);
      printf("%-12p      %ld\n", thread_handle, level);
    }
  }
}

const char* OMPDLevels::toString() const
{
  return "odb levels";
}


/* --- OMPDCallback ----------------------------------------------------------- */


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
    ompd_addr_t addr,
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
    ompd_addr_t addr = (ompd_addr_t)strtoll(extraArgs[1].c_str(), NULL, 0);
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
    ompd_addr_t *symbol_addr);*/

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
#if 0 // MARKER_MR: TODO: reimplement this functionality with breakpoints
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
#endif
    hout << "The 'odb api threads' command has been temporarily removed for the migration to a new ompd standard\n";
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
        addrhandle, ompd_thread_id_pthread, sizeof(i.second) ,&(i.second), &thread_handle);
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
  ret = functions->ompd_get_current_parallel_handle(
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
  sout << "Checking of parallel IDs has been disabled for upgrade of ompd in branch ompd-devices\n";
  // MARKER_MR: TODO: fix checking of parallel ids
  return true;
#if 0
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
#endif
}

bool odbCheckParallelNumThreads(OMPDFunctionsPtr functions, vector<ompd_parallel_handle_t*> phs)
{
  sout << "Checking of parallel IDs has been disable for upgrade of ompd in branch ompd-devices\n";
  // MARKER_MR: TODO: fix checking of parallel ids for num threads
  return true;
#if 0
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
#endif
}

bool odbCheckTaskIDs(OMPDFunctionsPtr functions, vector<ompd_task_handle_t*> ths)
{
  sout << "Checking of task IDs has been disable for upgrade of ompd in branch ompd-devices\n";
  // MARKER_MR: TODO: fix checking of task ids
  return true;
#if 0
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
#endif
}

vector<ompd_task_handle_t*> odbGetTaskRegions(OMPDFunctionsPtr functions, ompd_thread_handle_t* th)
{
  ompd_rc_t ret;
  ompd_task_handle_t *task_handle;
  vector<ompd_task_handle_t*> task_handles;
  ret = functions->ompd_get_current_task_handle(
          th, &task_handle);
  while(ret == ompd_rc_ok)
  {
    task_handles.push_back(task_handle);
    ret = functions->ompd_get_generating_task_handle(
          task_handle, &task_handle); // MARKER_MR: TODO: is it generating or scheduling task or something different?
  }
  return task_handles;
}

vector<ompd_task_handle_t*> odbGetImplicitTasks(OMPDFunctionsPtr functions, ompd_parallel_handle_t* ph)
{
//  ompd_rc_t ret;
  int num_tasks = evalGdbExpression("call omp_get_num_threads()");
  vector<ompd_task_handle_t*> return_handles;

  for (int i=0; i < num_tasks; ++i) {
    ompd_task_handle_t* task_handle;
    functions->ompd_get_task_in_parallel(
        ph, i, &task_handle);
    return_handles.push_back(task_handle);
  }
#if 0
  ompd_task_handle_t* task_handles;
  /*ret = */functions->ompd_get_task_in_parallel(
          ph, &task_handles, &num_tasks);
  for(int i=0; i<num_tasks; i++)
  {
    return_handles.push_back(task_handles[i]);
  }
  free(task_handles);
#endif
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
#if 0 //MARKER_MR: TODO: fix this
          uint64_t tid;
          functions->ompd_get_task_id(
               ith, &tid);
#endif
          sout << "0x" << hex << ith << " (" << "DISABLED IN ompd-devices" << "), ";
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
      functions->ompd_get_thread_id(thr_h, ompd_thread_id_pthread, sizeof(pthread_t), &osthread);
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

void OMPDParallelRegions::execute() const
{
  ompd_rc_t ret;
  auto host_thread_handles = odbGetThreadHandles(addrhandle, functions);

  OMPDParallelHandleCmp parallel_cmp_op(functions);
  std::map<ompd_parallel_handle_t *,
           std::vector<ompd_thread_handle_t *>,
           OMPDParallelHandleCmp> host_parallel_handles(parallel_cmp_op);
  for (auto t: host_thread_handles) {
    for (auto parallel_handle: odbGetParallelRegions(functions, t))
    {
      host_parallel_handles[parallel_handle].push_back(t);
    }
  }

  printf("HOST PARALLEL REGIONS\n");
  printf("Parallel Handle   Num Threads   ICV Num Threads   ICV level   ICV active level\n");
  printf("------------------------------------------------------------------------------\n");
  for (auto &p: host_parallel_handles) {
    ompd_word_t icv_num_threads, icv_level, icv_active_level;
    icvs->get(p.first, "ompd-team-size-var", &icv_num_threads);
    icvs->get(p.first, "levels-var", &icv_level);
    icvs->get(p.first, "active-levels-var", &icv_active_level);
    printf("%-15p   %-10zu   %-15ld   %-9ld   %ld\n", p.first, p.second.size(), icv_num_threads, icv_level, icv_active_level);
  }

  for (auto t: host_thread_handles) {
    functions->ompd_release_thread_handle(t);
  }
  for (auto &p: host_parallel_handles) {
    functions->ompd_release_parallel_handle(p.first);
  }
}

const char *OMPDParallelRegions::toString() const
{
  return "odb parallel";
}

void OMPDTasks::execute() const
{
  ompd_rc_t ret;
  auto host_thread_handles = odbGetThreadHandles(addrhandle, functions);
  OMPDTaskHandleCmp task_cmp_op(functions);
  std::map<ompd_task_handle_t *,
           std::vector<ompd_thread_handle_t *>,
           OMPDTaskHandleCmp> host_task_handles(task_cmp_op);
  for (auto t: host_thread_handles) {
    for (auto task_handle: odbGetTaskRegions(functions, t)) {
      host_task_handles[task_handle].push_back(t);
    }
  }

  printf("HOST TASKS\n");
  printf("Task Handle   Assoc. Threads   ICV Level   Enter Frame   Exit Frame\n");
  printf("-------------------------------------------------------------------\n");
  for (auto th: host_task_handles) {
    ompd_parallel_handle_t *ph;
    ret = functions->ompd_get_task_parallel_handle(th.first, &ph);
    if (ret != ompd_rc_ok) {
      printf("could not get parallel handle for nesting\n");
      continue;
    }

    ompd_word_t icv_level;
    icvs->get(ph, "levels-var", &icv_level);
    ompd_address_t enter_frame;
    ompd_address_t exit_frame;
    ret = functions->ompd_get_task_frame(th.first, &enter_frame, &exit_frame);
    printf("%-11p   %-14zu   %-9ld   %-11p   %-10p\n", th.first, th.second.size(), icv_level, (void*)enter_frame.address, (void*)exit_frame.address);
  }

  for (auto task: host_task_handles) {
    functions->ompd_release_task_handle(task.first);
  }

  for (auto thread: host_thread_handles) {
    functions->ompd_release_thread_handle(thread);
  }
}

const char *OMPDTasks::toString() const
{
  return "odb tasks";
}
