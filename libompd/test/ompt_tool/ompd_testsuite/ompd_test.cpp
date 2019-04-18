#include <stdio.h>
#include <stdlib.h>
#include "ompd_test_internal.h"
#include "ompd_test_Thread.h"
#include "OMPDCallbacks.h"
#include <pthread.h>
#include <sys/types.h>
//#define _GNU_SOURCE          /* See feature_test_macros(7) */
#include <unistd.h>
#include <sys/syscall.h>    /* For SYS_xxx definitions */
#include <dlfcn.h>
#include <omp.h>

//#define THREAD_ADDR 1

using namespace std;

static ompt_function_lookup_t ompt_lookup;
static ompd_address_space_handle_t * addrhandle;

#define declare_inquery_fn(F) static F##_t F;
FOREACH_OMPT_INQUIRY_FN(declare_inquery_fn)
//FOREACH_OMP_INQUIRY_FN(declare_inquery_fn)
#undef declare_inquery_fn
int checks=0;

static void
OMPT_Event_thread_begin(
  ompt_thread_type_t thread_type,
  ompt_thread_id_t thread_id)
{
  Thread* t = Thread::getInstance();
}

ompd_rc_t detectedError(ompd_rc_t ret = ompd_rc_error){ return ret; }
        
void ompt_initialize_fn (
ompt_function_lookup_t lookup,
const char *runtime_version,
unsigned int ompt_version
)
{
  ompt_lookup=lookup;
  printf("runtime_version: %s, ompt_version: %i\n", runtime_version, ompt_version);
  #define declare_inquery_fn(F) F = (F##_t)lookup(#F);
  FOREACH_OMPT_INQUIRY_FN(declare_inquery_fn)
  #undef declare_inquery_fn
/*  #define declare_inquery_fn(F) F = (F##_t) dlsym (NULL, #F);
  FOREACH_OMP_INQUIRY_FN(declare_inquery_fn)
  #undef declare_inquery_fn */
  ompt_set_callback_t ompt_set_callback = (ompt_set_callback_t) lookup("ompt_set_callback");
  ompt_set_callback(ompt_event_thread_begin, (ompt_callback_t) &OMPT_Event_thread_begin);
  
#ifndef THREAD_ADDR
  ompd_rc_t ret = initOMPDCallback(&addrhandle);
  if (ret != ompd_rc_ok)
  {
    printf("Init of OMPD failed! exiting ...\n");
    exit(1);
  }
#endif
  Thread* t = Thread::getInstance();
  
//  ompt_get_task_frame = (ompt_get_task_frame_t)lookup("ompt_get_task_frame");
//  ompt_get_task_id = (ompt_get_task_id_t)lookup("ompt_get_task_id");
}

ompt_initialize_t ompt_tool(void)
{
  return ompt_initialize_fn;
}


Thread::Parallel::Parallel(ompd_thread_handle_t* parent_thread)
{ 
  thread = parent_thread;     
    
  ompd_rc_t ret;   
  ret = create_vector_of_par_regions();
  if(ret != ompd_rc_ok) printf("error: could not initialize Parallel object\n");               
}
  
Thread::Parallel::~Parallel()
{
  int i;
  size_t num_regions;
  num_regions = regions.size();
  for( i = 0; i < num_regions; i++)
  {
    ompd_release_parallel_handle( regions.at(i) );  
  } 
}

/* --- 6 Parallel Region Inqueries ------------------------------------------ */

/* --- 6.1 Settings --------------------------------------------------------- */
ompd_rc_t Thread::Parallel::compare_get_num_threads()
{
  ompd_tword_t ompd_team_size;
  int omp_team_size;
  ompd_rc_t ret;
  
  ret = ompd_get_num_threads(regions[0], &ompd_team_size);  
  if(ret != ompd_rc_ok) return detectedError();

  omp_team_size = omp_get_num_threads();
    
  if((int) ompd_team_size != omp_team_size) return detectedError();

  return ompd_rc_ok;
} 

ompd_rc_t Thread::Parallel::compare_get_level()
{
  ompd_tword_t ompd_level;
  int omp_level; 
  ompd_rc_t ret;
  
  ret = ompd_get_level(regions[0], &ompd_level);
  if(ret != ompd_rc_ok) return detectedError();

  omp_level = omp_get_level();

  if((int) ompd_level != omp_level) return detectedError();          
  
  return ompd_rc_ok;
}

ompd_rc_t Thread::Parallel::compare_get_active_level()
{
  ompd_tword_t ompd_active_level;
  int omp_active_level;
  ompd_rc_t ret;
  
  ret = ompd_get_active_level(regions[0], &ompd_active_level);
  if(ret != ompd_rc_ok) return detectedError();
  
  omp_active_level = omp_get_active_level();
  
  if((int) ompd_active_level != omp_active_level) return detectedError();
  
  return ompd_rc_ok;
}

/* --- 6.2 OMPT Parallel Region Inquiry Analogues ------------------------- */

ompd_rc_t Thread::Parallel::create_vector_of_par_regions()
{
  ompd_parallel_handle_t* region;
  ompd_rc_t ret;

  ret = ompd_get_top_parallel_region(thread, &region);
  if(ret != ompd_rc_ok) printf("error: could not initialize vector of parallel regions\n");
  
  do
  {
    regions.push_back(region);
    ret = ompd_get_enclosing_parallel_handle(region, &region);
  }while(ret == ompd_rc_ok);
  
  return ompd_rc_ok; 
}

ompd_rc_t Thread::Parallel::compare_get_parallel_id()
{
  size_t num_regions;
  ompd_parallel_id_t ompd_id;
  ompt_parallel_id_t ompt_id;
  ompd_rc_t ret;
  
  num_regions = regions.size();
  
  int i;
  for( i = 0; i < num_regions; i++ )
  {
      ret = ompd_get_parallel_id(regions.at(i), &ompd_id);
      if(ret != ompd_rc_ok) return detectedError();
      
      ompt_id = ompt_get_parallel_id(i);        
/*      if(ompt_id == ompt_parallel_id_none)
      { 
        printf("ompt error: no parallel region at level: %d\n", i);
        return detectedError();
      }*/ 
      if(ompd_id != ompt_id)
      { 
        printf("compare_get_parallel_id: %i != %i \n", ompd_id, ompt_id);
        return detectedError();     
      }
  }

  return ompd_rc_ok;    
}

// helper functions to compare event parameters
ompd_rc_t Thread::Parallel::compare_eventparam_parallel_id(ompt_parallel_id_t ompt_parallel_id)
{
  ompd_parallel_id_t par_id;
  ompd_rc_t ret; 
  
  ret = ompd_get_parallel_id(regions[0], &par_id);
  if(ret != ompd_rc_ok) return detectedError(); 

  if((uint64_t) par_id != (uint64_t) ompt_parallel_id) return detectedError();
  
  return ompd_rc_ok;
}


Thread::Task::Task(ompd_thread_handle_t* parent_thread)
{
  thread = parent_thread; 
  
  ompd_rc_t ret;
  ret = create_vector_of_task_regions();
  if(ret != ompd_rc_ok) printf("error: could not initialize task object\n");
}

Thread::Task::~Task()
{
  int i;
  size_t num_regions;
  num_regions = regions.size();
  for( i = 0; i < num_regions; i++)
  {
    ompd_release_task_handle( regions.at(i) );  
  } 
}

/* --- 8 Task Inquiry ------------------------------------------------------- */

/* --- 8.1 Task Function Entry Point ---------------------------------------- */

ompd_rc_t Thread::Task::compare_eventparam_parent_task_id(ompt_task_id_t ompt_task_id)
{
  ompd_parallel_id_t task_id;
  ompd_rc_t ret;
  
  ret = ompd_get_task_id(regions[1], &task_id);
  if(ret != ompd_rc_ok) return detectedError();
  
  if( (task_id) != ompt_task_id) return detectedError();
  
  return ompd_rc_ok;
}

ompd_rc_t Thread::Task::compare_eventparam_parent_frame(ompt_frame_t ompt_frame)
{
  ompd_address_t ompd_exit;
  ompd_address_t ompd_reentry;
  ompd_rc_t ret;
  
  ret = ompd_get_task_frame(regions[0], &ompd_exit, &ompd_reentry);
  if(ret != ompd_rc_ok) return detectedError();
  
  if((void*)ompd_exit.address != (ompt_frame.exit_runtime_frame)) return detectedError();
  if((void*)ompd_reentry.address != (ompt_frame.reenter_runtime_frame)) return detectedError();
  
  return ompd_rc_ok;
}

ompd_rc_t Thread::Task::compare_eventparam_task_id(ompt_task_id_t ompt_task_id)
{
  ompd_task_id_t task_id;
  ompd_rc_t ret;
  
  ret = ompd_get_task_id(regions[0], &task_id);
  if(ret != ompd_rc_ok) return detectedError();
  
  if((uint64_t) (task_id) != (uint64_t) ompt_task_id) return detectedError();
  
  return ompd_rc_ok;
}


// TODO: 
ompd_rc_t Thread::Task::compare_get_task_function()
{
}  

/* --- 8.2 Task Settings ---------------------------------------------------- */
 
ompd_rc_t Thread::Task::compare_get_max_threads()
{
  ompd_tword_t ompd_max_threads;
  int omp_max_threads;   
  ompd_rc_t ret;

  ret = ompd_get_max_threads(regions[0], &ompd_max_threads);
  if(ret != ompd_rc_ok) return detectedError();

  omp_max_threads = omp_get_max_threads();

  if((int) (ompd_max_threads) != omp_max_threads) return detectedError();

  return ompd_rc_ok;
}

// TODO: see header file
ompd_rc_t Thread::Task::compare_get_thread_num()
{
  ompd_tword_t ompd_num;
  int omp_num;
  ompd_rc_t ret;
  
  ret = ompd_get_thread_num(thread, &ompd_num);
  if(ret != ompd_rc_ok) return detectedError();
  
  omp_num = omp_get_thread_num();
  
  if((int) ompd_num != omp_num) return detectedError();
  
  return ompd_rc_ok;
}

ompd_rc_t Thread::Task::compare_in_parallel()
{
  ompd_tword_t ompd_val;
  int omp_val; 
  ompd_rc_t ret;

  ret = ompd_in_parallel(regions[0], &ompd_val);
  if(ret != ompd_rc_ok) return detectedError();

  omp_val = omp_in_parallel();

  if((int) ompd_val != omp_val) return detectedError();

  return ompd_rc_ok; 
}

ompd_rc_t Thread::Task::compare_in_final()
{
  ompd_tword_t ompd_val;
  int omp_val; 
  ompd_rc_t ret;

  ret = ompd_in_final(regions[0], &ompd_val);
  if(ret != ompd_rc_ok) return detectedError();

  omp_val = omp_in_final();

  if((int) ompd_val != omp_val) return detectedError();

  return ompd_rc_ok; 
}

ompd_rc_t Thread::Task::compare_get_dynamic()
{
  ompd_tword_t ompd_val;
  int omp_val; 
  ompd_rc_t ret;

  ret = ompd_get_dynamic(regions[0], &ompd_val);
  if(ret != ompd_rc_ok) return detectedError();

  omp_val = omp_get_dynamic();

  if((int) ompd_val != omp_val) return detectedError();

  return ompd_rc_ok; 
}
          
ompd_rc_t Thread::Task::compare_get_nested()
{
  ompd_tword_t ompd_val;
  int omp_val; 
  ompd_rc_t ret;

  ret = ompd_get_nested(regions[0], &ompd_val);
  if(ret != ompd_rc_ok) return detectedError();

  omp_val = omp_get_nested();

  if((int) ompd_val != omp_val) return detectedError();

  return ompd_rc_ok; 
}

ompd_rc_t Thread::Task::compare_get_max_active_levels()
{
  ompd_tword_t ompd_max_levels;
  int omp_max_levels; 
  ompd_rc_t ret;

  ret = ompd_get_max_active_levels(regions[0], &ompd_max_levels);
  if(ret != ompd_rc_ok) return detectedError();

  omp_max_levels = omp_get_max_active_levels();

  if((int) ompd_max_levels != omp_max_levels) return detectedError();

  return ompd_rc_ok; 
}

ompd_rc_t Thread::Task::compare_get_schedule()
{
  ompd_sched_t ompd_kind;
  ompd_tword_t ompd_modifier;
  omp_sched_t omp_kind;
  int omp_modifier;
  ompd_rc_t ret;

  ret = ompd_get_schedule(regions[0], &ompd_kind, &ompd_modifier);
  if(ret != ompd_rc_ok) return detectedError();

  omp_get_schedule(&omp_kind, &omp_modifier);

  if((int) ompd_kind != (int) omp_kind) return detectedError();
  if( (   ompd_kind == ompd_sched_static
       || ompd_kind == ompd_sched_dynamic 
       || ompd_kind == ompd_sched_guided
      ) 
      && (int) ompd_modifier != omp_modifier
    ) return detectedError();
  
  return ompd_rc_ok; 
}

ompd_rc_t Thread::Task::compare_get_proc_bind()
{
  ompd_proc_bind_t ompd_bind;
  omp_proc_bind_t omp_bind;
  ompd_rc_t ret;
  
  ret = ompd_get_proc_bind(regions[0], &ompd_bind);
  if(ret != ompd_rc_ok) return detectedError();
  
  omp_bind = omp_get_proc_bind();
  
  if((int) ompd_bind != (int) omp_bind) return detectedError();
  
  return ompd_rc_ok;
} 


/* --- 8.3 OMPT Task Inquiry Analogues -------------------------------------- */
ompd_rc_t Thread::Task::create_vector_of_task_regions()
{
  ompd_task_handle_t* region;
  ompd_rc_t ret;

  ret = ompd_get_top_task_region(thread, &region);
  if(ret != ompd_rc_ok) 
  {
    printf("error: could not create vector of task regions\n");
    return detectedError();
  }
  
  do
  {
    regions.push_back(region);
    ret = ompd_get_ancestor_task_region(region, &region);
  }while(ret == ompd_rc_ok);
  
  return ompd_rc_ok;
}

ompd_rc_t Thread::Task::compare_get_task_frame()
{ 
  ompd_address_t ompd_exit;
  ompd_address_t ompd_reentry;
  ompt_frame_t* ompt_frame;
  size_t num_regions; 
  ompd_rc_t ret;  
  
  num_regions = regions.size();
  
  int i;
  for( i = 0; i < num_regions; i++)
  {
      ret = ompd_get_task_frame(regions.at(i), &ompd_exit, &ompd_reentry);
      if(ret != ompd_rc_ok) return detectedError();
      
      ompt_frame = ompt_get_task_frame( i );
      
      if((void*)ompd_exit.address != (ompt_frame->exit_runtime_frame)) return detectedError();
      if((void*)ompd_reentry.address != (ompt_frame->reenter_runtime_frame)) return detectedError();
  }
  
  return ompd_rc_ok;
}

ompd_rc_t Thread::Task::compare_get_task_id()
{

  size_t num_regions;
  ompd_task_id_t ompd_id;
  ompt_task_id_t ompt_id;
  ompd_rc_t ret;
  
  num_regions = regions.size();
  
  int i;
  for( i = 0; i < num_regions; i++ )
  {
      ret = ompd_get_task_id(regions.at(i), &ompd_id);
      if(ret != ompd_rc_ok) return detectedError();
      
      ompt_id = ompt_get_task_id(i);
/*      if(ompt_id == ompt_task_id_none)
      {
        printf("ompt error: no task region at level: %d\n", i);
        return detectedError();
      }*/
      if(ompd_id != ompt_id) return detectedError();    
  }
  
  return ompd_rc_ok;    
}
      

Thread::Thread(pthread_t tid):thread_handle(NULL)
{
  ompd_rc_t ret;
#ifdef THREAD_ADDR
  ret = initOMPDCallback(&addr_handle);
  if (ret != ompd_rc_ok)
  {
    printf("Init of OMPD failed! exiting ...\n");
    exit(1);
  }
#endif
  ret = ompd_get_thread_handle(
#ifdef THREAD_ADDR
              addr_handle, 
#else
              addrhandle, 
#endif
              ompd_osthread_lwp, sizeof(pid_t), (void*) &tid, &thread_handle);
  if(ret != ompd_rc_ok) 
  {  
    printf("error: could not initialize a thread object for tid: %i\n", tid);  
    thread_handle=NULL;
  }
};

Thread::~Thread()
{
  ompd_rc_t ret;
  ret = ompd_release_thread_handle(thread_handle);   
  closeOMPDCallback(addrhandle);  
};

__thread Thread* Thread::instance=NULL;
Thread* Thread::getInstance()
{
  if(!instance)  
//    instance = new Thread(syscall(SYS_gettid));
    instance = new Thread(pthread_self());
  return instance;
}

bool Thread::is_waiting(ompd_state_t state)
{
  switch(state)
  {
    case ompd_state_work_serial:  
      return false;
    case ompd_state_work_parallel:  
      return false;
    case ompd_state_work_reduction:  
      return false;
    case ompd_state_idle:
      return true;
    case ompd_state_overhead:
      return false;
    case ompd_state_wait_barrier:
      return true;
    case ompd_state_wait_barrier_implicit:
      return true;
    case ompd_state_wait_barrier_explicit:
      return true;
    case ompd_state_wait_taskwait:
      return true;
    case ompd_state_wait_taskgroup:
      return true;
    case ompd_state_wait_lock:
      return true;
    case ompd_state_wait_nest_lock:
      return true;
    case ompd_state_wait_critical:
      return true;
    case ompd_state_wait_atomic:
      return true;
    case ompd_state_wait_ordered:
      return true;
    case ompd_state_wait_single:
      return true;
    case ompd_state_undefined:
      return false;
    default:
      printf("error: unknown thread state\n");
      return false;
  }
}

ompd_rc_t Thread::compare_get_state()
{
  ompd_state_t ompd_state;
  ompd_wait_id_t ompd_wait_id;
  ompt_state_t ompt_state;
  ompt_wait_id_t ompt_wait_id;
  ompd_rc_t ret;
  
  ret = ompd_get_state(thread_handle, &ompd_state, &ompd_wait_id);
  if(ret != ompd_rc_ok) return detectedError();
  
  ompt_state = ompt_get_state(&ompt_wait_id);
  
  if((uint64_t)(ompd_state) != (uint64_t) ompt_state) return detectedError();
  if( is_waiting(ompd_state) && ((uint64_t) (ompd_wait_id) != (uint64_t) (ompt_wait_id)) )
    return detectedError();
    
  return ompd_rc_ok;
}

void Thread::runThreadTests()
{
  if (thread_handle == NULL)
    return;
  __sync_add_and_fetch(&checks, 1);
  int errors = 0;
  ompd_rc_t ret = compare_get_state(); 
  if(ret != ompd_rc_ok)
  {
    errors++;
    printf("error: comparison of get_state failed\n");
  }
  
  printf("result of thread tests: %d errors\n", errors);  
}

void Thread::runParallelTests()
{
  if (thread_handle == NULL)
    return;
  __sync_add_and_fetch(&checks, 1);
  Parallel parallel_test = Parallel(thread_handle);
  int errors = 0;
  ompd_rc_t ret;

  // 6.1
  ret = parallel_test.compare_get_num_threads();
  if(ret != ompd_rc_ok)
  {
    errors++;
    printf("error: comparison of get_num_threads failed\n");
  } 

  ret = parallel_test.compare_get_level();
  if(ret != ompd_rc_ok)
  {
    errors++;
    printf("error: comparison of get_level failed\n");
  } 

  ret = parallel_test.compare_get_active_level();
  if(ret != ompd_rc_ok)
  {
    errors++;
    printf("error: comparison of get_active_level failed\n");
  } 

  // 6.2
  ret = parallel_test.compare_get_parallel_id();
  if(ret != ompd_rc_ok)
  {
    errors++;
    printf("error: comparison of get_parallel_id failed\n");
  } 

  printf("result of parallel tests: %d errors\n", errors);
  
}

void Thread::runTaskTests()
{
  if (thread_handle == NULL)
    return;
  __sync_add_and_fetch(&checks, 1);
  Task task_test = Task(thread_handle);
  int errors = 0;
  ompd_rc_t ret;

  ret = task_test.compare_get_max_threads();
  if(ret != ompd_rc_ok)
  {
    errors++;
    printf("error: comparison of get_max_threads failed\n");
  }
  
  ret = task_test.compare_get_thread_num();
  if(ret != ompd_rc_ok)
  {
    errors++;
    printf("error: comparison of get_thread_num failed\n");
  }
  
  ret = task_test.compare_in_parallel();
  if(ret != ompd_rc_ok)
  {
    errors++;
    printf("error: comparison of in_parallel failed\n");
  }
  
  ret = task_test.compare_in_final();
  if(ret != ompd_rc_ok)
  {
    errors++;
    printf("error: comparison of in_final failed\n");
  }
  
  ret = task_test.compare_get_dynamic();
  if(ret != ompd_rc_ok)
  {
    errors++;
    printf("error: comparison of get_dynamic failed\n");
  }
  
  ret = task_test.compare_get_nested();
  if(ret != ompd_rc_ok)
  {
    errors++;
    printf("error: comparison of get_nested failed\n");
  }
  
  ret = task_test.compare_get_max_active_levels();
  if(ret != ompd_rc_ok)
  {
    errors++;
    printf("error: comparison of get_max_active_levels failed\n");
  }
  
/*  ret = task_test.compare_get_schedule();
  if(ret != ompd_rc_ok)
  {
    errors++;
    printf("error: comparison of get_schedule failed\n");
  }*/
  
  ret = task_test.compare_get_proc_bind();
  if(ret != ompd_rc_ok)
  {
    errors++;
    printf("error: comparison of get_proc_bind failed\n");
  }
  
  ret = task_test.compare_get_task_id();
  if(ret != ompd_rc_ok)
  {
    errors++;
    printf("error: comparison of get_task_id failed\n");
  }

  printf("result of task tests: %d errors\n", errors);
} 

#ifdef  __cplusplus
extern "C"
#endif
void* ompd_tool_test(void* n)
{
  Thread* t = Thread::getInstance();
  t->runThreadTests();
  t->runParallelTests();
  t->runTaskTests();
  return NULL;
}
    
    
__attribute__ ((__constructor__))
void init(void)
{
}

__attribute__ ((__destructor__))
void fini(void)
{
  printf("Finished %i testsuites.\n",checks);
}
