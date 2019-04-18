#include <vector>
#include <pthread.h>
class Thread
{
  // private classes of Thread
  private:
    class Parallel
    {
      private:
         ompd_thread_handle_t* thread;
         std::vector<ompd_parallel_handle_t*> regions;
      public:

        Parallel(ompd_thread_handle_t* parent_thread);
        ~Parallel();
        
        // tests on parallel regions
        
        // 6.1
        ompd_rc_t compare_get_num_threads();
        ompd_rc_t compare_get_level();
        ompd_rc_t compare_get_active_level();
    
        // 6.2
        
      private:
        ompd_rc_t create_vector_of_par_regions();
        
      public:
        // helper functions to compare event parameters
        ompd_rc_t compare_eventparam_parallel_id(ompt_parallel_id_t ompt_parallel_id);
        
        ompd_rc_t compare_get_parallel_id();
      
      
    };
    
    class Task
    {
      private:
        ompd_thread_handle_t* thread;
        std::vector<ompd_task_handle_t*> regions;
      
      public:
        Task(ompd_thread_handle_t* parent_thread);
        ~Task();
        
        // tests on task regions
        
        // 8.1
        
        ompd_rc_t compare_get_task_function();
        
        // 8.2
        
        // helper functions to compare event parameters
        ompd_rc_t compare_eventparam_parent_task_id(ompt_task_id_t ompt_task_id);
        ompd_rc_t compare_eventparam_parent_frame(ompt_frame_t ompt_frame);
        ompd_rc_t compare_eventparam_task_id(ompt_task_id_t ompt_task_id);
        
        ompd_rc_t compare_get_max_threads();
        
        //TODO: decide to which place this function belongs to
        // in ompd-tr: part of 8.2
        // in ompd.h: part of
        ompd_rc_t compare_get_thread_num();
        
        ompd_rc_t compare_in_parallel();
        ompd_rc_t compare_in_final();
        ompd_rc_t compare_get_dynamic();
        ompd_rc_t compare_get_nested();
        ompd_rc_t compare_get_max_active_levels();
        ompd_rc_t compare_get_schedule();
        ompd_rc_t compare_get_proc_bind();
      
      // 8.3
      
      private:
        ompd_rc_t create_vector_of_task_regions();
      
      public:
        ompd_rc_t compare_get_task_frame();
        ompd_rc_t compare_get_task_id();
    };
    
  // private members of Thread
  private:
    ompd_address_space_handle_t * addr_handle;
    ompd_thread_handle_t* thread_handle;
    Thread(pthread_t tid);
    Thread(const Thread&);
    static __thread Thread* instance;
    ~Thread();
    Thread& operator=(const Thread&);
    
    // tests on thread
    
    // 7.2
    bool is_waiting(ompd_state_t state);
    ompd_rc_t compare_get_state();
  
  // public member of Thread  
  public: 
    static Thread* getInstance();
    void runThreadTests();
    void runParallelTests();
    void runTaskTests();
    
};
