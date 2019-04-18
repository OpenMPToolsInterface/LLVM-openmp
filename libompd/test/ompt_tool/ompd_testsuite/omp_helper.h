typedef enum omp_sched_t { 
  omp_sched_static = 1, 
  omp_sched_dynamic = 2,
  omp_sched_guided = 3, 
  omp_sched_auto = 4
} omp_sched_t;
typedef enum omp_proc_bind_t {
  omp_proc_bind_false = 0, 
  omp_proc_bind_true = 1,
  omp_proc_bind_master = 2,
  omp_proc_bind_close = 3,
  omp_proc_bind_spread = 4 }
omp_proc_bind_t;
                                                
                                                    

#define FOREACH_OMP_INQUIRY_FN(macro)\
    macro (omp_get_num_threads) \
    macro (omp_get_dynamic) \
    macro (omp_get_nested) \
    macro (omp_get_max_threads) \
    macro (omp_get_thread_num) \
    macro (omp_get_num_procs) \
    macro (omp_in_parallel) \
    macro (omp_in_final) \
    macro (omp_get_active_level) \
    macro (omp_get_level) \
    macro (omp_get_ancestor_thread_num) \
    macro (omp_get_team_size) \
    macro (omp_get_thread_limit) \
    macro (omp_get_max_active_levels) \
    macro (omp_get_schedule) \
    macro (omp_get_proc_bind)

#define OMP_API_FNTYPE(fn) fn##_t
   
#define OMP_API_FUNCTION(return_type, fn, args)  \
    typedef return_type (*OMP_API_FNTYPE(fn)) args  


OMP_API_FUNCTION(int, omp_get_num_threads,  (void));
OMP_API_FUNCTION(int, omp_get_dynamic    ,  (void));
OMP_API_FUNCTION(int, omp_get_nested     ,  (void));
OMP_API_FUNCTION(int, omp_get_max_threads,  (void));
OMP_API_FUNCTION(int, omp_get_thread_num ,  (void));
OMP_API_FUNCTION(int, omp_get_num_procs  ,  (void));
OMP_API_FUNCTION(int, omp_in_parallel    ,  (void));
OMP_API_FUNCTION(int, omp_in_final       ,  (void));
OMP_API_FUNCTION(int, omp_get_active_level    ,    (void));
OMP_API_FUNCTION(int, omp_get_level           ,    (void));
OMP_API_FUNCTION(int, omp_get_ancestor_thread_num, (int));
OMP_API_FUNCTION(int, omp_get_team_size       ,    (int));
OMP_API_FUNCTION(int, omp_get_thread_limit    ,    (void));
OMP_API_FUNCTION(int, omp_get_max_active_levels,   (void));
OMP_API_FUNCTION(int, omp_get_schedule         ,   (omp_sched_t *, int *));
OMP_API_FUNCTION(omp_proc_bind_t, omp_get_proc_bind, (void));

