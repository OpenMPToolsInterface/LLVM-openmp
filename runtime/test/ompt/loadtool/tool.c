// RUN: true
#include <stdio.h>
#include <inttypes.h>
#include <omp.h>
#include <ompt.h>
#include <execinfo.h>

int ompt_initialize(
  ompt_function_lookup_t lookup,
  ompt_fns_t* fns)
{
  printf("0: NULL_POINTER=%p\n", (void*)NULL);
  return 1; //success
}

void ompt_finalize(ompt_fns_t* fns)
{
  printf("%d: ompt_event_runtime_shutdown\n", omp_get_thread_num());
}

ompt_fns_t* ompt_start_tool(
  unsigned int omp_version,
  const char *runtime_version)
{
  static ompt_fns_t ompt_fns = {&ompt_initialize,&ompt_finalize};
  return &ompt_fns;
}
