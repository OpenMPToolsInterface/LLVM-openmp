// RUN: %libomp-compile-and-run | FileCheck %s
// REQUIRES: ompt

#include "callback.h"

int main()
{
  int i,j;
  int x = 0;
#pragma omp target if(0)
#pragma omp teams num_teams(2) thread_limit(2)
#pragma omp distribute
  for (i=0; i< 2; i++)
#pragma omp parallel for num_threads(2)
    for (j=0;j< 2; j++)
#pragma omp atomic
      x++;

  // CHECK: 0: NULL_POINTER=[[NULL:.*$]]
  // CHECK: {{^}}[[MASTER_ID:[0-9]+]]: ompt_event_parallel_begin:
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_parallel_begin:
  // CHECK: {{^}}[[WORKER_ID:[0-9]+]]: ompt_event_parallel_begin:

  return 0;
}