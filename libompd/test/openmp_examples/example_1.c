// RUN: %gdb-compile-and-run

#include <stdio.h>
#include <unistd.h>
#include <omp.h>
#include <pthread.h>
#include "../ompt_plugin.h"

void createPthreads()
{
  int numThreads = 2;
  pthread_t threads[numThreads];
  int i;
  for(i=0; i < numThreads; ++i)
    pthread_create(&threads[i], NULL, ompd_tool_test, NULL);
}

int main()
{
  omp_set_num_threads(4);
  printf("Application: Process %d started.\n", getpid());
  createPthreads();

  // Parallel region 1
  #pragma omp parallel
  {
    ompd_tool_test(0);
  }

  return 0;
}

// CHECK-NOT: "Error"
// CHECK-NOT: "The program is not being run."
