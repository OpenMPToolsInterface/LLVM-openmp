// RUN: %libomp-tool %S/tool.c && %libomp-compile && env OMP_TOOL_LIBRARIES=%T/tool.so %libomp-run | FileCheck %s
// REQUIRES: ompt

#include "omp.h"

int main()
{
  #pragma omp parallel num_threads(2)
  {
  }


  // Check if libomp supports the callbacks for this test.
  // CHECK-NOT: {{^}}0: Could not register callback 
  
  // CHECK: {{^}}0: NULL_POINTER=[[NULL:.*$]]
  // CHECK: {{^}}0: ompt_event_runtime_shutdown

  return 0;
}
