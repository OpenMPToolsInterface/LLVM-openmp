// RUN: %libomp-compile-and-run | %sort-threads | FileCheck %s
// REQUIRES: ompt
// UNSUPPORTED: gcc-4, gcc-5, gcc-6, gcc-7
#include "callback.h"
#include <omp.h>

int main()
{
  int a[2][2];
  int i,j;
#pragma omp parallel num_threads(2)
#pragma omp for ordered(2) schedule(static, 2)
  for (i = 0; i < 2; i++)
    for (j = 0; j < 2; j++)
    {
      a[i][j] = a[i][j] + 1;
#pragma omp ordered depend(sink:i-1,j) depend(sink:i,j-1)
      if (i>0 && j>0)
        a[i][j] = a[i-1][j] + a[i][j-1] + 1;
#pragma omp ordered depend(source)
    }

  // CHECK: 0: NULL_POINTER=[[NULL:.*$]]

  // CHECK: {{^}}[[MASTER:[0-9]+]]: ompt_event_task_dependences:
  // CHECK-SAME: task_id={{[0-9]+}}, deps=[0:type_source, ], ndeps=1
  // CHECK: {{^}}[[MASTER]]: ompt_event_release_doacross:
  // CHECK-SAME: wait_id=[[WAIT_ID1:[0-9]+]]
  
  // CHECK: {{^}}[[MASTER]]: ompt_event_wait_doacross: 
  // CHECK-SAME: wait_id=[[WAIT_ID2:[0-9]+]], hint=0, impl=0, codeptr_ra=
  // CHECK: {{^}}[[MASTER]]: ompt_event_task_dependences: 
  // CHECK-SAME: task_id={{[0-9]+}}, deps=[0:type_sink, ], ndeps=1
  // CHECK: {{^}}[[MASTER]]: ompt_event_acquired_doacross: 
  // CHECK-SAME: wait_id=[[WAIT_ID2]], codeptr_ra=
  // CHECK: {{^}}[[MASTER:[0-9]+]]: ompt_event_task_dependences:
  // CHECK-SAME: task_id={{[0-9]+}}, deps=[1:type_source, ], ndeps=1
  // CHECK: {{^}}[[MASTER]]: ompt_event_release_doacross:
  // CHECK-SAME: wait_id=[[WAIT_ID2]]

  // CHECK: {{^}}[[WORKER:[0-9]+]]: ompt_event_wait_doacross: 
  // CHECK-SAME: wait_id=[[WAIT_ID3:[0-9]+]], hint=0, impl=0, codeptr_ra=
  // CHECK: {{^}}[[WORKER]]: ompt_event_task_dependences: 
  // CHECK-SAME: task_id={{[0-9]+}}, deps=[0:type_sink, ], ndeps=1
  // CHECK: {{^}}[[WORKER]]: ompt_event_acquired_doacross: 
  // CHECK-SAME: wait_id=[[WAIT_ID3]], codeptr_ra=
  // CHECK: {{^}}[[WORKER:[0-9]+]]: ompt_event_wait_doacross: 
  // CHECK-SAME: wait_id=[[WAIT_ID3:[0-9]+]], hint=0, impl=0, codeptr_ra=
  // CHECK: {{^}}[[WORKER]]: ompt_event_task_dependences: 
  // CHECK-SAME: task_id={{[0-9]+}}, deps=[1:type_sink, ], ndeps=1
  // CHECK: {{^}}[[WORKER]]: ompt_event_acquired_doacross: 
  // CHECK-SAME: wait_id=[[WAIT_ID3]], codeptr_ra=
  // CHECK: {{^}}[[WORKER]]: ompt_event_task_dependences:
  // CHECK-SAME: task_id={{[0-9]+}}, deps=[2:type_source, ], ndeps=1
  // CHECK: {{^}}[[WORKER]]: ompt_event_release_doacross:
  // CHECK-SAME: wait_id=[[WAIT_ID3]]

  // CHECK: {{^}}[[WORKER]]: ompt_event_wait_doacross: 
  // CHECK-SAME: wait_id=[[WAIT_ID4:[0-9]+]], hint=0, impl=0, codeptr_ra=
  // CHECK: {{^}}[[WORKER]]: ompt_event_task_dependences: 
  // CHECK-SAME: task_id={{[0-9]+}}, deps=[1:type_sink, ], ndeps=1
  // CHECK: {{^}}[[WORKER]]: ompt_event_acquired_doacross: 
  // CHECK-SAME: wait_id=[[WAIT_ID4]], codeptr_ra=
  // CHECK: {{^}}[[WORKER]]: ompt_event_wait_doacross: 
  // CHECK-SAME: wait_id=[[WAIT_ID4:[0-9]+]], hint=0, impl=0, codeptr_ra=
  // CHECK: {{^}}[[WORKER]]: ompt_event_task_dependences: 
  // CHECK-SAME: task_id={{[0-9]+}}, deps=[2:type_sink, ], ndeps=1
  // CHECK: {{^}}[[WORKER]]: ompt_event_acquired_doacross: 
  // CHECK-SAME: wait_id=[[WAIT_ID4]], codeptr_ra=
  // CHECK: {{^}}[[WORKER:[0-9]+]]: ompt_event_task_dependences:
  // CHECK-SAME: task_id={{[0-9]+}}, deps=[3:type_source, ], ndeps=1
  // CHECK: {{^}}[[WORKER]]: ompt_event_release_doacross:
  // CHECK-SAME: wait_id=[[WAIT_ID4]]
  return 0;
}
