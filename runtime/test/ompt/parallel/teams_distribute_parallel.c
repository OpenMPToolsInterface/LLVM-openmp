// RUN: %libomp-compile-and-run | %sort-threads | FileCheck %s
// REQUIRES: ompt, clang

#include "callback.h"

int main() {
  int i, j;
  int x = 0;
#pragma omp target if (0)
#pragma omp teams num_teams(2) thread_limit(2)
#pragma omp distribute
  for (i = 0; i < 2; i++)
#pragma omp parallel for num_threads(2)
    for (j = 0; j < 2; j++)
#pragma omp atomic
      x++;

  // CHECK: 0: NULL_POINTER=[[NULL:.*$]]
  // CHECK: {{^}}[[MASTER_ID:[0-9]+]]: ompt_event_task_create: parent_task_id=0,
  // CHECK-SAME: new_task_id=[[INITTIAL_TASK:[0-9]+]], codeptr_ra=(nil),
  // CHECK-SAME: task_type=ompt_task_initial=1
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_parallel_begin:
  // CHECK-SAME: parent_task_id=[[INITTIAL_TASK]]
  // CHECK-SAME: parallel_id=[[LEAGUE_ID:[0-9]+]]
  // CHECK-SAME: requested_team_size=2
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_distribute_begin:
  // CHECK-SAME: parallel_id=[[LEAGUE_ID]]
  // CHECK-SAME: count=2
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_parallel_begin:
  // CHECK-SAME: parallel_id=[[TEAM_ID1:[0-9]+]]
  // CHECK-SAME: requested_team_size=2
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_parallel_end:
  // CHECK-SAME: parallel_id=[[TEAM_ID1]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_distribute_end:
  // CHECK-SAME: parallel_id=[[LEAGUE_ID]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_parallel_end:
  // CHECK-SAME: parallel_id=[[LEAGUE_ID]]

  // CHECK: {{^}}[[WORKER_ID:[0-9]+]]: ompt_event_distribute_begin:
  // CHECK-SAME: parallel_id=[[LEAGUE_ID]]
  // CHECK-SAME: count=2
  // CHECK: {{^}}[[WORKER_ID]]: ompt_event_parallel_begin:
  // CHECK-SAME: parallel_id=[[TEAM_ID2:[0-9]+]], requested_team_size=2
  // CHECK: {{^}}[[WORKER_ID]]: ompt_event_parallel_end:
  // CHECK-SAME: parallel_id=[[TEAM_ID2]]
  // CHECK: {{^}}[[WORKER_ID]]: ompt_event_distribute_end:
  // CHECK-SAME: parallel_id=[[LEAGUE_ID]]

  return 0;
}