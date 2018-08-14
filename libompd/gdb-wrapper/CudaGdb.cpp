/*
 * CudaGdb.cpp
 *
 * Created on: Apr 11, 2017
 *     Author: Marty McFadden
 *    Contact: mmcfadden8@llnl.gov
 */
#include "CudaGdb.h"
#include "Callbacks.h"
#include <cstdint>
#include <iostream>
#include <vector>
#include <map>

using namespace std;

CudaGdb::CudaGdb()
{
  map<int, uint64_t> contexts = getCudaContextIDsFromDebugger();
  map<int, pair<int, int>> kernels = getCudaKernelIDsFromDebugger();

  if (contexts.size() == 0 || kernels.size() == 0) {
    cerr << "No CUDA Contexts(" << contexts.size() << ") or kernels(" << kernels.size() << "%d) present\n";
    return;
  }

  for (auto i : kernels) {
    int dev_id = i.second.first;
    int grid_id = i.second.second;
    int kernel_id = i.first;
    uint64_t ctx_id = contexts[dev_id];
    vector<CudaThread> t = getCudaKernelThreadsFromDebugger(ctx_id, dev_id, grid_id, kernel_id);

    threads.insert(threads.end(), t.begin(), t.end());
  }
}
