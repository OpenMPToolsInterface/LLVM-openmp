/*
 * Callbacks.cpp
 *
 *  Created on: Dec 23, 2014
 *      Author: Ignacio Laguna
 *     Contact: ilaguna@llnl.gov
 */

#include "Callbacks.h"
#include "OMPDContext.h"
#include "GdbProcess.h"
#include "CudaGdb.h"
#include <memory>
#include <map>
#include <vector>
#include <string>
#include <map>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <sstream>

using namespace ompd_gdb;
using namespace std;

static ompd_callbacks_t cb;
static GdbProcessPtr gdb(nullptr);
static StringParser parser;

unsigned int prim_sizes[ompd_type_max];

void init_sizes();

void initializeCallbacks(const GdbProcessPtr &proc)
{
  // Copy pointer of GDB process
  gdb = proc;

  // Initialize static table
  cb.alloc_memory       = CB_alloc_memory;
  cb.free_memory        = CB_free_memory;
  cb.print_string       = CB_print_string;
  cb.get_thread_context_for_thread_id = CB_thread_context;
  cb.sizeof_type        = CB_sizeof_prim;
  cb.symbol_addr_lookup = CB_symbol_addr;
  cb.read_memory        = CB_read_memory;
  cb.write_memory       = CB_write_memory;
  cb.read_string        = CB_read_memory;
  cb.host_to_device     = CB_host_to_device;
  cb.device_to_host     = CB_device_to_host;
}

ompd_callbacks_t * getCallbacksTable()
{
  return &cb;
}

ompd_rc_t CB_alloc_memory (
    ompd_size_t bytes,
    void **ptr)
{
  void *allocPtr = malloc(bytes);
  if (allocPtr != NULL)
    *ptr = allocPtr;
  else
    return ompd_rc_error;
  return ompd_rc_ok;
}

ompd_rc_t CB_free_memory(
    void *ptr)
{
  if (!ptr)
    return ompd_rc_error;
  free(ptr);
  return ompd_rc_ok;
}

ompd_rc_t CB_thread_context (
    ompd_address_space_context_t *context,
    ompd_thread_id_t             kind,
    ompd_size_t                   sizeof_osthread,
    const void*                   osthread,
    ompd_thread_context_t **tcontext
    )
{
  ompd_rc_t ret = context ? ompd_rc_ok : ompd_rc_stale_handle;
  if (kind == OMPD_THREAD_ID_CUDALOGICAL) {
    *tcontext = ((OMPDContext*)context)->getContextForThread((CudaThread*)osthread);
  }
  else {
    *tcontext = ((OMPDContext*)context)->getContextForThread((pthread_t*)osthread);
  }
  return ret;
}


void init_sizes(){
  prim_sizes[ompd_type_char]      = getSizeOf("char");
  prim_sizes[ompd_type_short]     = getSizeOf("short");
  prim_sizes[ompd_type_int]       = getSizeOf("int");
  prim_sizes[ompd_type_long]      = getSizeOf("long");
  prim_sizes[ompd_type_long_long] = getSizeOf("long long");
  prim_sizes[ompd_type_pointer]   = getSizeOf("void *");
}

ompd_rc_t CB_sizeof_prim(
    ompd_address_space_context_t *context,
    ompd_device_type_sizes_t *sizes)
{
  ompd_rc_t ret = context ? ompd_rc_ok : ompd_rc_stale_handle;
  static int inited = 0;
  if(!inited)
  {
    inited=1;
    init_sizes();
  }
  sizes->sizeof_char = prim_sizes[ompd_type_char];
  sizes->sizeof_short = prim_sizes[ompd_type_short];
  sizes->sizeof_int = prim_sizes[ompd_type_int];
  sizes->sizeof_long = prim_sizes[ompd_type_long];
  sizes->sizeof_long_long = prim_sizes[ompd_type_long_long];
  sizes->sizeof_pointer = prim_sizes[ompd_type_pointer];  

  return ret;
}

/* Returns zero if the type doesn't exist */
unsigned int getSizeOf(const char *str)
{
  assert(gdb.get() != nullptr && "Invalid GDB process!");
  string command("print sizeof(" + string(str) + ")");
  gdb->writeInput(command.c_str());
  char val[8];
  string gdbOut = gdb->readOutput();
  parser.matchRegularValue(gdbOut.c_str(), val);
  if (strlen(val) == 0) // type not found
    return 0;

  int intVal = atoi(val);
  return static_cast<unsigned int>(intVal);
}

ompd_rc_t CB_symbol_addr(
    ompd_address_space_context_t *context,
    ompd_thread_context_t *tcontext,
    const char *symbol_name,
    ompd_address_t *symbol_addr,
    const char *file_name)
{
  ompd_rc_t ret = context ? ompd_rc_ok : ompd_rc_stale_handle;
  assert(gdb.get() != nullptr && "Invalid GDB process!");

  if (tcontext)
    ((OMPDContext*)tcontext)->setThisGdbContext();

  string command("p &" + string(symbol_name));
  gdb->writeInput(command.c_str());
  char addr[64]; // long enough to hold an address
  addr[0] = '\0';
  parser.matchAddressValue(gdb->readOutput().c_str(), addr);

  if (strlen(addr) > 0)
    symbol_addr->address = (ompd_addr_t) strtoull (addr, NULL, 0);
  else if (strlen(addr) == 0)
    ret = ompd_rc_error;

  return ret;
}

map<int, uint64_t> getCudaContextIDsFromDebugger()
{
  string command("info cuda contexts");
  gdb->writeInput(command.c_str());
  string gdbOut = gdb->readOutput();
  return parser.matchCudaContextsInfo(gdbOut.c_str());
}

map<int, pair<int, int>> getCudaKernelIDsFromDebugger()
{
  string command("info cuda kernels");
  gdb->writeInput(command.c_str());
  string gdbOut = gdb->readOutput();
  return parser.matchCudaKernelsInfo(gdbOut.c_str());
}

vector<CudaThread> getCudaKernelThreadsFromDebugger(
        uint64_t ctx, uint64_t dev, uint64_t gid, uint64_t kernel
)
{
  vector<CudaThread> ret;

  gdb->writeInput("set cuda coalescing on");
  gdb->readOutput();

  stringstream command;
  command << "cuda kernel " << kernel;
  gdb->writeInput(command.str().c_str()); gdb->readOutput();

  gdb->writeInput("info cuda threads");
  string gdbOut = gdb->readOutput();
  ret = parser.matchCudaThreadsInfo(ctx, dev, kernel, gid, gdbOut.c_str());

  return ret;
}

/*
 * Run 'info threads' command in gdb and return vector thread IDs.
 * Returns a pair <int, system IDs>.
 */
vector<StringParser::ThreadID> getThreadIDsFromDebugger()
{
  string command("info threads");
  gdb->writeInput(command.c_str());
  string gdbOut = gdb->readOutput();
  return parser.matchThreadsInfo(gdbOut.c_str());
}

uint64_t evalGdbExpression(string command)
{
  char value[256];
  gdb->writeInput(command.c_str());
  string gdbOut = gdb->readOutput();
  parser.matchRegularValue(gdbOut.c_str(), value);
  return strtoll(value, NULL, 0);
}


template<typename T> 
inline void set_mem_strings(vector<string>& str, T* dest)
{
  for (size_t i=0; i < str.size(); ++i)
    dest[i]=(T)strtoll(str[i].c_str(), NULL, 0);
}

ompd_rc_t CB_write_memory (
    ompd_address_space_context_t *context,
    ompd_thread_context_t *tcontext,
    const ompd_address_t *addr,
    ompd_size_t nbytes,
    const void *buffer)
{
  return ompd_rc_unsupported;
}

ompd_rc_t CB_read_memory (
    ompd_address_space_context_t *context,
    ompd_thread_context_t *tcontext,
    const ompd_address_t *addr,
    ompd_size_t nbytes,
    void *buffer)
{
  if (!context)
    return ompd_rc_stale_handle;
  assert(gdb.get() != nullptr && "Invalid GDB process!");

  if (!(nbytes > 0))
    return ompd_rc_error;

  if (tcontext) {
    ((OMPDContext*)tcontext)->setThisGdbContext();
  }
  else {
    OMPDContext* ompc = (OMPDContext*)context;

    if (OMPDHostContext* host_c = dynamic_cast<OMPDHostContext *>(ompc)) {
      host_c->cp->getFirstThreadContext()->setThisGdbContext();
    }
  }

  // Get bytes of memory from gdb
  stringstream command;
  string cast;

  switch (addr->segment) {
      default:
          cast = "xb 0x"; break;
      case OMPD_SEGMENT_CUDA_PTX_GLOBAL:
          cast = "xb (@global unsigned char*) 0x"; break;
      case OMPD_SEGMENT_CUDA_PTX_LOCAL:
          cast = "xb (@local unsigned char*) 0x"; break;
      case OMPD_SEGMENT_CUDA_PTX_SHARED:
          cast = "xb (@shared unsigned char*) 0x"; break;
  }
  command << "x/" << nbytes << cast << std::hex << addr->address;

  gdb->writeInput(command.str().c_str());
  vector<string> words;
  string out = gdb->readOutput();
  words = parser.matchMemoryValues(out.c_str());
  assert((size_t)nbytes == words.size() && "Read more or less words from gdb");

  set_mem_strings(words,(uint8_t*)buffer);

  return ompd_rc_ok;
}

ompd_rc_t CB_device_to_host (
    ompd_address_space_context_t *address_space_context, /* IN */
    const void *input,          /* IN */
    ompd_size_t unit_size,              /* IN */
    ompd_size_t count,      /* IN: number of primitive type */
                    /* items to process */
    void *output    /* OUT */
)
{
  memmove(output, input, unit_size);
  return ompd_rc_ok;
}
    
ompd_rc_t CB_host_to_device (
    ompd_address_space_context_t *address_space_context, /* IN */
    const void *input,          /* IN */
    ompd_size_t unit_size,              /* IN */
    ompd_size_t count,      /* IN: number of primitive type */
                    /* items to process */
    void *output    /* OUT */
)
{
  memmove(output, input, unit_size);
  return ompd_rc_ok;
}
    

ompd_rc_t CB_print_string (
    const char *string,
    int category
    )
{
  printf("%s", string);
  return ompd_rc_ok;
}

