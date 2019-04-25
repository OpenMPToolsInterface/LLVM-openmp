/*
 * Callbacks.h
 *
 *  Created on: Dec 23, 2014
 *      Author: Ignacio Laguna
 *     Contact: ilaguna@llnl.gov
 */
#ifndef GDB_CALLBACKS_H_
#define GDB_CALLBACKS_H_

/******************************************************************************
 * This header file defines the callback functions that are provided by the
 * debugger to OMPD. In this case, we implement them using GDB. Other debuggers
 * will have different implementations.
 */

#include "omp-tools.h"
#include "GdbProcess.h"
#include "StringParser.h"
#include "CudaGdb.h"
#include <vector>
#include <map>
#include <string>

/******************************************************************************
 * Helper functions
 */
void initializeCallbacks(const ompd_gdb::GdbProcessPtr &proc);
ompd_callbacks_t * getCallbacksTable();
unsigned int getSizeOf(const char *str);
std::vector<ompd_gdb::StringParser::ThreadID> getThreadIDsFromDebugger();
uint64_t evalGdbExpression(std::string command);

std::map<int, uint64_t> getCudaContextIDsFromDebugger();
std::map<int, std::pair<int, int>> getCudaKernelIDsFromDebugger();
std::vector<CudaThread> getCudaKernelThreadsFromDebugger(uint64_t, uint64_t, uint64_t, uint64_t);

/******************************************************************************
 * Callbacks
 */

ompd_rc_t CB_alloc_memory (
    ompd_size_t bytes,
    void **ptr);

ompd_rc_t CB_free_memory (
    void *ptr);

ompd_rc_t CB_thread_context (
    ompd_address_space_context_t *context,
    ompd_thread_id_t          kind,
    ompd_size_t                   sizeof_osthread,
    const void*                   osthread,
    ompd_thread_context_t **tcontext);
                
ompd_rc_t CB_process_context (
    ompd_address_space_context_t* context,
    ompd_address_space_context_t** containing_process_context);

ompd_rc_t CB_sizeof_prim (
    ompd_address_space_context_t *context,
    ompd_device_type_sizes_t *sizes);

ompd_rc_t CB_symbol_addr (
    ompd_address_space_context_t *context,
    ompd_thread_context_t *tcontext,
    const char *symbol_name,
    ompd_address_t *symbol_addr,
    const char *file_name);

ompd_rc_t CB_read_memory (
    ompd_address_space_context_t *context,
    ompd_thread_context_t *tcontext,
    const ompd_address_t *addr,
    ompd_size_t nbytes,
    void *buffer
    );

ompd_rc_t CB_write_memory (
    ompd_address_space_context_t *context,
    ompd_thread_context_t *tcontext,
    const ompd_address_t *addr,
    ompd_size_t nbytes,
    const void *buffer
    );

ompd_rc_t CB_device_to_host (
    ompd_address_space_context_t *address_space_context, /* IN */
    const void *input,      /* IN */
    ompd_size_t unit_size,      /* IN */
    ompd_size_t count,      /* IN: number of primitive type */
                    /* items to process */
    void *output    /* OUT */
    );

ompd_rc_t CB_host_to_device (
    ompd_address_space_context_t *address_space_context, /* IN */
    const void *input,      /* IN */
    ompd_size_t unit_size,      /* IN */
    ompd_size_t count,      /* IN: number of primitive type */
                    /* items to process */
    void *output    /* OUT */
    );
    
ompd_rc_t CB_print_string (
    const char *string,
    int category
    );

#endif /* GDB_CALLBACKS_H_ */
