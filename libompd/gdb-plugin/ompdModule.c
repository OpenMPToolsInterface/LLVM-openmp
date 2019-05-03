#include <Python.h>
#include <omp-tools.h>
// #include <ompd.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <pthread.h>

static void* ompd_library;

#define OMPD_WEAK_ATTR __attribute__((weak))

struct _ompd_aspace_cont {int id;};
struct _ompd_thread_cont {int id;};
ompd_address_space_context_t acontext = {42};

PyObject* pModule;

ompd_rc_t _print(const char* str, int category);

// NOTE: start: implement functions to check arguments for correctness when compiling 
// for merging with latest OMPD version
OMPD_WEAK_ATTR ompd_rc_t ompd_get_api_version(ompd_word_t * addr) {
	static ompd_rc_t (*my_get_api_version)(ompd_word_t *) = NULL;
	if(!my_get_api_version){
		my_get_api_version = dlsym(ompd_library, "ompd_get_api_version");
		if (dlerror()) {
			return ompd_rc_error;
		}
	}
	return my_get_api_version(addr);
}

OMPD_WEAK_ATTR ompd_rc_t ompd_get_thread_handle(ompd_address_space_handle_t* handle, 
				      ompd_thread_id_t kind, ompd_size_t tidSize, 
				      const void* tid, ompd_thread_handle_t** threadHandle) {
	static ompd_rc_t (*my_get_thread_handle)(ompd_address_space_handle_t*, ompd_thread_id_t, ompd_size_t, const void*, ompd_thread_handle_t**) = NULL;
	if(!my_get_thread_handle) {
		my_get_thread_handle = dlsym(ompd_library, "ompd_get_thread_handle");
		if(dlerror()) {
			return ompd_rc_error;
		}
	}
	return my_get_thread_handle(handle, kind, tidSize, tid, threadHandle);
}

OMPD_WEAK_ATTR ompd_rc_t ompd_get_thread_in_parallel(ompd_parallel_handle_t* parallelHandle,
						     int threadNum,
						     ompd_thread_handle_t** threadHandle) {
	static ompd_rc_t (*my_get_thread_in_parallel)(ompd_parallel_handle_t*, int, ompd_thread_handle_t**) = NULL;
	if(!my_get_thread_in_parallel) {
		my_get_thread_in_parallel = dlsym(ompd_library, "ompd_get_thread_in_parallel");
		if(dlerror()) {
			return ompd_rc_error;
		}
	}
	return my_get_thread_in_parallel(parallelHandle, threadNum, threadHandle);
}

OMPD_WEAK_ATTR ompd_rc_t ompd_get_curr_parallel_handle(ompd_thread_handle_t* threadHandle,
							  ompd_parallel_handle_t** parallelHandle) {
	static ompd_rc_t (*my_get_current_parallel_handle)(ompd_thread_handle_t*, ompd_parallel_handle_t**) = NULL;
	if(!my_get_current_parallel_handle) {
		my_get_current_parallel_handle = dlsym(ompd_library, "ompd_get_curr_parallel_handle");
		if(dlerror()) {
			return ompd_rc_error;
		}
	}
	return my_get_current_parallel_handle(threadHandle, parallelHandle);
}

OMPD_WEAK_ATTR ompd_rc_t ompd_get_enclosing_parallel_handle(ompd_parallel_handle_t* parallelHandle,
							    ompd_parallel_handle_t** enclosing) {
	static ompd_rc_t (*my_get_enclosing_parallel_handle)(ompd_parallel_handle_t*, ompd_parallel_handle_t**) = NULL;
	if(!my_get_enclosing_parallel_handle) {
		my_get_enclosing_parallel_handle = dlsym(ompd_library, "ompd_get_enclosing_parallel_handle");
		if(dlerror()) {
			return ompd_rc_error;
		}
	}
	return my_get_enclosing_parallel_handle(parallelHandle, enclosing);
}

OMPD_WEAK_ATTR ompd_rc_t ompd_get_task_parallel_handle(ompd_task_handle_t* taskHandle,
						       ompd_parallel_handle_t** taskParallelHandle) {
	static ompd_rc_t (*my_get_task_parallel_handle)(ompd_task_handle_t*, ompd_parallel_handle_t**) = NULL;
	if(!my_get_task_parallel_handle) {
		my_get_task_parallel_handle = dlsym(ompd_library, "ompd_get_task_parallel_handle");
		if(dlerror()) {
			return ompd_rc_error;
		}
	}
	return my_get_task_parallel_handle(taskHandle, taskParallelHandle);
}

OMPD_WEAK_ATTR ompd_rc_t ompd_get_curr_task_handle(ompd_thread_handle_t* threadHandle,
						      ompd_task_handle_t** taskHandle) {
	static ompd_rc_t (*my_get_current_task_handle)(ompd_thread_handle_t*, ompd_task_handle_t**) = NULL;
	if(!my_get_current_task_handle) {
		my_get_current_task_handle = dlsym(ompd_library, "ompd_get_curr_task_handle");
		if(dlerror()) {
			return ompd_rc_error;
		}
	}
	return my_get_current_task_handle(threadHandle, taskHandle);
}

OMPD_WEAK_ATTR ompd_rc_t ompd_get_generating_task_handle(ompd_task_handle_t* taskHandle,
							 ompd_task_handle_t** generating) {
	static ompd_rc_t (*my_get_generating_task_handle)(ompd_task_handle_t*, ompd_task_handle_t**) = NULL;
	if(!my_get_generating_task_handle) {
		my_get_generating_task_handle = dlsym(ompd_library, "ompd_get_generating_task_handle");
		if(dlerror()) {
			return ompd_rc_error;
		}
	}
	return my_get_generating_task_handle(taskHandle, generating);
}

OMPD_WEAK_ATTR ompd_rc_t ompd_get_scheduling_task_handle(ompd_task_handle_t* taskHandle,
							 ompd_task_handle_t** scheduling) {
	static ompd_rc_t (*my_get_scheduling_task_handle)(ompd_task_handle_t*, ompd_task_handle_t**) =  NULL;
	if(!my_get_scheduling_task_handle) {
		my_get_scheduling_task_handle = dlsym(ompd_library, "ompd_get_scheduling_task_handle");
		if(dlerror()) {
			return ompd_rc_error;
		}
	}
	return my_get_scheduling_task_handle(taskHandle, scheduling);
}

OMPD_WEAK_ATTR ompd_rc_t ompd_get_task_in_parallel(ompd_parallel_handle_t* parallelHandle,
						   int threadNum,
						   ompd_task_handle_t** taskHandle) {
	static ompd_rc_t (*my_get_task_in_parallel)(ompd_parallel_handle_t*, int, ompd_task_handle_t**) = NULL;
	if(!my_get_task_in_parallel) {
		my_get_task_in_parallel = dlsym(ompd_library, "ompd_get_task_in_parallel");
		if(dlerror()) {
			return ompd_rc_error;
		}
	}
	return my_get_task_in_parallel(parallelHandle, threadNum, taskHandle);
}

OMPD_WEAK_ATTR ompd_rc_t ompd_get_task_frame(ompd_task_handle_t* taskHandle,
					     ompd_frame_info_t* exitFrame,
					     ompd_frame_info_t* enterFrame) {
	static ompd_rc_t (*my_get_task_frame)(ompd_task_handle_t*, ompd_frame_info_t*, ompd_frame_info_t*) = NULL;
	if(!my_get_task_frame) {
		my_get_task_frame = dlsym(ompd_library, "ompd_get_task_frame");
		if(dlerror()) {
			return ompd_rc_error;
		}
	}
	return my_get_task_frame(taskHandle, exitFrame, enterFrame);
}

OMPD_WEAK_ATTR ompd_rc_t ompd_get_icv_from_scope(void* handle,
						 ompd_scope_t scope,
						 ompd_icv_id_t icvId,
						 ompd_word_t* icvValue) {
	static ompd_rc_t (*my_get_icv_from_scope)(void*, ompd_scope_t, ompd_icv_id_t, ompd_word_t*) = NULL;
	if(!my_get_icv_from_scope) {
		my_get_icv_from_scope = dlsym(ompd_library, "ompd_get_icv_from_scope");
		if(dlerror()) {
			return ompd_rc_error;
		}
	}
	return my_get_icv_from_scope(handle, scope, icvId, icvValue);
}

OMPD_WEAK_ATTR ompd_rc_t ompd_enumerate_icvs(ompd_address_space_handle_t* handle,
					     ompd_icv_id_t current,
					     ompd_icv_id_t* next,
					     const char** nextIcvName,
					     ompd_scope_t* nextScope,
					     int* more) {
	static ompd_rc_t (*my_enumerate_icvs)(ompd_address_space_handle_t*, ompd_icv_id_t, ompd_icv_id_t*, const char**, ompd_scope_t*, int*) = NULL;
	if(!my_enumerate_icvs) {
		my_enumerate_icvs = dlsym(ompd_library, "ompd_enumerate_icvs");
		if(dlerror()) {
			return ompd_rc_error;
		}
	}
	return my_enumerate_icvs(handle, current, next, nextIcvName, nextScope, more);
}

OMPD_WEAK_ATTR ompd_rc_t ompd_enumerate_states(ompd_address_space_handle_t* addrSpaceHandle,
					       ompd_word_t currentState,
					       ompd_word_t* nextState,
					       const char** nextStateName,
					       ompd_word_t* moreEnums) {
	static ompd_rc_t (*my_enumerate_states)(ompd_address_space_handle_t*, ompd_word_t, ompd_word_t*, const char**, ompd_word_t*) = NULL;
	if(!my_enumerate_states) {
		my_enumerate_states = dlsym(ompd_library, "ompd_enumerate_states");
		if(dlerror()) {
			return ompd_rc_error;
		}
	}
	return my_enumerate_states(addrSpaceHandle, currentState, nextState, nextStateName, moreEnums);
}

OMPD_WEAK_ATTR ompd_rc_t ompd_get_state(ompd_thread_handle_t* threadHandle,
					ompd_word_t* state,
					ompd_wait_id_t* waitId) {
	static ompd_rc_t (*my_get_state)(ompd_thread_handle_t*, ompd_word_t*, ompd_wait_id_t*) = NULL;
	if(!my_get_state) {
		my_get_state = dlsym(ompd_library, "ompd_get_state");
		if(dlerror()) {
			return ompd_rc_error;
		}
	}
	return my_get_state(threadHandle, state, waitId);
}

OMPD_WEAK_ATTR ompd_rc_t ompd_get_task_function(ompd_task_handle_t* taskHandle,
				      ompd_address_t* entryPoint) {
	static ompd_rc_t (*my_get_task_function)(ompd_task_handle_t*, ompd_address_t*) = NULL;
	if(!my_get_task_function) {
		my_get_task_function = dlsym(ompd_library, "ompd_get_task_function");
		if(dlerror()) {
			return ompd_rc_error;
		}
	}
	return my_get_task_function(taskHandle, entryPoint);
}

OMPD_WEAK_ATTR ompd_rc_t ompd_get_thread_id(ompd_thread_handle_t* threadHandle,
					    ompd_thread_id_t kind,
					    ompd_size_t tidSize,
					    void* tid) {
	static ompd_rc_t (*my_get_thread_id)(ompd_thread_handle_t*, ompd_thread_id_t, ompd_size_t, void*) = NULL;
	if(!my_get_thread_id) {
		my_get_thread_id = dlsym(ompd_library, "ompd_get_thread_id");
		if(dlerror()) {
			return ompd_rc_error;
		}
	}
	return my_get_thread_id(threadHandle, kind, tidSize, tid);
}

OMPD_WEAK_ATTR ompd_rc_t ompd_get_tool_data(void* handle,
					    ompd_scope_t scope,
					    ompd_word_t* value,
					    ompd_address_t* ptr) {
	static ompd_rc_t (*my_get_tool_data)(void*, ompd_scope_t, ompd_word_t*, ompd_address_t*) = NULL;
	if(!my_get_tool_data) {
		my_get_tool_data = dlsym(ompd_library, "ompd_get_tool_data");
		if(dlerror()) {
			return ompd_rc_error;
		}
	}
	return my_get_tool_data(handle, scope, value, ptr);
}

OMPD_WEAK_ATTR ompd_rc_t ompd_get_icv_string_from_scope(void* handle,
							ompd_scope_t scope,
							ompd_icv_id_t icvId,
							const char** icvString) {
	static ompd_rc_t (*my_get_icv_string_from_scope)(void*, ompd_scope_t, ompd_icv_id_t, const char**) = NULL;
	if(!my_get_icv_string_from_scope) {
		my_get_icv_string_from_scope = dlsym(ompd_library, "ompd_get_icv_string_from_scope");
		if(dlerror()) {
			return ompd_rc_error;
		}
	}
	return my_get_icv_string_from_scope(handle, scope, icvId, icvString);
}

OMPD_WEAK_ATTR ompd_rc_t ompd_rel_thread_handle(ompd_thread_handle_t* threadHandle) {
	static ompd_rc_t (*my_release_thread_handle)(ompd_thread_handle_t *) = NULL;
	if (!my_release_thread_handle) {
		my_release_thread_handle = dlsym(ompd_library, "ompd_rel_thread_handle");
		if (dlerror()) {
			return ompd_rc_error;
		}
	}
	return my_release_thread_handle(threadHandle);
}

OMPD_WEAK_ATTR ompd_rc_t ompd_rel_parallel_handle(ompd_parallel_handle_t* parallelHandle) {
	static ompd_rc_t (*my_release_parallel_handle)(ompd_parallel_handle_t*) = NULL;
	if (!my_release_parallel_handle) {
		my_release_parallel_handle = dlsym(ompd_library, "ompd_rel_parallel_handle");
		if (dlerror()) {
			return ompd_rc_error;
		}
	}
	return my_release_parallel_handle(parallelHandle);
}

OMPD_WEAK_ATTR ompd_rc_t ompd_rel_task_handle(ompd_task_handle_t* taskHandle) {
	static ompd_rc_t (*my_release_task_handle)(ompd_task_handle_t*) = NULL;
	if(!my_release_task_handle) {
		my_release_task_handle = dlsym(ompd_library, "ompd_rel_task_handle");
		if(dlerror()) {
			return ompd_rc_error;
		}
	}
	return my_release_task_handle(taskHandle);
}
// end: dlsym functions to check OMPD function arguments for correctness with new OMPD version


/**
 * Loads the OMPD library (libompd.so). Returns an integer with the version if the OMPD
 * library could be loaded successfully.
 * Error codes:
 * -1: argument could not be converted to string
 * -2: error when calling dlopen
 * -3: error when fetching version of OMPD API
 * else: see ompd return codes
 */
static PyObject* ompd_open(PyObject* self, PyObject* args)
{
	const char* name, *dlerr;
	dlerror();
	if(!PyArg_ParseTuple(args, "s", &name))
	{
		return Py_BuildValue("i", -1);
	}
	ompd_library = dlopen(name, RTLD_LAZY);
	if ((dlerr = dlerror())){
		_print(dlerr, 0);
		return Py_BuildValue("i", -2);
	}
	//ompd_rc_t (*my_get_api_version)(ompd_word_t *)  = dlsym(ompd_library, "ompd_get_api_version");
	if (dlerror()) {
		return Py_BuildValue("i", -3);
	}
	ompd_word_t version;
	//ompd_rc_t rc = my_get_api_version(&version);
        // NOTE: call dlsym function for OMPD API call to check args for correctness
	ompd_rc_t rc = ompd_get_api_version(&version);
	if (rc != ompd_rc_ok)
		return Py_BuildValue("i", -10-rc);
	
	int returnValue = version;
	return Py_BuildValue("i", returnValue);
}

/**
 * Have the debugger print a string.
 */
ompd_rc_t _print(
	const char* str,
	int category)
{
	PyObject* pFunc = PyObject_GetAttrString(pModule, "_print");
	if(pFunc && PyCallable_Check(pFunc)) {
		PyObject* pArgs = PyTuple_New(1);
		PyTuple_SetItem(pArgs, 0, Py_BuildValue("s", str));
		PyObject_CallObject(pFunc, pArgs);
		Py_XDECREF(pArgs);
	}
	Py_XDECREF(pFunc);
	return ompd_rc_ok;
}

void _printf(
	char* format,
	...)
{
	va_list args;
	va_start(args, format);
	char output[1024];
	vsnprintf(output, 1024, format, args);
	va_end(args);
	_print(output, 0);
}

/**
 * Capsule destructors for thread, parallel and task handles.
 */
static void call_ompd_rel_thread_handle_temp(PyObject* capsule)
{
	ompd_thread_handle_t* threadHandle = (ompd_thread_handle_t*)(PyCapsule_GetPointer(capsule, "ThreadHandle"));
	// NOTE: call dlsym function
	ompd_rc_t retVal = ompd_rel_thread_handle(threadHandle);
	if(retVal != ompd_rc_ok) {
		_printf("An error occurred when calling ompd_rel_thread_handle! Error code: %d", retVal);
	}
}

static void destroyThreadCapsule(PyObject* capsule) {
	call_ompd_rel_thread_handle_temp(capsule);
}
static void (*my_thread_capsule_destructor)(PyObject*) = destroyThreadCapsule;

static void call_ompd_rel_parallel_handle_temp(PyObject* capsule) {
	ompd_parallel_handle_t* parallelHandle = (ompd_parallel_handle_t*)(PyCapsule_GetPointer(capsule, "ParallelHandle"));
	// NOTE: call dlsym function
	ompd_rc_t retVal = ompd_rel_parallel_handle(parallelHandle);
	if(retVal != ompd_rc_ok) {
		_printf("An error occurred when calling ompd_rel_parallel_handle! Error code: %d", retVal);
	}
}

static void destroyParallelCapsule(PyObject* capsule) {
	call_ompd_rel_parallel_handle_temp(capsule);
}
static void (*my_parallel_capsule_destructor)(PyObject*) = destroyParallelCapsule;

static void call_ompd_rel_task_handle_temp(PyObject* capsule) {
	ompd_task_handle_t* taskHandle = (ompd_task_handle_t*)(PyCapsule_GetPointer(capsule, "TaskHandle"));
	// NOTE: call dlsym
	ompd_rc_t retVal = ompd_rel_task_handle(taskHandle);
	if(retVal != ompd_rc_ok) {
		_printf("An error occurred when calling ompd_rel_task_handle!\n");
	}
}

static void destroyTaskCapsule(PyObject* capsule) {
	call_ompd_rel_task_handle_temp(capsule);
}
static void (*my_task_capsule_destructor)(PyObject*) = destroyTaskCapsule;

/**
 * Release thread handle. Called inside destructor for Python thread_handle object.
 */
static PyObject* call_ompd_rel_thread_handle(PyObject* self, PyObject* args)
{
	PyObject* threadHandlePy = PyTuple_GetItem(args, 0);
	ompd_thread_handle_t* threadHandle = (ompd_thread_handle_t*)(PyCapsule_GetPointer(threadHandlePy, "ThreadHandle"));
	
	// NOTE: call dlsym function
	ompd_rc_t retVal = ompd_rel_thread_handle(threadHandle);
	if(retVal != ompd_rc_ok) {
		_printf("An error occurred when calling ompd_rel_thread_handle! Error code: %d", retVal);
	}
	return Py_BuildValue("i", retVal);
}

/**
 * Allocate memory in the debugger's address space.
 */
ompd_rc_t _alloc(
	ompd_size_t bytes,
	void** ptr)
{
	if(ptr == NULL) {
		return ompd_rc_bad_input;
	}
	*ptr = malloc(bytes);
	return ompd_rc_ok;
}

/**
 * Free memory in the debugger's address space.
 */
ompd_rc_t _free(
	void* ptr)
{
	free(ptr);
	return ompd_rc_ok;
}

/**
 * Look up the sizes of primitive types in the target.
 */
ompd_rc_t _sizes(
	ompd_address_space_context_t * _acontext,	/* IN */
	ompd_device_type_sizes_t * sizes)		/* OUT */
{
	if (acontext.id != _acontext->id)
		return ompd_rc_stale_handle;
	ompd_device_type_sizes_t mysizes = {
		(uint8_t)sizeof(char),
		(uint8_t)sizeof(short),
		(uint8_t)sizeof(int),
		(uint8_t)sizeof(long),
		(uint8_t)sizeof(long long),
		(uint8_t)sizeof(void*)
	};
	*sizes = mysizes;
	return ompd_rc_ok;
}

/**
 * Look up the address of a global symbol in the target.
 */
ompd_rc_t _sym_addr(
	ompd_address_space_context_t *context,		/* IN */
	ompd_thread_context_t *tcontext,		/* IN */
	const char* symbol_name,			/* IN */
	ompd_address_t *symbol_addr,			/* OUT */
	const char* file_name)				/* IN */
{
	int thread_id = -1;
	PyObject* symbolAddress;
	if(tcontext != NULL) {
		thread_id = tcontext->id;
	}
	PyObject* pFunc = PyObject_GetAttrString(pModule, "_sym_addr");
	if(pFunc && PyCallable_Check(pFunc)) {
		PyObject* pArgs = PyTuple_New(2);
		PyTuple_SetItem(pArgs, 0, Py_BuildValue("i", thread_id));
		PyTuple_SetItem(pArgs, 1, Py_BuildValue("s", symbol_name));
		symbolAddress = PyObject_CallObject(pFunc, pArgs);
		symbol_addr->address = PyInt_AsLong(symbolAddress);
		Py_XDECREF(pArgs);
		Py_XDECREF(symbolAddress);
	}
	Py_XDECREF(pFunc);
	return ompd_rc_ok;
}

/**
 * Read memory from the target.
 */
ompd_rc_t _read (
	ompd_address_space_context_t *context,		/* IN */
	ompd_thread_context_t *tcontext,		/* IN */
	const ompd_address_t *addr,			/* IN */
	ompd_size_t nbytes,				/* IN */
	void* buffer)					/* OUT */
{	
	uint64_t readMem = (uint64_t) addr->address;
	PyObject* pFunc = PyObject_GetAttrString(pModule, "_read");
	if(pFunc && PyCallable_Check(pFunc)) {
		PyObject* pArgs = PyTuple_New(2);
		PyTuple_SetItem(pArgs, 0, Py_BuildValue("l", readMem));
		PyTuple_SetItem(pArgs, 1, Py_BuildValue("i", nbytes));
		PyObject* retArray = PyObject_CallObject(pFunc, pArgs);
		Py_XDECREF(pArgs);
		if(!PyByteArray_Check(retArray)) {
			return ompd_rc_error;
		}
		int retSize = (int) PyByteArray_Size((PyObject*) retArray);
		if(retSize != nbytes) {
			return ompd_rc_error;
		}
		memcpy(buffer, (void*) PyByteArray_AsString(retArray), nbytes);
		Py_XDECREF(retArray);
	}
	Py_XDECREF(pFunc);
	return ompd_rc_ok;
}

/**
 * Reads string from target.
 */
ompd_rc_t _read_string (
	ompd_address_space_context_t *context,		/* IN */
	ompd_thread_context_t *tcontext,		/* IN */
	const ompd_address_t *addr,			/* IN */
	ompd_size_t nbytes,				/* IN */
	void* buffer)					/* OUT */
{
	uint64_t readMem = (uint64_t) addr->address;
	PyObject* pFunc = PyObject_GetAttrString(pModule, "_read_string");
	PyObject* pArgs = PyTuple_New(2);
	PyTuple_SetItem(pArgs, 0, Py_BuildValue("l", readMem));
	PyTuple_SetItem(pArgs, 1, Py_BuildValue("i", nbytes));
	PyObject* retString = PyObject_CallObject(pFunc, pArgs);
	Py_XDECREF(pArgs);
	if(!PyString_Check(retString)) {
		return ompd_rc_error;
	}
	int retSize = (int) PyString_Size(retString);
	if(retSize != nbytes) {
		return ompd_rc_error;
	}
	char* strbuffer = (char*) buffer;
 	strncpy(strbuffer, PyString_AsString(retString), nbytes);
	strbuffer[nbytes-1]='\0';
	return ompd_rc_ok;
}

/**
 * Write memory from the target.
 */
ompd_rc_t _endianess(
	ompd_address_space_context_t *address_space_context, 	/* IN */
	const void *input,                                   	/* IN */
	ompd_size_t unit_size,					/* IN */
	ompd_size_t count,   					/* IN: number of primitive type */
	void *output)
{
	if (acontext.id != address_space_context->id)
		return ompd_rc_stale_handle;
	memmove(output,input,count*unit_size);
	return ompd_rc_ok;
}

/**
 * Returns thread context for thread id; helper function for _thread_context callback.
 */
ompd_thread_context_t* get_thread_context(int id)
{
	static ompd_thread_context_t* tc = NULL;
	static int size=0;
	int i;
	if(id<1)
		return NULL;
	if (tc == NULL)
	{
		size=16;
		tc = malloc(size*sizeof(ompd_thread_context_t));
		for (i=0; i<size; i++)
			tc[i].id = i+1;
	}
	if (id-1>=size)
	{
		size += 16;
		tc = realloc(tc, size*sizeof(ompd_thread_context_t));
		for (i=0; i<size; i++)
			tc[i].id = i+1;
	}
	return tc+id-1;
}

/**
 * Get thread specific context.
 */
ompd_rc_t _thread_context(
	ompd_address_space_context_t *context,		/* IN */
	ompd_thread_id_t kind,				/* IN, 0 for pthread, 1 for lwp */
	ompd_size_t sizeof_thread_id,			/* IN */
	const void *thread_id,				/* IN */
	ompd_thread_context_t **thread_context)		/* OUT */
{
	if (acontext.id != context->id)
		return ompd_rc_stale_handle;
	if (kind != 0 && kind != 1)
		return ompd_rc_unsupported;
	long int tid;
	if (sizeof(long int) >= 8 && sizeof_thread_id == 8)
		tid = *(uint64_t*)thread_id;
	else if (sizeof(long int) >= 4 && sizeof_thread_id == 4)
		tid = *(uint32_t*)thread_id;
	else if (sizeof(long int) >= 2 && sizeof_thread_id == 2)
		tid = *(uint16_t*)thread_id;
	else return ompd_rc_bad_input;
	PyObject* pFunc = PyObject_GetAttrString(pModule, "_thread_context");
	if(pFunc && PyCallable_Check(pFunc)) {
		PyObject* pArgs = PyTuple_New(2);
		PyTuple_SetItem(pArgs, 0, Py_BuildValue("i", kind));
		PyTuple_SetItem(pArgs, 1, Py_BuildValue("l", tid));
		PyObject* res = PyObject_CallObject(pFunc, pArgs);
		int resAsInt = (int) PyInt_AsLong(res);
		(*thread_context) = get_thread_context(resAsInt);
		Py_XDECREF(pArgs);
		Py_XDECREF(res);
		Py_XDECREF(pFunc);
		if(*thread_context == NULL)
			return ompd_rc_bad_input;
		return ompd_rc_ok;
	}
	Py_XDECREF(pFunc);
	return ompd_rc_error;
}

/**
 * Calls ompd_process_initialize; returns pointer to ompd_address_space_handle.
 */
static PyObject* call_ompd_initialize(PyObject* self, PyObject* noargs)
{
	pModule = PyImport_Import(PyString_FromString("ompd_callbacks"));
	
	static ompd_callbacks_t table = 
		{
			_alloc,
			_free,
			_print,
			_sizes, 
			_sym_addr,
			_read,
			NULL,
			_read_string,
			_endianess,
			_endianess,
			_thread_context
		};
	
	ompd_rc_t (*my_ompd_init)(ompd_word_t version, ompd_callbacks_t*) = dlsym(ompd_library, "ompd_initialize");
	ompd_rc_t returnInit = my_ompd_init(42, &table);
	if(returnInit != ompd_rc_ok) {
		_printf("An error occurred when calling ompd_initialize! Error code: %d", returnInit);
	}
	ompd_address_space_handle_t* addr_space = NULL;
	ompd_rc_t (*my_proc_init)(ompd_address_space_context_t*, ompd_address_space_handle_t**)  = dlsym(ompd_library, "ompd_process_initialize");
	ompd_rc_t retProcInit = my_proc_init(&acontext, &addr_space);
	if(retProcInit != ompd_rc_ok) {
		_printf("An error occurred when calling ompd_process_initialize! Error code: %d", retProcInit);
	}
	return PyCapsule_New(addr_space, "AddressSpace", NULL);
}

/**
 * Returns a PyCapsule pointer to thread handle for thread with the given id.
 */
static PyObject* get_thread_handle(PyObject* self, PyObject* args)
{
	PyObject* threadIdTup = PyTuple_GetItem(args, 0);
	uint64_t threadId = (uint64_t) PyLong_AsLong(threadIdTup);
	
	// NOTE: compiler does not know what thread handle looks like, so no memory 
	// is allocated automatically in the debugger's memory space
	char dummyBuffer[40];
	
	PyObject* addrSpaceTup = PyTuple_GetItem(args, 1);
	ompd_thread_handle_t* threadHandle;
	ompd_address_space_handle_t* addrSpace = (ompd_address_space_handle_t*) PyCapsule_GetPointer(addrSpaceTup, "AddressSpace");

	ompd_size_t sizeof_tid = (ompd_size_t) sizeof(uint64_t);
	
	// NOTE: call dlsym function; hard-coded 1 for LWPTID usage (OMPD_THREAD_ID_LWP)
	ompd_rc_t retVal = ompd_get_thread_handle(addrSpace, 1, sizeof_tid, &threadId, &threadHandle);
	//ompd_rc_t (*my_get_thread_handle)(ompd_address_space_handle_t*, ompd_thread_id_t, ompd_size_t, const void*, ompd_thread_handle_t**) = dlsym(ompd_library, "ompd_get_thread_handle");
	//ompd_rc_t retVal = my_get_thread_handle(addrSpace, 1, sizeof_tid, &threadId, &threadHandle);
	
	free(dummyBuffer);
	if(retVal == ompd_rc_unavailable) {
		return Py_None;
	} else if(retVal != ompd_rc_ok) {
		_printf("An error occured when calling ompd_get_thread_handle! Error code: %d", retVal);
		return Py_BuildValue("i", retVal);
	}
	return PyCapsule_New(threadHandle, "ThreadHandle", my_thread_capsule_destructor);
}

/**
 * Returns a PyCapsule pointer to a thread handle for a specific thread id in the current parallel context.
 */
static PyObject* call_ompd_get_thread_in_parallel(PyObject* self, PyObject* args)
{
	PyObject* parallelHandlePy = PyTuple_GetItem(args, 0);
	int threadNum = (int) PyLong_AsLong(PyTuple_GetItem(args, 1));
	ompd_parallel_handle_t* parallelHandle = (ompd_parallel_handle_t*)(PyCapsule_GetPointer(parallelHandlePy, "ParallelHandle"));
	ompd_thread_handle_t* threadHandle;
	
	// NOTE: call dlsym function
	ompd_rc_t retVal = ompd_get_thread_in_parallel(parallelHandle, threadNum, &threadHandle);
	//ompd_rc_t (*my_get_thread_in_parallel)(ompd_parallel_handle_t*, int, ompd_thread_handle_t**) = dlsym(ompd_library, "ompd_get_thread_in_parallel");
	//ompd_rc_t retVal = my_get_thread_in_parallel(parallelHandle, threadNum, &threadHandle);
	
	if(retVal != ompd_rc_ok) {
		_printf("An error occurred when calling ompd_get_thread_in_parallel! Error code: %d", retVal);
		return Py_BuildValue("i", retVal);
	}
	return PyCapsule_New(threadHandle, "ThreadHandle", my_thread_capsule_destructor);
}

/**
 * Returns a PyCapsule pointer to the parallel handle of the current parallel region associated with a thread.
 */
static PyObject* call_ompd_get_curr_parallel_handle(PyObject* self, PyObject* args)
{
	PyObject* threadHandlePy = PyTuple_GetItem(args, 0);
	ompd_thread_handle_t* threadHandle = (ompd_thread_handle_t*)(PyCapsule_GetPointer(threadHandlePy, "ThreadHandle"));
	ompd_parallel_handle_t* parallelHandle;
	
	// NOTE: call dlsym function
	ompd_rc_t retVal = ompd_get_curr_parallel_handle(threadHandle, &parallelHandle);
	//ompd_rc_t (*my_get_current_parallel_handle)(ompd_thread_handle_t*, ompd_parallel_handle_t**) = dlsym(ompd_library, "ompd_get_curr_parallel_handle");
	//ompd_rc_t retVal = my_get_current_parallel_handle(threadHandle, &parallelHandle);
	
	if(retVal != ompd_rc_ok) {
		_printf("An error occurred when calling ompd_get_curr_parallel_handle! Error code: %d", retVal);
		return Py_BuildValue("i", retVal);
	}
	return PyCapsule_New(parallelHandle, "ParallelHandle", my_parallel_capsule_destructor);
}

/**
 * Returns a PyCapsule pointer to the parallel  handle for the parallel region enclosing the parallel region specified by parallel_handle.
 */
static PyObject* call_ompd_get_enclosing_parallel_handle(PyObject* self, PyObject* args) {
	PyObject* parallelHandlePy = PyTuple_GetItem(args, 0);
	ompd_parallel_handle_t* parallelHandle = (ompd_parallel_handle_t*)(PyCapsule_GetPointer(parallelHandlePy, "ParallelHandle"));
	ompd_parallel_handle_t* enclosingParallelHandle;
	
	// NOTE: call dlsym function
	ompd_rc_t retVal = ompd_get_enclosing_parallel_handle(parallelHandle, &enclosingParallelHandle);
	//ompd_rc_t (*my_get_enclosing_parallel_handle)(ompd_parallel_handle_t*, ompd_parallel_handle_t**) = dlsym(ompd_library, "ompd_get_enclosing_parallel_handle");
	//ompd_rc_t retVal = my_get_enclosing_parallel_handle(parallelHandle, &enclosingParallelHandle);
	
	if(retVal != ompd_rc_ok) {
		_printf("An error occurred when calling ompd_get_enclosing_parallel_handle! Error code: %d", retVal);
		return Py_BuildValue("i", retVal);
	}
	return PyCapsule_New(enclosingParallelHandle, "ParallelHandle", my_parallel_capsule_destructor);
}

/**
 * Returns a PyCapsule pointer to the parallel handle for the parallel region enclosing the task specified.
 */
static PyObject* call_ompd_get_task_parallel_handle(PyObject* self, PyObject* args) {
	PyObject* taskHandlePy = PyTuple_GetItem(args, 0);
	ompd_task_handle_t* taskHandle = PyCapsule_GetPointer(taskHandlePy, "TaskHandle");
	ompd_parallel_handle_t* taskParallelHandle;
	
	// NOTE: call dlsym function
	ompd_rc_t retVal = ompd_get_task_parallel_handle(taskHandle, &taskParallelHandle);
	//ompd_rc_t (*my_get_task_parallel_handle)(ompd_task_handle_t*, ompd_parallel_handle_t**) = dlsym(ompd_library, "ompd_get_task_parallel_handle");
	//ompd_rc_t retVal = my_get_task_parallel_handle(taskHandle, &taskParallelHandle);
	
	if(retVal != ompd_rc_ok) {
		_printf("An error occurred when calling ompd_get_task_parallel_handle! Error code: %d");
		return Py_BuildValue("i", retVal);
	}
	return PyCapsule_New(taskParallelHandle, "ParallelHandle", my_parallel_capsule_destructor);
}

/**
 * Releases a parallel handle; is called in by the destructor of a Python parallel_handle object.
 */
static PyObject* call_ompd_rel_parallel_handle(PyObject* self, PyObject* args) {
	PyObject* parallelHandlePy = PyTuple_GetItem(args, 0);
	ompd_parallel_handle_t* parallelHandle = (ompd_parallel_handle_t*)(PyCapsule_GetPointer(parallelHandlePy, "ParallelHandle"));
	
	// NOTE: call dlsym function
	ompd_rc_t retVal = ompd_rel_parallel_handle(parallelHandle);
	if(retVal != ompd_rc_ok) {
		_printf("An error occurred when calling ompd_rel_parallel_handle! Error code: %d", retVal);
	}
	return Py_BuildValue("i", retVal);
}

/**
 * Returns a PyCapsule pointer to the task handle of the current task region associated with a thread.
 */
static PyObject* call_ompd_get_curr_task_handle(PyObject* self, PyObject* args) {
	PyObject* threadHandlePy = PyTuple_GetItem(args, 0);
	ompd_thread_handle_t* threadHandle = (ompd_thread_handle_t*)(PyCapsule_GetPointer(threadHandlePy, "ThreadHandle"));
	ompd_task_handle_t* taskHandle;
	
	// NOTE: call dlsym function
	ompd_rc_t retVal = ompd_get_curr_task_handle(threadHandle, &taskHandle);
	//ompd_rc_t (*my_get_current_task_handle)(ompd_thread_handle_t*, ompd_task_handle_t**) = dlsym(ompd_library, "ompd_get_curr_task_handle");
	//ompd_rc_t retVal = my_get_current_task_handle(threadHandle, &taskHandle);
	
	if(retVal != ompd_rc_ok) {
		_printf("An error occurred when calling ompd_get_curr_task_handle! Error code: %d", retVal);
		return Py_BuildValue("i", retVal);
	}
	return PyCapsule_New(taskHandle, "TaskHandle", my_task_capsule_destructor);
}

/**
 * Returns a task handle for the task that created the task specified.
 */
static PyObject* call_ompd_get_generating_task_handle(PyObject* self, PyObject* args) {
	PyObject* taskHandlePy = PyTuple_GetItem(args, 0);
	ompd_task_handle_t* taskHandle = (ompd_task_handle_t*)(PyCapsule_GetPointer(taskHandlePy, "TaskHandle"));
	ompd_task_handle_t* generatingTaskHandle;
	
	// NOTE: call dlsym function
	ompd_rc_t retVal = ompd_get_generating_task_handle(taskHandle, &generatingTaskHandle);
	//ompd_rc_t (*my_get_generating_task_handle)(ompd_task_handle_t*, ompd_task_handle_t**) = dlsym(ompd_library, "ompd_get_generating_task_handle");
	//ompd_rc_t retVal = my_get_generating_task_handle(taskHandle, &generatingTaskHandle);
	
	if(retVal != ompd_rc_ok) {
		_printf("An error occurred when calling ompd_get_generating_task_handle! Error code: %d", retVal);
		return Py_BuildValue("i", retVal);
	}
	return PyCapsule_New(generatingTaskHandle, "TaskHandle", my_task_capsule_destructor);
}

/**
 * Returns the task handle for the task that scheduled the task specified.
 */
static PyObject* call_ompd_get_scheduling_task_handle(PyObject* self, PyObject* args) {
	PyObject* taskHandlePy = PyTuple_GetItem(args, 0);
	ompd_task_handle_t* taskHandle = (ompd_task_handle_t*)(PyCapsule_GetPointer(taskHandlePy, "TaskHandle"));
	ompd_task_handle_t* schedulingTaskHandle;
	
	// NOTE: call dlsym function
	ompd_rc_t retVal = ompd_get_scheduling_task_handle(taskHandle, &schedulingTaskHandle);
	//ompd_rc_t (*my_get_scheduling_task_handle)(ompd_task_handle_t*, ompd_task_handle_t**) = dlsym(ompd_library, "ompd_get_scheduling_task_handle");
	//ompd_rc_t retVal = my_get_scheduling_task_handle(taskHandle, &schedulingTaskHandle);
	
	if(retVal == ompd_rc_unavailable) {
		return Py_None;
	} else if(retVal != ompd_rc_ok) {
		_printf("An error occurred when calling ompd_get_scheduling_task_handle! Error code: %d", retVal);
		return Py_BuildValue("i", retVal);
	}
	return PyCapsule_New(schedulingTaskHandle, "TaskHandle", my_task_capsule_destructor);
}

/**
 * Returns task handles for the implicit tasks associated with a parallel region.
 */
static PyObject* call_ompd_get_task_in_parallel(PyObject* self, PyObject* args) {
	PyObject* parallelHandlePy = PyTuple_GetItem(args, 0);
	int threadNum = (int) PyLong_AsLong(PyTuple_GetItem(args, 1));
	ompd_parallel_handle_t* parallelHandle = (ompd_parallel_handle_t*)(PyCapsule_GetPointer(parallelHandlePy, "ParallelHandle"));
	ompd_task_handle_t* taskHandle;
	
	// NOTE: call dlsym function
	ompd_rc_t retVal = ompd_get_task_in_parallel(parallelHandle, threadNum, &taskHandle);
	//ompd_rc_t (*my_get_task_in_parallel)(ompd_parallel_handle_t*, int, ompd_task_handle_t**) = dlsym(ompd_library, "ompd_get_task_in_parallel");
	//ompd_rc_t retVal = my_get_task_in_parallel(parallelHandle, threadNum, &taskHandle);
	
	if(retVal != ompd_rc_ok) {
		_printf("An error occurred when calling ompd_get_task_in_parallel! Error code: %d", retVal);
		return Py_BuildValue("i", retVal);
	}
	return PyCapsule_New(taskHandle, "TaskHandle", my_task_capsule_destructor);
}

/**
 * Releases a task handle; is called by the destructor of a Python task_handle object.
 */
static PyObject* call_ompd_rel_task_handle(PyObject* self, PyObject* args) {
	PyObject* taskHandlePy = PyTuple_GetItem(args, 0);
	ompd_task_handle_t* taskHandle = (ompd_task_handle_t*)(PyCapsule_GetPointer(taskHandlePy, "TaskHandle"));
	
	// NOTE: call dlsym function
	ompd_rc_t retVal = ompd_rel_task_handle(taskHandle);
	if(retVal != ompd_rc_ok) {
		_printf("An error occurred when calling ompd_rel_task_handle! Error code: %d", retVal);
	}
	return Py_BuildValue("i", retVal);
}

/**
 * Calls ompd_get_task_frame and returns a PyCapsule for the enter frame of the given task.
 */
static PyObject* call_ompd_get_task_frame(PyObject* self, PyObject* args) {
	PyObject* taskHandlePy = PyTuple_GetItem(args, 0);
	ompd_task_handle_t* taskHandle = (ompd_task_handle_t*) PyCapsule_GetPointer(taskHandlePy, "TaskHandle");
	ompd_frame_info_t exitFrameInfo;
	ompd_frame_info_t enterFrameInfo;
	
	// NOTE: call dlsym function
	ompd_rc_t retVal = ompd_get_task_frame(taskHandle, &exitFrameInfo, &enterFrameInfo);
	//ompd_rc_t (*my_get_task_frame)(ompd_task_handle_t*, ompd_address_t*, ompd_address_t*) = dlsym(ompd_library, "ompd_get_task_frame");
	//ompd_rc_t retVal = my_get_task_frame(taskHandle, &exitFrame, &enterFrame);
	
	if(retVal != ompd_rc_ok) {
		_printf("An error occurred when calling ompd_get_task_frame! Error code: %d", retVal);
		return Py_BuildValue("i", retVal);
	}
	
	PyObject* result = PyTuple_New(4);
	PyTuple_SetItem(result, 0, PyLong_FromUnsignedLong(enterFrameInfo.frame_address.address));
	PyTuple_SetItem(result, 1, PyLong_FromUnsignedLong(enterFrameInfo.frame_flag));
	PyTuple_SetItem(result, 2, PyLong_FromUnsignedLong(exitFrameInfo.frame_address.address));
	PyTuple_SetItem(result, 3, PyLong_FromUnsignedLong(exitFrameInfo.frame_flag));
	return result;
}

/**
 * Calls ompd_get_icv_from_scope.
 */
static PyObject* call_ompd_get_icv_from_scope(PyObject* self, PyObject* args) {
	PyObject* addrSpaceHandlePy = PyTuple_GetItem(args, 0);
	PyObject* scopePy = PyTuple_GetItem(args, 1);
	PyObject* icvIdPy = PyTuple_GetItem(args, 2);
	
	ompd_scope_t scope = (ompd_scope_t) PyLong_AsLong(scopePy);
	ompd_address_space_handle_t* addrSpaceHandle;
	switch (scope) {
		case ompd_scope_thread:
			addrSpaceHandle = (ompd_address_space_handle_t*) PyCapsule_GetPointer(addrSpaceHandlePy, "ThreadHandle");
			break;
		case ompd_scope_parallel:
			addrSpaceHandle = (ompd_address_space_handle_t*) PyCapsule_GetPointer(addrSpaceHandlePy, "ParallelHandle");
			break;
		case ompd_scope_implicit_task:
			addrSpaceHandle = (ompd_address_space_handle_t*) PyCapsule_GetPointer(addrSpaceHandlePy, "TaskHandle");
			break;
		case ompd_scope_task: 
			addrSpaceHandle = (ompd_address_space_handle_t*) PyCapsule_GetPointer(addrSpaceHandlePy, "TaskHandle");
			break;
		default:
			addrSpaceHandle = (ompd_address_space_handle_t*) PyCapsule_GetPointer(addrSpaceHandlePy, "AddressSpace");
			break;
	}
	
	ompd_icv_id_t icvId = (ompd_icv_id_t) PyLong_AsLong(icvIdPy);
	ompd_word_t icvValue;
	
	// NOTE: call dlsym function
	ompd_rc_t retVal = ompd_get_icv_from_scope(addrSpaceHandle, scope, icvId, &icvValue);
	//ompd_rc_t (*my_get_icv_from_scope)(void*, ompd_scope_t, ompd_icv_id_t, ompd_word_t*) = dlsym(ompd_library, "ompd_get_icv_from_scope");
	//ompd_rc_t retVal = my_get_icv_from_scope(addrSpaceHandle, scope, icvId, &icvValue);
	
	if(retVal != ompd_rc_ok) {
		_printf("An error occurred when calling ompd_get_icv_from_scope: Error code: %d", retVal);
		return Py_None;
	}
	return PyLong_FromLong(icvValue);
}

/**
 * Calls ompd_enumerate_icvs.
 */
static PyObject* call_ompd_enumerate_icvs(PyObject* self, PyObject* args) {
	PyObject* addrSpaceHandlePy = PyTuple_GetItem(args, 0);
	PyObject* currentPy = PyTuple_GetItem(args, 1);
	ompd_icv_id_t current = (ompd_icv_id_t) (PyLong_AsLong(currentPy));
	ompd_address_space_handle_t* addrSpaceHandle = (ompd_address_space_handle_t*) PyCapsule_GetPointer(addrSpaceHandlePy, "AddressSpace");
	
	const char* nextIcv;
	ompd_scope_t nextScope;
	int more;
	ompd_icv_id_t nextId;
	
	// NOTE: call dlsym function
	ompd_rc_t retVal = ompd_enumerate_icvs(addrSpaceHandle, current, &nextId, &nextIcv, &nextScope, &more);
	//ompd_rc_t (*my_enumerate_icvs)(ompd_address_space_handle_t*, ompd_icv_id_t, ompd_icv_id_t*, const char**, ompd_scope_t*, int*) = dlsym(ompd_library, "ompd_enumerate_icvs");
	//ompd_rc_t retVal = my_enumerate_icvs(addrSpaceHandle, current, &nextId, &nextIcv, &nextScope, &more);
	
	if(retVal != ompd_rc_ok) {
		_printf("An error occurred when calling ompd_enumerate_icvs! Error code: %d", retVal);
		return Py_None;
	}
	PyObject* retTuple = PyTuple_New(4);
	PyTuple_SetItem(retTuple, 0, PyLong_FromUnsignedLong(nextId));
	PyTuple_SetItem(retTuple, 1, PyString_FromString(nextIcv));
	PyTuple_SetItem(retTuple, 2, PyLong_FromUnsignedLong(nextScope));
	PyTuple_SetItem(retTuple, 3, PyLong_FromLong(more));
	return retTuple;
}

/**
 * Calls ompd_enumerate_states.
 */
static PyObject* call_ompd_enumerate_states(PyObject* self, PyObject* args) {
	PyObject* addrSpaceHandlePy = PyTuple_GetItem(args, 0);
	ompd_address_space_handle_t* addrSpaceHandle = (ompd_address_space_handle_t*) PyCapsule_GetPointer(addrSpaceHandlePy, "AddressSpace");
	ompd_word_t currentState = (ompd_word_t) PyLong_AsLong(PyTuple_GetItem(args, 1));
	
	ompd_word_t nextState;
	const char* nextStateName;
	ompd_word_t moreEnums;
	
	// NOTE: call dlsym function
	ompd_rc_t retVal = ompd_enumerate_states(addrSpaceHandle, currentState, &nextState, &nextStateName, &moreEnums);
	//ompd_rc_t (*my_enumerate_states)(ompd_address_space_handle_t*, ompd_word_t, ompd_word_t*, const char**, ompd_word_t*) = dlsym(ompd_library, "ompd_enumerate_states");
	//ompd_rc_t retVal = my_enumerate_states(addrSpaceHandle, currentState, &nextState, &nextStateName, &moreEnums);
	
	if(retVal != ompd_rc_ok) {
		_printf("An error occurred when calling ompd_enumerate_states! Error code: %d", retVal);
		return Py_None;
	}
	PyObject* retTuple = PyTuple_New(3);
	PyTuple_SetItem(retTuple, 0, PyLong_FromLong(nextState));
	PyTuple_SetItem(retTuple, 1, PyString_FromString(nextStateName));
	PyTuple_SetItem(retTuple, 2, PyLong_FromLong(moreEnums));
	return retTuple;
}

/**
 * Calls ompd_get_state.
 */
static PyObject* call_ompd_get_state(PyObject* self, PyObject* args) {
	PyObject* threadHandlePy = PyTuple_GetItem(args, 0);
	ompd_thread_handle_t* threadHandle = (ompd_thread_handle_t*) PyCapsule_GetPointer(threadHandlePy, "ThreadHandle");
	ompd_word_t state;
	ompd_wait_id_t waitId;
	
	// NOTE: call dlsym function
	ompd_rc_t retVal = ompd_get_state(threadHandle, &state, &waitId);
	//ompd_rc_t (*my_get_state)(ompd_thread_handle_t*, ompd_word_t*, ompd_wait_id_t*) = dlsym(ompd_library, "ompd_get_state");
	//ompd_rc_t retVal = my_get_state(threadHandle, &state, &waitId);
	
	if(retVal != ompd_rc_ok) {
		_printf("An error occurred when calling ompd_get_state! Error code: %d", retVal);
		return Py_None;
	}
	PyObject* retTuple = PyTuple_New(2);
	PyTuple_SetItem(retTuple, 0, PyLong_FromLong(state));
	PyTuple_SetItem(retTuple, 1, PyLong_FromUnsignedLong(waitId));
	return retTuple;
}

/**
 * Calls ompd_get_task_function and returns entry point of the code that corresponds to the code 
 * executed by the task.
 */
static PyObject* call_ompd_get_task_function(PyObject* self, PyObject* args) {
	PyObject* taskHandlePy = PyTuple_GetItem(args, 0);
	ompd_task_handle_t* taskHandle = (ompd_task_handle_t*) PyCapsule_GetPointer(taskHandlePy, "TaskHandle");	
	ompd_address_t entryPoint;
	
	// NOTE: call dlsym function
	ompd_rc_t retVal = ompd_get_task_function(taskHandle, &entryPoint);
	//ompd_rc_t (*my_get_task_function)(ompd_task_handle_t*, ompd_address_t*) = dlsym(ompd_library, "ompd_get_task_function");
	//ompd_rc_t retVal = my_get_task_function(taskHandle, &entryPoint);
	
	if(retVal != ompd_rc_ok) {
		_printf("An error occurred when calling ompd_get_task_function! Error code: %d", retVal);
		return Py_None;
	}
	return PyLong_FromLong((long) entryPoint.address);
}

/**
 * Prints pointer stored inside PyCapusle.
 */
static PyObject* print_capsule(PyObject* self, PyObject* args) {
	PyObject* capsule = PyTuple_GetItem(args, 0);
	PyObject* name = PyTuple_GetItem(args, 1);
	void* pointer = PyCapsule_GetPointer(capsule, PyString_AsString(name));
	_printf("Capsule pointer: %p", pointer);
	return Py_None;
}

/**
 * Calls ompd_get_thread_id for given handle and returns the thread id as a long.
 */
static PyObject* call_ompd_get_thread_id(PyObject* self, PyObject* args) {
	PyObject* threadHandlePy = PyTuple_GetItem(args, 0);
	ompd_thread_handle_t* threadHandle = (ompd_thread_handle_t*) (PyCapsule_GetPointer(threadHandlePy, "ThreadHandle"));
	ompd_thread_id_t kind = 0; // OMPD_THREAD_ID_PTHREAD
	ompd_size_t sizeOfId = (ompd_size_t) sizeof(pthread_t);
	
	uint64_t thread;
	// NOTE: call dlsym function
	ompd_rc_t retVal = ompd_get_thread_id(threadHandle, kind, sizeOfId, &thread);
	//ompd_rc_t (*my_get_thread_id)(ompd_thread_handle_t*, ompd_thread_id_t, ompd_size_t, void*) = dlsym(ompd_library, "ompd_get_thread_id");
	//ompd_rc_t retVal = my_get_thread_id(threadHandle, kind, sizeOfId, &thread);
	
	if(retVal != ompd_rc_ok) {
		kind = 1; // OMPD_THREAD_ID_LWP
		// NOTE: call dlsym function
		retVal = ompd_get_thread_id(threadHandle, kind, sizeOfId, &thread);
		//retVal = my_get_thread_id(threadHandle, kind, sizeOfId, &thread);
		if(retVal != ompd_rc_ok) {
			_printf("An error occurred when calling ompd_get_thread_id! Error code: %d", retVal);
			return Py_None;
		}
	}
	return PyLong_FromLong(thread);
}

/**
 * Calls ompd_get_tool_data and returns a tuple containing the value and pointer of the 
 * ompt_data_t union for the selected scope.
 */
static PyObject* call_ompd_get_tool_data(PyObject* self, PyObject* args) {
	PyObject* scopePy = PyTuple_GetItem(args, 0);
	ompd_scope_t scope = (ompd_scope_t) (PyInt_AsLong(scopePy));
	PyObject* handlePy = PyTuple_GetItem(args, 1);
	void* handle = NULL;
	
	if (scope == 3) {
		ompd_thread_handle_t* threadHandle = (ompd_thread_handle_t*)(PyCapsule_GetPointer(handlePy, "ThreadHandle"));
		handle = threadHandle;
	} else if (scope == 4) {
		ompd_parallel_handle_t* parallelHandle = (ompd_parallel_handle_t*)(PyCapsule_GetPointer(handlePy, "ParallelHandle"));
		handle = parallelHandle;
	} else if (scope == 5 || scope == 6) {
		ompd_task_handle_t* taskHandle = (ompd_task_handle_t*)(PyCapsule_GetPointer(handlePy, "TaskHandle"));
		handle = taskHandle;
	} else {
		_printf("An error occured when calling ompd_get_tool_data! Scope type not supported.");
		return Py_None;
	}
	
	ompd_word_t value;
	ompd_address_t ptr;
	
	// NOTE: call dlsym function
	ompd_rc_t retVal = ompd_get_tool_data(handle, scope, &value, &ptr);
	//ompd_rc_t (*my_get_tool_data)(void*, ompd_scope_t, ompd_word_t*, ompd_address_t*) = dlsym(ompd_library, "ompd_get_tool_data");
	//ompd_rc_t retVal = my_get_tool_data(handle, scope, &value, &ptr);
	
	if(retVal != ompd_rc_ok) {
		_printf("An error occured when calling ompd_get_tool_data! Error code: %d", retVal);
		return Py_None;
	}
	
	PyObject* retTuple = PyTuple_New(2);
	PyTuple_SetItem(retTuple, 0, PyLong_FromLong(value));
	PyTuple_SetItem(retTuple, 1, PyLong_FromLong(ptr.address));
	return retTuple;
}

/** Calls ompd_get_icv_string_from_scope.
 */
static PyObject* call_ompd_get_icv_string_from_scope(PyObject* self, PyObject* args) {
	PyObject* handlePy = PyTuple_GetItem(args, 0);
	PyObject* scopePy = PyTuple_GetItem(args, 1);
	PyObject* icvIdPy = PyTuple_GetItem(args, 2);
	
	ompd_scope_t scope = (ompd_scope_t) PyLong_AsLong(scopePy);
	void* handle = NULL;
	switch (scope) {
		case ompd_scope_thread:
			handle = (ompd_thread_handle_t*) PyCapsule_GetPointer(handlePy, "ThreadHandle");
			break;
		case ompd_scope_parallel:
			handle = (ompd_parallel_handle_t*) PyCapsule_GetPointer(handlePy, "ParallelHandle");
			break;
		case ompd_scope_implicit_task:
			handle = (ompd_task_handle_t*) PyCapsule_GetPointer(handlePy, "TaskHandle");
			break;
		case ompd_scope_task:
			handle = (ompd_task_handle_t*) PyCapsule_GetPointer(handlePy, "TaskHandle");
			break;
		default:
			handle = (ompd_address_space_handle_t*) PyCapsule_GetPointer(handlePy, "AddressSpace");
			break;
	}
	
	ompd_icv_id_t icvId = (ompd_icv_id_t) PyLong_AsLong(icvIdPy);
	const char* icvString;
	
	// NOTE: call dlsym function
	ompd_rc_t retVal = ompd_get_icv_string_from_scope(handle, scope, icvId, &icvString);
	//ompd_rc_t (*my_get_icv_string_from_scope)(void*, ompd_scope_t, ompd_icv_id_t, const char**) = dlsym(ompd_library, "ompd_get_icv_string_from_scope");
	//ompd_rc_t retVal = my_get_icv_string_from_scope(handle, scope, icvId, &icvString);
	
	if(retVal != ompd_rc_ok) {
		_printf("An error occurred when calling ompd_get_icv_string_from_scope! Error code: %d", retVal);
		return Py_None;
	}
	return PyString_FromString(icvString);
}

/**
 * Binds Python function names to C functions.
 */
static PyMethodDef ompdModule_methods[] = {
	{"ompd_open", ompd_open, METH_VARARGS, "Execute dlopen, return OMPD version."},
	{"call_ompd_initialize", call_ompd_initialize, METH_NOARGS, "Initializes OMPD environment and callbacks."},
	{"call_ompd_rel_thread_handle", call_ompd_rel_thread_handle, METH_VARARGS, "Releases a thread handle."},
	{"get_thread_handle", get_thread_handle, METH_VARARGS, "Collects information on threads."},
	{"call_ompd_get_thread_in_parallel", call_ompd_get_thread_in_parallel, METH_VARARGS, "Obtains handle for a certain thread within parallel region."},
	{"call_ompd_get_curr_parallel_handle", call_ompd_get_curr_parallel_handle, METH_VARARGS, "Obtains a pointer to the parallel handle for the current parallel region."},
	{"call_ompd_get_enclosing_parallel_handle", call_ompd_get_enclosing_parallel_handle, METH_VARARGS, "Obtains a pointer to the parallel handle for the parallel region enclosing the parallel region specified."},
	{"call_ompd_get_task_parallel_handle", call_ompd_get_task_parallel_handle, METH_VARARGS, "Obtains a pointer to the parallel handle for the parallel region enclosing the task region specified."},
	{"call_ompd_rel_parallel_handle", call_ompd_rel_parallel_handle, METH_VARARGS, "Releases a parallel region handle."},
	{"call_ompd_get_curr_task_handle", call_ompd_get_curr_task_handle, METH_VARARGS, "Obtains a pointer to the task handle for the current task region associated with an OpenMP thread."},
	{"call_ompd_get_generating_task_handle", call_ompd_get_generating_task_handle, METH_VARARGS, "Obtains a pointer to the task handle for the task that was created when the task handle specified was encountered."},
	{"call_ompd_get_scheduling_task_handle", call_ompd_get_scheduling_task_handle, METH_VARARGS, "Obtains a pointer to the task handle for the task that scheduled the task specified."},
	{"call_ompd_get_task_in_parallel", call_ompd_get_task_in_parallel, METH_VARARGS, "Obtains the handle for implicit tasks associated with a parallel region."},
	{"call_ompd_rel_task_handle", call_ompd_rel_task_handle, METH_VARARGS, "Releases a task handle."},
	{"call_ompd_get_task_frame", call_ompd_get_task_frame, METH_VARARGS, "Returns a pointer to the enter and exit frame address and flag of the given task."},
	{"call_ompd_enumerate_icvs", call_ompd_enumerate_icvs, METH_VARARGS, "Saves ICVs in map."},
	{"call_ompd_get_icv_from_scope", call_ompd_get_icv_from_scope, METH_VARARGS, "Gets ICVs from scope."},
	{"call_ompd_enumerate_states", call_ompd_enumerate_states, METH_VARARGS, "Enumerates OMP states."},
	{"call_ompd_get_state", call_ompd_get_state, METH_VARARGS, "Returns state for given thread handle."},
	{"call_ompd_get_task_function", call_ompd_get_task_function, METH_VARARGS, "Returns point of code where task starts executing."},
	{"print_capsule", print_capsule, METH_VARARGS, "Print capsule content"},
	{"call_ompd_get_thread_id", call_ompd_get_thread_id, METH_VARARGS, "Maps an OMPD thread handle to a native thread."},
	{"call_ompd_get_tool_data", call_ompd_get_tool_data, METH_VARARGS, "Returns value and pointer of ompd_data_t for given scope and handle."},
	{"call_ompd_get_icv_string_from_scope", call_ompd_get_icv_string_from_scope, METH_VARARGS, "Gets ICV string representation from scope."},
	{NULL, NULL, 0, NULL}
};

/**
 * Lets Python initialize module.
 */
void initompdModule(void)
{
	(void) Py_InitModule("ompdModule", ompdModule_methods);
}