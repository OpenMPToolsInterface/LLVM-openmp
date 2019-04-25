/*
 * OMPDCommand.h
 *
 *  Created on: Dec 28, 2014
 *      Author: Ignacio Laguna
 *     Contact: ilaguna@llnl.gov
 */
#ifndef GDB_OMPDCOMMAND_H_
#define GDB_OMPDCOMMAND_H_

/*******************************************************************************
 * These classes implement ompd commands for GDB.
 * Commands start with the "ompd" word followed by the command name. Thus, each
 * command comprises two words only in this format: "ompd [COMMAND]".
 *
 * There is a command factory that must be instantiated to create commands.
 * Instantiating the factory class allows loading the DLLs that provide
 * OMPD function calls (and looking up these functions).
 *
 * All commands are derived from the OMPDCommand class. There is a null command
 * (OMPDNull) that is used when an invalid command is entered or when a regular
 * command cannot be executed (for any reason).
 */

#include "OutputString.h"
#include <cstring>
#include <cassert>
#include <memory>
#include <vector>
#include <string>
#include <map>
#include "omp-tools.h"
#include "ompd_typedefs.h"
//#include "ompd_test.h"

#define FOREACH_OMPD_API_FN(macro) \
macro(ompd_process_initialize) \
macro(ompd_device_initialize) \
macro(ompd_rel_address_space_handle) \
macro(ompd_initialize) \
macro(ompd_finalize) \
macro(ompd_get_thread_in_parallel) \
macro(ompd_rel_thread_handle) \
macro(ompd_thread_handle_compare) \
macro(ompd_get_thread_id) \
macro(ompd_get_curr_parallel_handle) \
macro(ompd_get_enclosing_parallel_handle) \
macro(ompd_get_task_parallel_handle) \
macro(ompd_rel_parallel_handle) \
macro(ompd_parallel_handle_compare) \
macro(ompd_get_curr_task_handle) \
macro(ompd_get_generating_task_handle) \
macro(ompd_get_task_in_parallel) \
macro(ompd_rel_task_handle) \
macro(ompd_task_handle_compare) \
macro(ompd_get_thread_handle) \
macro(ompd_enumerate_states) \
macro(ompd_get_state) \
macro(ompd_get_task_function) \
macro(ompd_get_task_frame) \
macro(ompd_get_api_version) \
macro(ompd_enumerate_icvs) \
macro(ompd_get_icv_from_scope) \


namespace ompd_gdb {

/**
 * Function pointers of OMPD function calls. These functions are used by the
 * OMPD commands that our gdb warper supports.
 */
typedef struct
{
  /* Handle of OMPD DLL */
  void *ompdLibHandle = nullptr;


  /* Test function calls (only from the ompd_test library) */
  void (*test_CB_dmemory_alloc)()   = nullptr;
  void (*test_CB_tsizeof_prim)()    = nullptr;

  /* OMPD API function pointer
   * The Macro generates function pointer for all implemented API function listed in OMPDCommand.h:41
   */
#define OMPD_API_FUNCTION_POINTER_MEMBER(FN) FN##_fn_t FN = nullptr;
FOREACH_OMPD_API_FN(OMPD_API_FUNCTION_POINTER_MEMBER)
#undef OMPD_API_FUNCTION_POINTER_MEMBER

} OMPDFunctions;

typedef std::shared_ptr<OMPDFunctions> OMPDFunctionsPtr;

class OMPDIcvs
{
private:
  OMPDFunctionsPtr functions;
  std::map<std::string, std::pair<ompd_icv_id_t, ompd_scope_t>> availableIcvs;
public:
  OMPDIcvs(OMPDFunctionsPtr functions,
           ompd_address_space_handle_t *addrhandle);
  ompd_rc_t get(ompd_parallel_handle_t *handle, const char *name,
                ompd_word_t *value);
};

typedef std::shared_ptr<OMPDIcvs> OMPDIcvsPtr;

class OMPDParallelHandleCmp
{
  OMPDFunctionsPtr functions;
public:
  OMPDParallelHandleCmp(const OMPDFunctionsPtr &f)
    : functions(f) {}
  bool operator()(ompd_parallel_handle_t *a, ompd_parallel_handle_t *b) {
    int cmp = 0;
    functions->ompd_parallel_handle_compare(a, b, &cmp);
    return cmp < 0;
  }
};

class OMPDThreadHandleCmp
{
  OMPDFunctionsPtr functions;
public:
  OMPDThreadHandleCmp(const OMPDFunctionsPtr &f)
    : functions(f) {}
  bool operator()(ompd_thread_handle_t *a, ompd_thread_handle_t *b) {
    int cmp = 0;
    functions->ompd_thread_handle_compare(a, b, &cmp);
    return cmp < 0;
  }
};

class OMPDTaskHandleCmp
{
  OMPDFunctionsPtr functions;
public:
  OMPDTaskHandleCmp(const OMPDFunctionsPtr &f)
    : functions(f) {}
  bool operator()(ompd_task_handle_t *a, ompd_task_handle_t *b) {
    int cmp = 0;
    functions->ompd_task_handle_compare(a, b, &cmp);
    return cmp < 0;
  }
};

class OMPDCommand;

class OMPDCommandFactory
{
private:
  void * findFunctionInLibrary(const char *fun) const;
  void initOmpd();
  OMPDFunctionsPtr functions = nullptr;
  OMPDIcvsPtr icvs = nullptr;
//   ompd_process_handle_t* prochandle = nullptr;
  ompd_address_space_handle_t* addrhandle = nullptr;
  OutputString out;

public:
  OMPDCommandFactory();
  ~OMPDCommandFactory();
//  OMPDCommand* create(const char *str) const;
  OMPDCommand* create(const char *str, const std::vector<std::string>& extraArgs=std::vector<std::string>());
};

typedef std::unique_ptr<OMPDCommandFactory> OMPDCommandFactoryPtr;

/**
 * Abstract class for OMPD command of the type: "ompd [COMMAND]"
 */
class OMPDCommand
{
protected:
  OMPDFunctionsPtr functions = nullptr;
  ompd_address_space_handle_t* addrhandle = nullptr;
  std::vector<std::string> extraArgs;
public:
  OMPDCommand(): extraArgs(){}
  OMPDCommand(const std::vector<std::string>& args): extraArgs(args){}
  OMPDCommand(const OMPDFunctionsPtr &f, ompd_address_space_handle_t* ah, const std::vector<std::string>& args) : functions(f), addrhandle(ah), extraArgs(args) {};
  virtual ~OMPDCommand(){}
  virtual void execute() const = 0;
  virtual const char* toString() const = 0;
};

/**
 * Null command.
 * This command doesn't do anything useful. It should be called when an invalid
 * ompd command is requested by the user.
 */
class OMPDNull : public OMPDCommand
{
public:
  ~OMPDNull(){};
  void execute() const;
  static void staticExecute();
  const char* toString() const;
};

/**
 * COMMAND: "ompd test"
 * This command tests all the debugger callbacks and print useful information
 * about them. This is to be used when we want to test that callbacks are
 * functioning properly.
 */
class OMPDTestCallbacks : public OMPDCommand
{
public:
  ~OMPDTestCallbacks(){};
  void execute() const;
  const char* toString() const;
protected:
  OMPDTestCallbacks(const OMPDFunctionsPtr &f, ompd_address_space_handle_t* ah, const std::vector<std::string>& args) : OMPDCommand(f, ah, args){};

  friend OMPDCommandFactory;
};


class OMPDSpaces : public OMPDCommand
{
public:
  ~OMPDSpaces(){};
  void execute() const;
  const char* toString() const;
protected:
  OMPDSpaces(const OMPDFunctionsPtr &f, ompd_address_space_handle_t* ah, const std::vector<std::string>& args) : OMPDCommand(f, ah, args){};

  friend OMPDCommandFactory;
};

class OMPDThreads : public OMPDCommand
{
public:
  ~OMPDThreads(){};
  void execute() const;
  const char* toString() const;
protected:
  OMPDThreads(const OMPDFunctionsPtr &f, ompd_address_space_handle_t* ah, const std::vector<std::string>& args) : OMPDCommand(f, ah, args){};

  friend OMPDCommandFactory;
};


class OMPDLevels : public OMPDCommand
{
  OMPDIcvsPtr icvs;
public:
  ~OMPDLevels(){};
  void execute() const;
  const char* toString() const;
protected:
  OMPDLevels(const OMPDFunctionsPtr &f, ompd_address_space_handle_t* ah, const OMPDIcvsPtr &icvs, const std::vector<std::string>& args)
      : OMPDCommand(f, ah, args), icvs(icvs) {};

  friend OMPDCommandFactory;
};

class OMPDCallback : public OMPDCommand
{
public:
  ~OMPDCallback(){};
  void execute() const;
  const char* toString() const;
protected:
  OMPDCallback(const OMPDFunctionsPtr &f, ompd_address_space_handle_t* ah, const std::vector<std::string>& args) : OMPDCommand(f, ah, args){};

  friend OMPDCommandFactory;
};

class OMPDApi : public OMPDCommand
{
public:
  ~OMPDApi(){};
  void execute() const;
  const char* toString() const;
protected:
  OMPDApi(const OMPDFunctionsPtr &f, ompd_address_space_handle_t* ah, const std::vector<std::string>& args) : OMPDCommand(f, ah, args){};

  friend OMPDCommandFactory;
};

class OMPDTest : public OMPDCommand
{
public:
  ~OMPDTest(){};
  void execute() const;
  const char* toString() const;
protected:
  OMPDTest(const OMPDFunctionsPtr &f, ompd_address_space_handle_t* ah,
           const OMPDIcvsPtr &icvs, const std::vector<std::string>& args)
    : OMPDCommand(f, ah, args), icvs(icvs) {};

  friend OMPDCommandFactory;
private:
  OMPDIcvsPtr icvs;
};

class OMPDParallelRegions : public OMPDCommand
{
public:
  ~OMPDParallelRegions() {};
  void execute() const;
  const char *toString() const;
protected:
  OMPDParallelRegions(const OMPDFunctionsPtr &f,
                      ompd_address_space_handle_t *ah, const OMPDIcvsPtr &icvs,
                      const std::vector<std::string>& args)
    : OMPDCommand(f, ah, args), icvs(icvs) {}

  friend OMPDCommandFactory;
private:
  OMPDIcvsPtr icvs;
};

class OMPDTasks : public OMPDCommand
{
public:
  ~OMPDTasks() {}
  void execute() const;
  const char *toString() const;
protected:
  OMPDTasks(const OMPDFunctionsPtr &f,
            ompd_address_space_handle_t *ah, const OMPDIcvsPtr &icvs,
            const std::vector<std::string>& args)
    : OMPDCommand(f, ah, args), icvs(icvs) {}
  friend OMPDCommandFactory;
private:
  OMPDIcvsPtr icvs;
};

}

#endif /* GDB_OMPDCOMMAND_H_ */
