/*
 * GdbProcess.h
 *
 *  Created on: Dec 27, 2014
 *      Author: Ignacio Laguna
 *     Contact: ilaguna@llnl.gov
 */
#ifndef GDB_GDBPROCESS_H_
#define GDB_GDBPROCESS_H_


#include "ChildProcess.h"
//#include "ProcessSpawn.h"
#include "StringParser.h"
#include <memory>
#include <string>

namespace ompd_gdb {

/**
 * This class allows sending commands to a gdb process and receiving its output.
 * It assumes that the main process spawns a gdb process and that a pipe is
 * used for communication.
 */
class GdbProcess
{
private:

#if defined(CUDA_GDB_PATH)
  const char *gdbPath = CUDA_GDB_PATH;
#elif defined(GDB_PATH)
  const char *gdbPath = GDB_PATH;
#else
  const char *gdbPath = "/usr/bin/gdb";
#endif

  void getArgvParameters(int argc, char **argv, const char **newArgv);

  //typedef std::unique_ptr<ProcessSpawn> GdbProc;
  typedef std::unique_ptr<ChildProcess> GdbProc;
  GdbProc gdbProc = nullptr;
  StringParser parser;

public:
  GdbProcess(int argc, char **argv);

  /**
   * Read data from the process until a gdb prompt is seen (i.e., "(gdb) ").
   * NOTE: this call may block if the gdb prompt is not seen.
   */
  std::string readOutput() const;

  /**
   * Send data to a gdb process (usually in the form of a command that is
   * terminated by a new line character ('\n').
   */
  void writeInput(const char* str) const;

  /**
   * Finalize GDB.
   */
  void finalize();
  /**
   * Send GDB a signal.
   */
  void kill(int sig);

  /**
   * Wait for GDB to finalize
   */
  int wait();
};

typedef std::shared_ptr<GdbProcess> GdbProcessPtr;

}

#endif /* GDB_GDBPROCESS_H_ */
