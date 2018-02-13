/*
 * InputOutputManager.h
 *
 *  Created on: Jan 7, 2015
 *      Author: Ignacio Laguna
 *     Contact: ilaguna@llnl.gov
 */
#ifndef GDB_INPUTOUTPUTMANAGER_H_
#define GDB_INPUTOUTPUTMANAGER_H_

#include "InputChecker.h"
#include "StringParser.h"
#include "GdbProcess.h"
#include "OMPDCommand.h"

#include <ostream>
#include <string>

namespace ompd_gdb {

class InputOutputManager
{
private:
  StringParser parser;
  OMPDCommandFactoryPtr commandFactory = nullptr;
  bool output;

  void initializeErrorHandlers();
  void processOMPDCommand(const char *str);

public:
  InputOutputManager(int argc, char **argv, bool output);
  //~InputOutputManager();

  /**
   * Run the manager (the main loop).
   */
  void run();

  /**
   * Read output from the debugger
   */
  std::string readFromDebugger();

  /**
   * Write input to the debugger
   */
  void writeToDebugger(const char* str);

  void finalize();

};

}

void terminateGDB();
void sigHandler(int signo);
void sigChildKilled(int signo);
void sigForwardGdb(int signo);


#endif /* GDB_INPUTOUTPUTMANAGER_H_ */
