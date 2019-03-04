/*
 * gdb_wrapper.cpp
 *
 *  Created on: Dec 21, 2014
 *      Author: Ignacio Laguna
 *     Contact: ilaguna@llnl.gov
 */

#include "InputChecker.h"
#include "StringParser.h"
#include "GdbProcess.h"
#include "Callbacks.h"
#include "OMPDCommand.h"
#include "ompd_test.h"

#include <vector>
#include <string>
#include <memory>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <signal.h>
#include <dlfcn.h>

using namespace std;
using namespace ompd_gdb;

/* --- Global static variables ---------------------------------------------- */
static StringParser parser;
static GdbProcessPtr gdbProc(nullptr);
static OMPDCommandFactoryPtr commandFactory(nullptr);

/* --- Initialization routines ---------------------------------------------- */
//static void parseInputParameters(int argc, char **argv);
static void initializeErrorHandlers();
static void sigHandler(int signo);
//static void printUsage();
/* --- Processing routines -------------------------------------------------- */
static void processOMPDCommand(const char *str);
/* --- Finalization routines ------------------------------------------------ */
static void terminateGDB();

int main(int argc, char **argv)
{
  // Initial steps
  InputChecker::parseParameters(argc, argv);
  gdbProc = GdbProcessPtr(new GdbProcess(argv));
  commandFactory = OMPDCommandFactoryPtr(new OMPDCommandFactory);
  initializeErrorHandlers();
  initializeCallbacks(gdbProc);

  // Main loop.
  // Alternate between reading GDB's output and reading user's input.
  bool readOutput = true;
  char userInput[256];
  while(true) {
    if (readOutput)
      cout << gdbProc->readOutput();

    // Read command from the user
    userInput[0] = '\0';
    std::cin.getline(userInput,255);
    if (parser.isQuitCommand(userInput)) // if quit command was sent, terminate
      break;
    else
    {
      if (parser.isOMPDCommand(userInput)) // process OMPD command if sent
      {
        processOMPDCommand(userInput);
        // print GDB prompt since it is consumed by the processing routine
        cout << StringParser::GDB_PROMPT;
        readOutput = false; // we don't read output
        continue;
      }
      gdbProc->writeInput(userInput); // send user command to GDB
    }
    readOutput = true;
  }

  // Clean up everything before ending
  terminateGDB();
  return EXIT_SUCCESS;
}

/*void parseInputParameters(int argc, char**argv)
{
  // Check input is correct
  if (argc != 2)
    printUsage();
  else
  {
    int pid = atoi(argv[1]);
    if (pid == 0 || pid < 0)
    {
      cerr << "ERROR: incorrect PID!\n";
      printUsage();
    }
  }
}

void printUsage()
{
  cerr << "Usage:\n\tgdb_wrapper ID\n";
  cerr << "ID: process ID (integer)\n";
  exit(EXIT_FAILURE);
}*/

void initializeErrorHandlers()
{
  // Register signal handlers
  signal(SIGINT,    sigHandler);
  signal(SIGTSTP,   sigHandler);
  signal(SIGSTOP,   sigHandler);
}

void sigHandler(int signo)
{
  cerr << "Got a signal. Exiting...\n";
  terminateGDB();
  exit(EXIT_FAILURE);
}

void terminateGDB()
{
  gdbProc->finalize();
  cout << "Waiting to terminate GDB..." << endl;
  cout << "GDB exit status: ";
  cout << gdbProc->wait() << endl;
}

void processOMPDCommand(const char *str)
{
  vector<string> params;
  tokenize(str, params, " \t");

  OMPDCommand *command;
  if (params.size() > 1)
    command = commandFactory->create(params[1].c_str());
  else
    command = commandFactory->create("None"); // in case no command is passed

  command->execute();
}
