/*
 * InputOutputManager.cpp
 *
 *  Created on: Jan 7, 2015
 *      Author: Ignacio Laguna
 *     Contact: ilaguna@llnl.gov
 */

#include "InputOutputManager.h"
#include "InputChecker.h"
#include "StringParser.h"
#include "GdbProcess.h"
#include "OMPDCommand.h"
#include "OMPDContext.h"
#include "Callbacks.h"
#include "OutputString.h"

#include <cstdlib>
#include <iostream>
#include <ostream>
#include <sstream>
#include <string>
#include <readline/readline.h>
#include <readline/history.h>
#include <signal.h>

using namespace ompd_gdb;
using namespace std;

/* --- Initialize ----------------------------------------------------------- */

static GdbProcessPtr gdbProc(nullptr);
static OutputString out;
OMPDHostContextPool * host_contextPool;

/**
 * FIXME: Pass cout as output stream. Do not use boolean input
 */
InputOutputManager::InputOutputManager(int argc, char **argv, bool _out)
: output(_out)
{
  // Initial steps
  initializeErrorHandlers();
  gdbProc = GdbProcessPtr(new GdbProcess(argc, argv));
  initializeCallbacks(gdbProc);
  host_contextPool=new OMPDHostContextPool(gdbProc);
  commandFactory = OMPDCommandFactoryPtr(new OMPDCommandFactory);
}

void InputOutputManager::initializeErrorHandlers()
{
  // Register signal handlers
  signal(SIGINT,    sigForwardGdb);
  signal(SIGTSTP,   sigForwardGdb);
  signal(SIGSTOP,   sigForwardGdb);

// child(gdb) died
  signal(SIGCHLD,   sigChildKilled);
}

/**
 * FIXME: signal handlers should be defined in a separate file
 */
void sigHandler(int signo)
{
  cerr << "Got a signal. Exiting...\n";
  terminateGDB();
  exit(EXIT_FAILURE);
}

/**
 * FIXME: signal handlers should be defined in a separate file
 */
void sigChildKilled(int signo)
{
    cerr << "GDB process finished. Shutting down...\n";
    gdbProc->wait(); // We do not care about the return value of wait call
    exit(EXIT_SUCCESS);
}

/**
 * FIXME: signal handlers should be defined in a separate file
 */
void sigForwardGdb(int signo)
{
  if (gdbProc.get())
  {
    gdbProc->kill(signo);
  }
}

/* --- Finalize ------------------------------------------------------------- */

//InputOutputManager::~InputOutputManager()
//{
//  terminateGDB();
//}

void InputOutputManager::finalize()
{
  terminateGDB();
}

/**
 * FIXME: this function should be within a namespace scope
 */
void terminateGDB()
{
  if (gdbProc.get())
  {
    gdbProc->finalize();
    stringstream msg;
    msg << "Waiting to terminate GDB..." << endl;
    msg << "GDB exit status: ";
    msg << gdbProc->wait() << endl;
    out << msg.str().c_str();
  }
}

/* --- Process -------------------------------------------------------------- */

/* This is the main loop that takes input commands from users
 * The logic is the following:
 *   (1) If the command is the quit command, it terminates the loop and the
 *   function returns
 *   (2) If it is an ODB command, it process it internally
 *   (3) If it is a gdb command, it send the command to the gdb process
 */
void InputOutputManager::run()
{
  // Alternate between reading GDB's output and reading user's input.
  bool readOutput = false;
  //char userInput[256];
//  gdbProc->writeInput("");
  //cout << StringParser::GDB_PROMPT;
  while(true) {
    if (readOutput)
      if (output)
      {
        string deb = readFromDebugger();
        // Eliminate gdb prompt
        parser.eliminateGDBPromptInplace(deb);
        cout << deb;
      }

    // Read command from the user
    //userInput[0] = '\0';
    //cin.getline(userInput,255);

    // Using readline library
    char *userInput = readline(StringParser::GDB_PROMPT);

    // If the line has any text in it, save it on the history
    if (userInput && *userInput)
      add_history(userInput);

    // if quit command was sent, terminate
    //if ((cin.rdstate() & cin.eofbit) || parser.isQuitCommand(userInput))
    if (!userInput || parser.isQuitCommand(userInput))
    {
      if (userInput)
        free(userInput);
      break;
    }
    else
    {
      if (parser.isOMPDCommand(userInput)) // process OMPD command if sent
      {
        processOMPDCommand(userInput);
        // print GDB prompt since it is consumed by the processing routine
        //if (output)
        //  cout << StringParser::GDB_PROMPT;
        readOutput = false; // we don't read output
        if (userInput)
          free(userInput);
        continue;
      }
      //gdbProc->writeInput(userInput);
      writeToDebugger(userInput); // send user command to GDB
    }
    readOutput = true;
  }
}

/**
 * FIXME: create class to manage OMPD (or ODB) commands
 * FIXME: Commands should be renamed to ODB?
 */
void InputOutputManager::processOMPDCommand(const char *str)
{
  vector<string> params;
  tokenize(str, params, " \t");

  OMPDCommand *command;
  if (params.size() > 1)
  {
    if (params.size() > 2)
    {
      auto i=params.begin();
      i+=2;
      std::vector<string> extraArgs(i, params.end());
      command = commandFactory->create(params[1].c_str(), extraArgs);
    }
    else
      command = commandFactory->create(params[1].c_str());
  }
  else
    command = commandFactory->create("None"); // in case no command is passed

  command->execute();
}

string InputOutputManager::readFromDebugger()
{
  return gdbProc->readOutput();
}

void InputOutputManager::writeToDebugger(const char* str)
{
  gdbProc->writeInput(str);
}
