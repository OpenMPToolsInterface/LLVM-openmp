/*
 * GdbProcess.cpp
 *
 *  Created on: Dec 27, 2014
 *      Author: Ignacio Laguna
 *     Contact: ilaguna@llnl.gov
 */

#include "ChildProcess.h"
#include "GdbProcess.h"
#include "Debug.h"
#include <memory>
#include <string>
#include <iostream>
#include <cstring>

using namespace ompd_gdb;
using namespace std;

#ifdef DEBUG
int display_gdb_output = true;
#else
int display_gdb_output = false;
#endif

/**
 * This function gets all the parameters passed via the argv array
 */
void GdbProcess::getArgvParameters(int argc, char **argv, const char **newArgv)
{
  newArgv[0] = gdbPath;
  for (int i=1; i < argc; ++i)
    newArgv[i] = argv[i];
  newArgv[argc] = (const char*)0;
}

GdbProcess::GdbProcess(int argc, char **argv)
{
  // Prepare GDB path and input parameters of GDB
  const char **newArgv = (const char **)malloc(sizeof(char *) * (argc+1));
  getArgvParameters(argc, argv, newArgv);

  //gdbProc = GdbProc(new ProcessSpawn(prog));
  gdbProc = GdbProc(new ChildProcess(newArgv));

  // Print initial gdb output in stdout
  cout << readOutput();
  free(newArgv);
}

string GdbProcess::readOutput() const
{
  string ret("");
  char str[256]; // read in 256-byte chunks
  str[0] = '\0';
  while (true)
  {
    //gdbProc->stdout.readsome(str, 255);
    size_t c = gdbProc->readSome(str, 255);
    //int c =  gdbProc->stdout.gcount(); // number of characters read
    str[c] = '\0';
    if (c > 0) // if we read something, add it to the returning string
      ret += str;
    if (parser.hasGDBPrompt(str))
      break; // Stop when gdb prompt is found
    str[0] = '\0';
  }

  if (display_gdb_output)
    dout << "READING FROM GDB:===" << ret << "===" << endl;

  return ret;
}

void GdbProcess::writeInput(const char *str) const
{
  if (display_gdb_output)
    dout << "SENDING TO GDB:===" << str << "===" << std::endl;
  gdbProc->writeSome(str);
  gdbProc->writeSome("\n");
}
void GdbProcess::finalize()
{
  gdbProc->sendEOF();
}
void GdbProcess::kill(int sig)
{
  gdbProc->sendSignal(sig);
}

int GdbProcess::wait()
{
  return gdbProc->wait();
}
