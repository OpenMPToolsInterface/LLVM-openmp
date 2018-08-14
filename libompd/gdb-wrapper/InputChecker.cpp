/*
 * InputChecker.cpp
 *
 *  Created on: Jan 7, 2015
 *      Author: Ignacio Laguna
 *     Contact: ilaguna@llnl.gov
 */

#include "InputChecker.h"
#include <iostream>
#include <cstdlib>

using namespace std;
using namespace ompd_gdb;

void InputChecker::printUsage()
{
  cerr << "Usage:\n\tgdb_wrapper ID\n";
  cerr << "ID: process ID (integer)\n";
  exit(EXIT_FAILURE);
}

void InputChecker::parseParameters(int argc, char **argv)
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
