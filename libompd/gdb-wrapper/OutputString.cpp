/*
 * OutputString.cpp
 *
 *  Created on: Jan 9, 2015
 *      Author: Ignacio Laguna
 *     Contact: ilaguna@llnl.gov
 */

#include "OutputString.h"
#include <iostream>
#include <cstdlib>

using namespace ompd_gdb;
using namespace std;

OutputString::OutputString()
{
  char * val = getenv("OBD_DO_NOT_USE_STDOUT");
  useSTDOUT = val ? false : true;
}

void OutputString::operator <<(const char *str) const
{
  if (useSTDOUT)
    cout << str;
}

