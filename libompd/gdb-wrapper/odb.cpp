/*
 * odb.cpp
 *
 *  Created on: Jan 7, 2015
 *      Author: Ignacio Laguna
 *     Contact: ilaguna@llnl.gov
 */

#include "InputOutputManager.h"
#include <cstdlib>
#include <iostream>

using namespace ompd_gdb;

int main(int argc, char **argv)
{
  InputOutputManager ioManager(argc, argv, true);
  ioManager.run();
  ioManager.finalize();
  return EXIT_SUCCESS;
}

