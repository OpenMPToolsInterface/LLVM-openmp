/*
 * InputChecker.h
 *
 *  Created on: Jan 7, 2015
 *      Author: Ignacio Laguna
 *     Contact: ilaguna@llnl.gov
 */
#ifndef GDB_INPUTCHECKER_H_
#define GDB_INPUTCHECKER_H_

namespace ompd_gdb {

class InputChecker
{
public:
  static void printUsage();
  static void parseParameters(int argc, char**argv);
};

}




#endif /* GDB_INPUTCHECKER_H_ */
