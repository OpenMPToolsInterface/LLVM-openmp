/*
 * OutputString.h
 *
 *  Created on: Jan 9, 2015
 *      Author: Ignacio Laguna
 *     Contact: ilaguna@llnl.gov
 */
#ifndef GDB_OUTPUTSTRING_H_
#define GDB_OUTPUTSTRING_H_

namespace ompd_gdb {

class OutputString
{
private:
  bool useSTDOUT = true;
public:
  OutputString();
  void operator << (const char *str) const;
};

}

#endif /* GDB_OUTPUTSTRING_H_ */
