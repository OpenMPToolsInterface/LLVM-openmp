/*
 * regex_test.cpp
 *
 *  Created on: Dec 28, 2014
 *      Author: Ignacio Laguna
 *     Contact: ilaguna@llnl.gov
 */

#include <regex.h>
#include <cassert>
#include <iostream>

using namespace std;

int main()
{

  int ret;
  regex_t REEXP;

  //const char *str = "some text here 0x103ed7064 more text here\n(gdb) ";
  const char *str = "  1    Thread 0x2aaaab483040 (LWP 91928) \"target\" 0x00002aaaab19aa3d in nanosleep () from /lib64/libc.so.6";

  ret = regcomp(&REEXP, "^(\\*)?[ \t]+[0-9]+[ \t]+", REG_EXTENDED);
  assert(!ret && "Could not compile regex!");

  size_t     nmatch = 1;
  regmatch_t pmatch[1];

  ret = regexec(&REEXP, str, nmatch, pmatch, 0);
  if (!ret)
  {
    cout << "It matches!!\n";
    cout << "start: " << pmatch[0].rm_so << " end: " << pmatch[0].rm_eo << "\n";
  }
  else
  {
    cout << "Did not match.\n";
  }

  return 0;
}
