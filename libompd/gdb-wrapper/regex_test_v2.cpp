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
#include <string>
#include <vector>
#include <cstring>
#include <sstream>

using namespace std;

void tokenize(const std::string &str,
    std::vector<std::string> &tokens,
    const std::string &delimiters)
{
  // Skip delimiters at beginning.
  std::string::size_type lastPos = str.find_first_not_of(delimiters, 0);
  // Find first "non-delimiter".
  std::string::size_type pos = str.find_first_of(delimiters, lastPos);

  while (std::string::npos != pos || std::string::npos != lastPos)
  {
    // Found a token, add it to the vector.
    tokens.push_back(str.substr(lastPos, pos - lastPos));
    // Skip delimiters.  Note the "not_of"
    lastPos = str.find_first_not_of(delimiters, pos);
    // Find next "non-delimiter"
    pos = str.find_first_of(delimiters, lastPos);
  }
}

const char input1[] =
    "0x1000010c8 <simpleArray>:      0xd0    0x4f    0x10    0x00    0x01    0x00    0x00    0x00\n"
    "0x1000010d0 <simpleLinkedList>: 0xd0    0x4d    0x10    0x00    0x01    0x00    0x00    0x00\n"
    "0x1000010d8:    0x00    0x00    0x00    0x00    0x00    0x00    0x00    0x00\n"
    "0x1000010e0:    0x00    0x00    0x00    0x00    0x00    0x00    0x00    0x00\n"
    "0x1000010e8:    0x00    0x00    0x00    0x00    0x00    0x00    0x00    0x00\n"
    "0x1000010f0:    0x00    0x00    0x00    0x00    0x00    0x00    0x00    0x00\n"
    "0x1000010f8:    0x00    0x00    0x00    0x00    0x00    0x00    0x00    0x00\n"
    "0x100001100:    0x00    0x00    0x00    0x00    0x00    0x00    0x00    0x00\n"
    "(gdb) ";

const char input2[] =
    "0x1000010c8 <simpleArray>:      0xd0    0x4f    0x10\n"
    "0x1000010d0 <simpleLinkedList>: 0xa0    0x5c    0x20\n"
    "(gdb) ";

regex_t REEXP;

vector<string> parseBytesFromDebuggerOutput(const char input[])
{
  vector<string> ret;

  // split by \n (tokenize by lines)
  string inputStr(input);
  vector<string> lines;
  tokenize(inputStr, lines, "\n");

  for (size_t i=0; i < lines.size()-1; ++i)
  {
    vector<string> addresses;
    tokenize(lines[i], addresses, ":");
    if (addresses.size() == 0) // error if no ':' character is found
      return ret;

    int regRet = regexec(&REEXP, addresses[1].c_str(), 0, NULL, 0);

    if (!regRet)
    {
      vector<string> hexValues;
      tokenize(addresses[1], hexValues, " \t");
      for (size_t k=hexValues.size()-1; k > 0; --k)
         ret.push_back(hexValues[k]);
      ret.push_back(hexValues[0]);
    }
    else
      return ret; // error if no match is found
  }

  return ret;
}

int main()
{

  /** ---- Initialization --------------------------------------------------- */
  int ret;
  ret = regcomp(&REEXP, "^[ \t]*((0){1}[xX]{1}[0-9a-fA-F]{2})+[ \t]*", REG_EXTENDED);
  assert(!ret && "Could not compile regex!");

  /** ---- Processing ------------------------------------------------------- */

  vector<string> ret1 = parseBytesFromDebuggerOutput(input1);
  cout << "Address: ";
  for (size_t i=0; i < ret1.size(); ++i)
    cout << ret1[i] << "-";
  cout << "\n";

  vector<string> ret2 = parseBytesFromDebuggerOutput(input2);
  cout << "Address: ";
  for (size_t i=0; i < ret2.size(); ++i)
    cout << ret2[i] << "-";
  cout << "\n";



  return 0;
}
