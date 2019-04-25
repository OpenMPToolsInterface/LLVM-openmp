/*
 * StringParser.cpp
 *
 *  Created on: Dec 26, 2014
 *      Author: Ignacio Laguna
 *     Contact: ilaguna@llnl.gov
 */

#include "StringParser.h"
#include <regex.h>
#include <cassert>
#include <string>
#include <cstring>
#include <vector>
#include <map>
#include <iostream>
#include "omp-tools.h"

using namespace ompd_gdb;
using namespace std;

StringParser::StringParser()
{
  // Compile regular expressions
  int ret = 0;

  // To check if gdb prompt is present
  ret = regcomp(&rePROMPT, "[(](cuda-)?gdb[)][ ]$", REG_EXTENDED);
  assert(!ret && "Could not compile regex rePROMPT!");

  // To check if quit command was invoked
  ret = regcomp(&reQUIT, "^[ \t]*(quit){1}[ \t]*$", REG_EXTENDED);
  assert(!ret && "Could not compile regex reQUIT!");

  // To check if OMPD command was invoked
  ret = regcomp(&reOMPD_COMMAND, "^[ \t]*(odb){1}[ \t]*", REG_EXTENDED);
  assert(!ret && "Could not compile regex reOMPD_COMMAND!");

  // To check to a regular value given by GDB (e.g., "$1 = 4")
  ret = regcomp(&reREGULAR_VALUE,
      //"^(\\$){1}[0-9]+[ ](=){1}[ ][0-9a-zA-Z]+\n", REG_EXTENDED);
      "^(\\$){1}[0-9]+[ ](=){1}[ ][0-9a-zA-Z]+", REG_EXTENDED);
  assert(!ret && "Could not compile regex reREGULAR_VALUE!");

  /* To check a he address value, e.g. 0x0FFAA */
  ret = regcomp(&reADDRESS_VALUE,
        "(0){1}[xX]{1}[0-9a-fA-F]+", REG_EXTENDED);
  assert(!ret && "Could not compile regex reADDRESS_VALUE!");

  // Match output of gdb when reading memory (e.g '$ x/32xb' for 32 bytes)
  ret = regcomp(&reMEMORY_VALUES,
         "[0-9]+", REG_EXTENDED);
  assert(!ret && "Could not compile regex reMEMORY_VALUES!");

/*  // Match output of gdb when reading memory (e.g '$ x/32xb' for 32 bytes)
  ret = regcomp(&reMEMORY_VALUES,
         "(0){1}[xX]{1}[0-9a-fA-F]+", REG_EXTENDED);
  assert(!ret && "Could not compile regex reMEMORY_VALUES!");*/

  /* Match thread ids in the output of "info threads"*/
  ret = regcomp(&reTHREAD_ID, "^(\\*)?[ \t]+[0-9]+[ \t]+", REG_EXTENDED);
  assert(!ret && "Could not compile regex reTHREAD_ID!");

  /* Match thread ids in the output of "info cuda contexts"*/
  ret = regcomp(&reCONTEXT_ID, "([0-9]+)[ \t]+active", REG_EXTENDED);
  assert(!ret && "Could not compile regex reCONTEXT_ID!");

  /* Match thread ids in the output of "info cuda kernels"*/
  ret = regcomp(&reKERNELS_ID, "([0-9]+)[ \t-]+([0-9]+)[ \t-]+([0-9]+)", REG_EXTENDED);
  assert(!ret && "Could not compile regex reKERNELS_ID!");

  /* Match thread ids in the output of "info cuda threads"*/
	ret = regcomp(&reBLOCKS_ID, "([0-9]+).*[ \t]+([0-9]+)[ \t]0x", REG_EXTENDED);
  assert(!ret && "Could not compile regex reBLOCKS_ID!");

  /* Match thread id in the output of "thread" 
      [Current thread is 1 (Thread 0x2aaaad394d60 (LWP 17641))] */
  ret = regcomp(&reTHREAD, "^\[Current thread is [0-9]+ ", REG_EXTENDED);
  assert(!ret && "Could not compile regex reTHREAD!");
}

bool StringParser::regexpMatches(const char *str, const regex_t *re) const
{
  int ret = regexec(re, str, 0, NULL, 0);
  if (!ret)
    return true;
  return false;
}

bool StringParser::isOMPDCommand(const char *str) const
{
  return regexpMatches(str, &reOMPD_COMMAND);
}

bool StringParser::isQuitCommand(const char *str) const
{
  return regexpMatches(str, &reQUIT);
}

bool StringParser::hasGDBPrompt(const char *str) const
{
  return regexpMatches(str, &rePROMPT);
}

void StringParser::matchRegularValue(const char *str, char *value) const
{
  bool match = regexpMatches(str, &reREGULAR_VALUE);
  if (!match) // regular value not found
  {
    value[0] = '\0';
    return;
  }

  vector<string> tokens;
  tokenize(str, tokens, "\n");
  vector<string> v;
  tokenize(tokens[0], v, " "); // get first line and tokenize by space
  strcpy(value, v[2].c_str());
  value[strlen( v[2].c_str() )] = '\0';
}

void StringParser::matchAddressValue(const char *str, char *addr) const
{
  size_t     nmatch = 1;
  regmatch_t pmatch[1];

  int ret = regexec(&reADDRESS_VALUE, str, nmatch, pmatch, 0);
  if (ret) // if address not found
  {
    addr[0] = '\0';
    return;
  }

  int size =  pmatch[0].rm_eo - pmatch[0].rm_so;
  //char dst[size];
  strncpy (addr, (str + pmatch[0].rm_so), size);
  addr[size] = '\0';
  //return dst;
}

vector<string> StringParser::matchMemoryValues(const char *str) const
{
  vector<string> ret;

  // split by \n (tokenize by lines)
  string inputStr(str);
  vector<string> lines;
  tokenize(inputStr, lines, "\n");

  for (size_t i=0; i < lines.size()-1; ++i)
  {
    vector<string> addresses;
    tokenize(lines[i], addresses, ":");
    if (addresses.size() == 0) // error if no ':' character is found
      return ret;

    int regRet = regexec(&reMEMORY_VALUES, addresses[1].c_str(), 0, NULL, 0);

    if (!regRet)
    {
      tokenize(addresses[1], ret, " \t");
      /*vector<string> hexValues;
      tokenize(addresses[1], hexValues, " \t");
      for (size_t k=0, e = hexValues.size(); k < e; ++k)
        ret.push_back(hexValues[k]);*/
    }
    else
      return ret; // error if no match is found
  }

  return ret;
}

/**
 * This function parses the following command in gdb:
 * ----------------------------------------------------------------------------
 * (gdb) info cuda threads
 *
 * (cuda-gdb) info cuda threads
 *   BlockIdx ThreadIdx To BlockIdx ThreadIdx Count         Virtual PC Filename  Line 
 * Kernel 0
 * *  (0,0,0)   (0,0,0)     (0,0,0)  (31,0,0)    32 0x00001000086ba1b8      n/a     0 
 *    (0,0,0)  (32,0,0)     (0,0,0)  (32,0,0)     1 0x00001000086b27a8      n/a     0 
 * (cuda-gdb) 
 * ----------------------------------------------------------------------------
 *
 * Returns vector of blocks of vector of threads
 */
vector<CudaThread> StringParser::matchCudaThreadsInfo(
        uint64_t ctx, uint64_t dev, uint64_t kernel, 
        uint64_t grid, const char *str
) const
{
  string inputStr(str);
  vector<string> lines;
  tokenize(inputStr, lines, "\n");
  map<int, int> threadcounts;
  vector<CudaThread> ret;

  // Do not parse the first two lines and the last line
  for (size_t i=2; i < lines.size()-1; ++i)
  {
    string     block_num;
    string     threadcnt;
    size_t     nmatch = 3;
    regmatch_t pmatch[3];

    int regRet = regexec(&reBLOCKS_ID, lines[i].c_str(), nmatch, pmatch, 0);
    if (regRet)
      return ret;

    block_num = lines[i].substr(pmatch[1].rm_so, pmatch[1].rm_eo - pmatch[1].rm_so);
    threadcnt = lines[i].substr(pmatch[2].rm_so, pmatch[2].rm_eo - pmatch[2].rm_so);

    threadcounts[atoi(block_num.c_str())] += atoi(threadcnt.c_str());
  }

  ompd_cudathread_coord_t coord;

  //coord.gridDim = _gdim_;       // TODO (needed by TotalView, not GDB)
  //coord.blockDim = _bdim_;      // TODO (needed by TotalView, not GDB)
  //coord.warpSize = _wsize_;     // TODO (needed by TotalView, not GDB)

  coord.gridId = grid;
  coord.cudaContext = ctx;
  coord.cudaDevId = dev;
  coord.warpSize = 0;

  for (int b = 0; b < threadcounts.size(); ++b) {
    coord.blockIdx.x = b;
    coord.blockIdx.y = 0;
    coord.blockIdx.z = 0;
    for (int t = 0; t < threadcounts[b]; ++t) {
      coord.threadIdx.x = t;
      coord.threadIdx.y = 0;
      coord.threadIdx.z = 0;
      ret.push_back(CudaThread{coord});
    }
  }

  return ret;
}

/**
 * This function parses the following command in gdb:
 * ----------------------------------------------------------------------------
 * (cuda-gdb) info cuda kernels
 *  Kernel Parent Dev Grid Status   SMs Mask GridDim  BlockDim Invocation 
 *     3      -   3    7 Active 0x00000001 (1,1,1) (160,1,1) __omp_offloading_50_
 *     2      -   2    7 Active 0x00000001 (1,1,1) (128,1,1) __omp_offloading_50_
 *     1      -   1    7 Active 0x00000001 (1,1,1)  (96,1,1) __omp_offloading_50_
 * *   0      -   0    7 Active 0x00000001 (1,1,1)  (64,1,1) __omp_offloading_50_
 * ----------------------------------------------------------------------------
 *
 * It returns a map of kernel ID to <device,grid> pairs.
 */
map<int, pair<int,int>> StringParser::matchCudaKernelsInfo(const char *str) const
{
  // split by \n (tokenize by lines)
  string inputStr(str);
  vector<string> lines;
  tokenize(inputStr, lines, "\n");

  map<int, pair<int, int>> ret;

  // Do not parse the first two lines and the last line
  for (size_t i=1; i < lines.size()-1; ++i)
  {
    string     kid;
    string     dev;
    string     gid;
    size_t     nmatch = 4;
    regmatch_t pmatch[4];

    int regRet = regexec(&reKERNELS_ID, lines[i].c_str(), nmatch, pmatch, 0);
    if (regRet)
      return ret;

    kid = lines[i].substr(pmatch[1].rm_so, pmatch[1].rm_eo - pmatch[1].rm_so);
    dev = lines[i].substr(pmatch[2].rm_so, pmatch[2].rm_eo - pmatch[2].rm_so);
    gid = lines[i].substr(pmatch[3].rm_so, pmatch[3].rm_eo - pmatch[3].rm_so);
    ret[atoi(kid.c_str())] = make_pair(atoi(dev.c_str()), atoi(gid.c_str()));
  }

  return ret;
}


/**
 * This function parses the following command in gdb:
 * ----------------------------------------------------------------------------
 * (gdb) info cuda contexts
 *   Context              Dev State
 *   0x00001000080038f0   0   active 
 *   0x00001000100038f0   1   active 
 *   0x00001000140038f0   2   active 
 * * 0x00001000180038f0   3   active 
 * ----------------------------------------------------------------------------
 *
 * It returns a map of device ID (int) -> Cuda Context ID (uint64_t)
 */
map<int, uint64_t> StringParser::matchCudaContextsInfo(const char *str) const
{
  map<int, uint64_t> ret;

  // split by \n (tokenize by lines)
  string inputStr(str);
  vector<string> lines;
  tokenize(inputStr, lines, "\n");

  // Do not parse the first and the last lines
  for (size_t i=1; i < lines.size()-1; ++i)
  {
    char ctx[64]; // long enough to hold a cuda context
    matchAddressValue(lines[i].c_str(), ctx);
    if (strlen(ctx) == 0)
      return ret;

    string     device_id;
    size_t     nmatch = 2;
    regmatch_t pmatch[2];

    int regRet = regexec(&reCONTEXT_ID, lines[i].c_str(), nmatch, pmatch, 0);
    if (regRet)
      return ret;

    device_id = lines[i].substr(pmatch[1].rm_so, pmatch[1].rm_eo - pmatch[1].rm_so);
    ret[atoi(device_id.c_str())] = strtoll(string(ctx).c_str(), NULL, 0);
  }

  return ret;
}

/**
 * This function parses the following command in gdb:
 * ----------------------------------------------------------------------------
 * (gdb) info threads
 *   Id   Target Id         Frame
 *   4    Thread 0x2aaaaba87700 (LWP 45661) "target" 0x00002aaaab19aa3d in nanosleep () from /lib64/libc.so.6
 *   3    Thread 0x2aaaab886700 (LWP 45660) "target" 0x00002aaaab19aa3d in nanosleep () from /lib64/libc.so.6
 *   2    Thread 0x2aaaab685700 (LWP 45659) "target" 0x00002aaaab19aa3d in nanosleep () from /lib64/libc.so.6
 * * 1    Thread 0x2aaaab483040 (LWP 45655) "target" 0x00002aaaab19aa3d in nanosleep () from /lib64/libc.so.6
 * ----------------------------------------------------------------------------
 *
 * It returns a vector of pairs containing the gdb Id (unsigned int) and the
 * thread Id (first string that contains a hex value) for each thread.
 * A pair, for example, is: <4, "0x2aaaaba87700">.
 */
vector<StringParser::ThreadID> StringParser::matchThreadsInfo(const char *str) const
{
  vector<StringParser::ThreadID> ret;

  // split by \n (tokenize by lines)
  string inputStr(str);
  vector<string> lines;
  tokenize(inputStr, lines, "\n");

  // Do not parse the first and the last lines
  for (size_t i=1; i < lines.size()-1; ++i)
  {
    char addr[64]; // long enough to hold an address
    matchAddressValue(lines[i].c_str(), addr);
    if (strlen(addr) == 0)
      return ret;
    //ret.push_back( string(addr) );

    // Match thread ID
    string id;
    size_t     nmatch = 1;
    regmatch_t pmatch[1];
    int regRet = regexec(&reTHREAD_ID, lines[i].c_str(), nmatch, pmatch, 0);
    if (!regRet)
    {
      int size =  pmatch[0].rm_eo - pmatch[0].rm_so;
      char IDStr[size+1];
      IDStr[0] = '\0';
      strncpy (IDStr, (lines[i].c_str() + pmatch[0].rm_so), size);

      vector<string> t;
      tokenize(IDStr, t, " \t");
      id = t.size() == 1 ? t[0] : t[1];
    }
    else
      return ret;

    ret.push_back(
        ThreadID(static_cast<unsigned int>(atoi(id.c_str())), strtoll(string(addr).c_str(), NULL, 0) ));
  }

  return ret;
}

int StringParser::matchThreadID(const char *str) const
{
  size_t     nmatch = 1;
  regmatch_t pmatch[1];

  int ret = regexec(&reTHREAD, str, nmatch, pmatch, 0);
  if (ret) // if thread not found
  {
    return -1;
  }

  return atoi(str + pmatch[0].rm_so);
}

/*
 * Eliminates the GDB prompt from input string.
 * It assumes that the the string contains lines (separated by '\n')
 * and that GDB prompt is in the last line.
 */
void StringParser::eliminateGDBPrompt(char *newStr, const char *oldStr) const
{
  assert(oldStr && "Invalid input string");
  size_t s = strlen(oldStr);
  if (s==0) // if empty string, just return an empty string
  {
    newStr[0] = '\0';
    return;
  }

  // Iterate from end to begin, and find the first '\n' char
  size_t end = 0;
  for (long long i=(static_cast<long long>(s)-1); i >= 0; --i)
  {
    if (oldStr[i] == '\n')
    {
      end = i;
      break;
    }
  }

  // Couldn't find a '\n' char; it means the string contains a single line.
  // Thus we eliminate this line.
  if (end == 0)
  {
    newStr[0] = '\0';
    return;
  }

  strncpy(newStr, oldStr, end+1);
  newStr[end+1] = '\0';
}

void StringParser::eliminateGDBPromptInplace(string &input) const
{
  size_t s = input.size();
  if (s==0) // if empty string, just return
    return;

  // Iterate from end to begin, and find the first '\n' char
  size_t end = 0;
  for (long long i=(static_cast<long long>(s)-1); i >= 0; --i)
  {
    if (input[i] == '\n')
    {
      end = i;
      break;
    }
  }

  // Couldn't find a '\n' char; it means the string contains a single line.
  // Thus we eliminate this line.
  if (end == 0)
  {
    input.resize(0);
    return;
  }

  input.resize(end+1);
}

/******************************************************************************
 * String utilities
 */

void ompd_gdb::tokenize(const std::string &str,
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

