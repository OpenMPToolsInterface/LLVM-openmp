/*
 * StringParser.h
 *
 *  Created on: Dec 26, 2014
 *      Author: Ignacio Laguna
 *     Contact: ilaguna@llnl.gov
 */
#ifndef GDB_STRINGPARSER_H_
#define GDB_STRINGPARSER_H_

#include <regex.h>
#include <string>
#include <vector>
#include <map>
#include "CudaGdb.h"

namespace ompd_gdb {

/*******************************************************************************
 * This class implements regular expressions to parse GDB output.
 * Member functions are useful in parsing commands from users and the output of
 * GDB. It assumes that GDB has a particular prompt.
 *
 * It is also useful in parsing utility commands to make use of OMPD. These
 * commands are defined by: "ompd COMMAND".
 */

class StringParser
{
private:
  regex_t reQUIT;
  regex_t rePROMPT;
  regex_t reOMPD_COMMAND;
  regex_t reREGULAR_VALUE;
  regex_t reADDRESS_VALUE;
  regex_t reMEMORY_VALUES;
  regex_t reTHREAD_ID;
  regex_t reCONTEXT_ID;
  regex_t reKERNELS_ID;
	regex_t	reBLOCKS_ID;
  regex_t reTHREAD;

  bool regexpMatches(const char *str, const regex_t *re) const;

public:

#if defined(CUDA_GDB_PATH)
  static constexpr const char *GDB_PROMPT = "(cuda-gdb) ";
#else
  static constexpr const char *GDB_PROMPT = "(gdb) ";
#endif

  StringParser();

  /**
   * Return true if string has an OMPD command of the form: "ompd COMMAND".
   */
  bool isOMPDCommand(const char *str) const;

  /**
   * Returns true if the "quit" command is in the string.
   */
  bool isQuitCommand(const char *str) const;

  /**
   * Return true if the string contains the gdb prompt.
   */
  bool hasGDBPrompt(const char *str) const;

  /**
   * Eliminate GBD prompt (i.e., (gdb)) from string
   */
  void eliminateGDBPrompt(char *newStr, const char *oldStr) const;
  void eliminateGDBPromptInplace(std::string &input) const;

  /**
   * Matches values given by GDB of the form:
   *    $[digit] = [alphanumeric]
   * where the second elements (alphanumeric) is returned.
   */
  void matchRegularValue(const char *str, char *value) const;

  /**
   * Matches values given by GDB of the form:
   *    text [address] text
   * where [address] is a memory address in hex format
   */
  void matchAddressValue(const char *str, char *addr) const;

  /**
   * Matches GDB output from command:
   *    x/Nxb, where N is a number of bytes
   * Returns a vector of bytes in hex format: "0x00".
   */
  std::vector<std::string> matchMemoryValues(const char *str) const;

  /**
   * Matches GDB output from the command:
   *    "info cuda threads"
   */
  std::vector<CudaThread> matchCudaThreadsInfo(uint64_t ctx, uint64_t dev, 
          uint64_t kernel, uint64_t grid, const char *str) const;

  /**
   * Matches GDB output from the command:
   *    "info cuda kernels"
   * Returns cuda context IDs (one per device)
   */
  std::map<int, std::pair<int,int>> matchCudaKernelsInfo(const char *str) const;

  /**
   * Matches GDB output from the command:
   *    "info cuda contexts"
   * Returns cuda context IDs (one per device)
   */
  std::map<int, uint64_t> matchCudaContextsInfo(const char *str) const;

  typedef std::pair<unsigned int, uint64_t> ThreadID;

  /**
   * Matches GDB output from the command:
   *    "info threads"
   * Returns system thread IDs
   */
  std::vector<ThreadID> matchThreadsInfo(const char *str) const;

  /**
   * Matches GDB output from the command:
   *    "thread"
   * Returns the ID of the currently selected thread
   */
  int matchThreadID(const char *str) const;
};

/******************************************************************************
 * String utilities
 */

/**
 * Tokenize a string
 */
void tokenize(const std::string &str,
    std::vector<std::string> &tokens,
    const std::string &delimiters);

}

#endif /* GDB_STRINGPARSER_H_ */
