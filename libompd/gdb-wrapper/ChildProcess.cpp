/*
 * ChildProcess.cpp
 *
 *  Created on: Jul 19, 2016
 *      Author: Ignacio Laguna
 *     Contact: ilaguna@llnl.gov
 */

#include "ChildProcess.h"

#include <sys/wait.h>
#include <unistd.h>
#include <assert.h>
#include <signal.h>
#include <string.h>
#include <iostream>

using namespace std;
using namespace ompd_gdb;

ChildProcess::ChildProcess(const char **argv)
{
  if ( (pipe(fd1) < 0) || (pipe(fd2) < 0) )
      cerr << "ERROR: pipe error\n";

  if ( (childPid = fork()) < 0 )
      cerr << "ERROR: fork error\n";

  else  if (childPid == 0) // Child process
  {
      close(fd1[1]);
      close(fd2[0]);

      if (fd1[0] != STDIN_FILENO)
      {
          if (dup2(fd1[0], STDIN_FILENO) != STDIN_FILENO)
            cerr << "ERROR: dup2 error to stdin\n" << endl;
          close(fd1[0]);
      }

      if (fd2[1] != STDOUT_FILENO)
      {
          if (dup2(fd2[1], STDOUT_FILENO) != STDOUT_FILENO)
              cerr << "ERROR: dup2 error to stdout\n" << endl;
          close(fd2[1]);
      }

      //int result = execv(argv[0], const_cast<char* const*>(argv));
      int result = execv((const char*)argv[0], (char * const *)argv);

      // On successful execution we should not see the following message
      printf( "ERROR: could not start program '%s' return code %i\n",
          argv[0] , result);
      exit(EXIT_FAILURE);
  }
  else // Parent process
  {
      close(fd1[0]);
      close(fd2[1]);
  }
}

std::size_t ChildProcess::readSome(char *str, std::size_t s)
{
  ssize_t rv;
  if ( (rv = read(fd2[0], str, s)) < 0 )
    cerr << "ERROR: read error from pipe" << endl;

  return rv;
}

// FIXME
// Call write in a loop?
void ChildProcess::writeSome(const char *str)
{
  if ( write(fd1[1], str, strlen(str) ) != strlen(str) )
    cerr << "ERROR: Write error from pipe" << endl;
}

/** Close child pipe */
void ChildProcess::sendEOF()
{
  kill(childPid, SIGKILL);
  wait();
  int c1 = close(fd1[1]);
  int c2 = close(fd2[0]);
  if (!c1 || !c2)
    cerr << "ERROR: closing pipes to gdb process failed.\n";
}

/** Send child a signal */
void ChildProcess::sendSignal(int sig)
{
  kill(childPid, sig);
}

/** Wait for child to finish */
int ChildProcess::wait()
{
  int status;
  waitpid(childPid, &status, 0);
  //cout << "Status " << status << "\n";
  return status;
}
