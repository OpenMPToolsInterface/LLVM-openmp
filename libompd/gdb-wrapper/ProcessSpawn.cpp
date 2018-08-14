/*
 * ProcessSpawn.cpp
 *
 *  Created on: Dec 17, 2014
 *      Author: Ignacio Laguna
 *     Contact: ilaguna@llnl.gov
 */

#include "ProcessSpawn.h"

#include <ext/stdio_filebuf.h>
#include <sys/wait.h>
#include <unistd.h>
#include <assert.h>
#include <iostream>
#include <memory>
#include <exception>
#include <stdexcept>
#include <sys/types.h>
#include <signal.h>
#include <stdio.h>

using namespace ompd_gdb;
using namespace __gnu_cxx;

/*
 * ----------------------------------------------------------------------------
 * Pipe class methods
 * ----------------------------------------------------------------------------
 */
Pipe::Pipe()
{
	if (pipe(fd))
		throw std::runtime_error("Couldn't create a pipe!");
}

const int Pipe::readFd() const
{
	return fd[0];
}

const int Pipe::writeFd() const
{
	return fd[1];
}


void Pipe::close()
{
	::close(fd[0]);
	::close(fd[1]);
}

Pipe::~Pipe()
{
	close();
}

/*
 * ----------------------------------------------------------------------------
 * ProcessSpawn class methods
 * ----------------------------------------------------------------------------
 */
ProcessSpawn::ProcessSpawn(const char* const argv[]): stdin(NULL), stdout(NULL)
{
	childPid = fork();
	if (childPid == -1)
		throw std::runtime_error("Couldn't start child process!");

	if (childPid == 0) // Child process
	{
		dup2(writePipe.readFd(), STDIN_FILENO);
		dup2(readPipe.writeFd(), STDOUT_FILENO);
		dup2(readPipe.writeFd(), STDERR_FILENO);
		writePipe.close();
		readPipe.close();

		int result = execv(argv[0], const_cast<char* const*>(argv));
		// on successful exec we are not here anymore, so something went wrong
		printf( "ERROR: could not start program '%s' return code %i\n", argv[0] , result);
		exit(EXIT_FAILURE);
	}
	else // Parent process
	{
		close(writePipe.readFd());
		close(readPipe.writeFd());
		writeBuf = std::unique_ptr<stdio_filebuf<char>>
				(new stdio_filebuf<char>(writePipe.writeFd(), std::ios::out));
		readBuf = std::unique_ptr<stdio_filebuf<char>>
				(new stdio_filebuf<char>(readPipe.readFd(), std::ios::in));
		stdin.rdbuf(writeBuf.get());
		stdout.rdbuf(readBuf.get());
	}
}


void ProcessSpawn::sendEOF()
{
	writeBuf->close();
}


void ProcessSpawn::sendSignal(int signal)
{
	kill(childPid, signal);
}


int ProcessSpawn::wait()
{
	int status;
	waitpid(childPid, &status, 0);
	return status;
}

