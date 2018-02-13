/*
 * ProcessSpawn.h
 *
 *  Created on: Dec 17, 2014
 *      Author: Ignacio Laguna
 *     Contact: ilaguna@llnl.gov
 */

#ifndef PROCESSSPAWN_H_
#define PROCESSSPAWN_H_

#include <ext/stdio_filebuf.h>
#include <sys/wait.h>
#include <unistd.h>
#include <assert.h>
#include <iostream>
#include <memory>
#include <exception>
#include <stdexcept>

namespace ompd_gdb {
using namespace __gnu_cxx;

/**
 * Implements methods to handle a pipe: reading, writing and closing.
 */
class Pipe {
private:
    int fd[2];
public:
    Pipe();
    const int readFd() const;
    const int writeFd() const;
    void close();
    ~Pipe();
};

/**
 * Spawns a process (a child) and implements methods to write and read from a
 * pipe that is used to communicate with the child process.
 */
class ProcessSpawn {
private:
    Pipe writePipe;
    Pipe readPipe;
public:
    int childPid = -1;
    std::unique_ptr<stdio_filebuf<char>> writeBuf = nullptr;
    std::unique_ptr<stdio_filebuf<char>> readBuf = nullptr;
    std::ostream stdin;
    std::istream stdout;

    ProcessSpawn(const char* const argv[]);

    /** Close child pipe */
    void sendEOF();

    /** Send child a signal */
    void sendSignal(int sig);

    /** Wait for child to finish */
    int wait();
};

}

#endif /* PROCESSSPAWN_H_ */
