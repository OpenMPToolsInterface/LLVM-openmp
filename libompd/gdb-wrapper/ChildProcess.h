/*
 * ChildProcess.h
 *
 *  Created on: Jul 19, 2016
 *      Author: Ignacio Laguna
 *     Contact: ilaguna@llnl.gov
 */
#ifndef OMPD_GDB_CHILDPROCESS_H_
#define OMPD_GDB_CHILDPROCESS_H_

#include <cstddef>

namespace ompd_gdb {

class ChildProcess {
private:
    //Pipe writePipe;
    //Pipe readPipe;
  int fd1[2];
  int fd2[2];
public:
    int childPid = -1;
    //std::unique_ptr<stdio_filebuf<char>> writeBuf = nullptr;
    //std::unique_ptr<stdio_filebuf<char>> readBuf = nullptr;
    //std::ostream stdin;
    //std::istream stdout;

    //ChildProcess(char* const argv[]);
    ChildProcess(const char **argv);

    /** Read and write some characters **/
    std::size_t readSome(char *str, std::size_t s);
    void writeSome(const char *str);

    /** Close child pipe */
    void sendEOF();

    /** Send child a signal */
    void sendSignal(int sig);

    /** Wait for child to finish */
    int wait();
};

}



#endif /* OMPD_GDB_CHILDPROCESS_H_ */
