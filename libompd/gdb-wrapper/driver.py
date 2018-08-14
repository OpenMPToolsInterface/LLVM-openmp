#!/usr/bin/env python

import sys
import os
from time import sleep
from subprocess import Popen

def main():
    program = sys.argv[1]
#    os.environ['LD_LIBRARY_PATH'] = "/Users/lagunaperalt1/projects/OMPD/ompd_code/src"
#    os.environ['LD_LIBRARY_PATH'] = "/g/g90/laguna/projects/OMPD/ompd_code/OMPD/src"

    p = Popen(program)
    pid = p.pid
    print "Process ID of test:", pid
    print "Waiting a few seconds before attaching GDB..."
    sleep(2)
    
    # Run gdb wrapper
    gdb = Popen(["./odb", str(pid)])

    # Wait until programs end
    gdb.communicate()
    p.terminate()

main()
