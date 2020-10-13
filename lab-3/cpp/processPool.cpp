#pragma once
#include <mpi.h>

class ProcessPool {
public:
    MPI_Comm communicator;
    int currentProcess;
    int totalProcesses;

    ProcessPool(MPI_Comm communicator) : communicator(communicator) {
        MPI_Comm_rank(communicator, &currentProcess);
        MPI_Comm_size(communicator, &totalProcesses);
    }

    bool leftHalf() {
        return currentProcess < totalProcesses / 2;
    }

};
