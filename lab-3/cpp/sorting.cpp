#pragma once

#include "sortArray.cpp"
#include "processPool.cpp"
#include <mpi.h>

using namespace std;

const int PRIMARY_PROCESS = 0;
const int MAX_SIZE = 10000000;

class Sorter {
private:
    int currentPivot = 0;
    int pivotType = 0;
    splittableArray *currentArray = nullptr;

public:
    ProcessPool *globalGroup;
    ProcessPool *group;

    Sorter(int pivotType) : pivotType(pivotType) {
        globalGroup = new ProcessPool(MPI_COMM_WORLD);
        group = globalGroup;
    }

    void initialize(splittableArray *array) {
        if (globalGroup->currentProcess == PRIMARY_PROCESS) {
            MPI_Request request;
            for (int process = 0; process < globalGroup->totalProcesses; process++) {
                auto part = array->getPart(process, globalGroup->totalProcesses);
                MPI_Isend(part->content, part->size, MPI_INT, process, 0, MPI_COMM_WORLD, &request);
            }
        }

        MPI_Status status;
        int *buffer = new int[MAX_SIZE];
        MPI_Recv(buffer, MAX_SIZE, MPI_INT, PRIMARY_PROCESS, 0, MPI_COMM_WORLD, &status);

        int size;
        MPI_Get_count(&status, MPI_INT, &size);
        currentArray = new splittableArray(buffer, size, pivotType);

    }

    void pivot() {
        if (group->currentProcess == PRIMARY_PROCESS) {
            currentPivot = currentArray->pivot();
        }
        MPI_Bcast(&currentPivot, 1, MPI_INT, PRIMARY_PROCESS, group->communicator);
    }

    void exchange() {
        int partner;
        if (group->leftHalf()) {
            partner = group->currentProcess + group->totalProcesses / 2;
        } else {
            partner = group->currentProcess - group->totalProcesses / 2;
        }

        splittableArray *low, *high;
        currentArray->partition(currentPivot, &low, &high);

        MPI_Request request;
        if (group->leftHalf()) {
            MPI_Isend(high->content, high->size, MPI_INT, partner, 0, group->communicator, &request);
        } else {
            MPI_Isend(low->content, low->size, MPI_INT, partner, 0, group->communicator, &request);
        }

        MPI_Status status;
        int *buffer = new int[MAX_SIZE];
        MPI_Recv(buffer, MAX_SIZE, MPI_INT, partner, 0, group->communicator, &status);

        int size;
        MPI_Get_count(&status, MPI_INT, &size);
        if (group->leftHalf()) {
            high = new splittableArray(buffer, size, pivotType);
        } else {
            low = new splittableArray(buffer, size, pivotType);
        }

        currentArray = new splittableArray(low, high);

    }

    void regroup() {
        int newGroup = group->leftHalf() ? 0 : 1;
        MPI_Comm newCommunicator;
        MPI_Comm_split(group->communicator, newGroup, 0, &newCommunicator);
        group = new ProcessPool(newCommunicator);
    }

    void sort() {
        qsort(currentArray->content, currentArray->size, sizeof(int), compareIntegers);
    }

    splittableArray *collect() {
        MPI_Request req;
        MPI_Isend(currentArray->content, currentArray->size, MPI_INT, PRIMARY_PROCESS, 0, MPI_COMM_WORLD, &req);

        if (globalGroup->currentProcess == PRIMARY_PROCESS) {
            splittableArray *result = nullptr;
            for (int process = 0; process < globalGroup->totalProcesses; process++) {
                MPI_Status status;
                int *buffer = new int[MAX_SIZE];
                MPI_Recv(buffer, MAX_SIZE, MPI_INT, process, 0, MPI_COMM_WORLD, &status);

                int size;
                MPI_Get_count(&status, MPI_INT, &size);

                auto *array = new splittableArray(buffer, size, pivotType);
                if (result == nullptr) {
                    result = array;
                } else {
                    result = new splittableArray(result, array);
                }
            }
            return result;
        }

        return nullptr;
    }

private:

    static int compareIntegers(const void *a, const void *b) {
        int int1 = *((int *) a);
        int int2 = *((int *) b);
        return int1 - int2;
    }
};
