#include <fstream>
#include <mpi.h>
#include "sorting.cpp"
#include <chrono>
#include "iostream"

using namespace std;
using namespace std::chrono;

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int pivotType = atoi(argv[1]);

    Sorter quicksort(pivotType);

    if (quicksort.globalGroup->currentProcess == PRIMARY_PROCESS) {
        ifstream stream("quicksort.input");
        int *content, size;
        stream >> size;
        content = new int[size];
        for (auto i = 0; i < size; i++) {
            stream >> content[i];
        }
        stream.close();
        quicksort.initialize(new splittableArray(content, size, pivotType));
    } else {
        quicksort.initialize(nullptr);
    }

    auto start = high_resolution_clock::now();
    while (true) {
        if (quicksort.group->totalProcesses == 1) {
            quicksort.sort();
            break;
        }
        quicksort.pivot();
        quicksort.exchange();
        quicksort.regroup();
    }

    splittableArray *array;
    if (quicksort.globalGroup->currentProcess == PRIMARY_PROCESS) {
        array = quicksort.collect();
    } else {
        quicksort.collect();
    }
    auto finish = high_resolution_clock::now();
    cout << duration<double>(finish - start).count() << endl;

    if (quicksort.globalGroup->currentProcess == PRIMARY_PROCESS) {
        ofstream stream("quicksort.output");
        stream << array->size << endl;
        for (auto i = 0; i < array->size; i++) {
            stream << array->content[i] << " ";
        }
        stream.close();
    }

    MPI_Finalize();
    return 0;
}