#include <mpi.h>
#include "Jacobi.cpp"
#include <chrono>
#include "iostream"

using namespace std::chrono;

int main() {
    MPI_Init(nullptr, nullptr);

    Jacobi jacobi;
    jacobi.process_params();

    auto start = high_resolution_clock::now();

    jacobi.init();

    for (int iteration = 0; iteration < 1000 && !jacobi.precisionReached(); iteration++) {
        jacobi.computeIteration();
    }

    auto finish = high_resolution_clock::now();
    cout << duration<double>(finish - start).count() << endl;

    jacobi.outputResult();

    MPI_Finalize();
    return jacobi.precisionReached() ? 0 : -1;
}
