#pragma once

#include <fstream>
#include <vector>
#include <math.h>
using namespace std;

const int PRIMARY_PROCESS = 0;

class Jacobi {
private:

    std::vector<double> A;
    std::vector<double> b;
    std::vector<double> x;

    std::vector<double> tempA;

    int systemRank;
    double precision;

    std::vector<double> xPrev;

    int currentProcess;
    int totalProcesses;

    vector<int> offsets;
    vector<int> batchSizes;
    vector<int> scaledBatchSize;
    vector<int> scaledOffset;
    int offset;
    int batchSize;

public:
    Jacobi() {
        MPI_Comm_rank(MPI_COMM_WORLD, &currentProcess);
        MPI_Comm_size(MPI_COMM_WORLD, &totalProcesses);
    }

    void process_params(){
        if (currentProcess == PRIMARY_PROCESS) {
            readParameters();
        }
    }

    void init() {

        offsets.resize(totalProcesses);
        batchSizes.resize(totalProcesses);

        scaledBatchSize.resize(totalProcesses);
        scaledOffset.resize(totalProcesses);
        MPI_Bcast(&systemRank, 1, MPI_INT, PRIMARY_PROCESS, MPI_COMM_WORLD);
        MPI_Bcast(&precision, 1, MPI_DOUBLE, PRIMARY_PROCESS, MPI_COMM_WORLD);
        if (currentProcess == PRIMARY_PROCESS) {
            init_offsets();
        } else {
            MPI_Status status;
            MPI_Recv(&offset, 1, MPI_INT, PRIMARY_PROCESS, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            MPI_Recv(&batchSize, 1, MPI_INT, PRIMARY_PROCESS, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        }
        MPI_Bcast(offsets.data(), offsets.size(), MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(batchSizes.data(), batchSizes.size(), MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(scaledOffset.data(), scaledOffset.size(), MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(scaledBatchSize.data(), scaledBatchSize.size(), MPI_INT, 0, MPI_COMM_WORLD);
        A.resize(batchSize * systemRank);
        if (currentProcess != PRIMARY_PROCESS) {
            b.resize(systemRank);
            x.resize(systemRank);
        }
        MPI_Bcast(b.data(), systemRank, MPI_DOUBLE, PRIMARY_PROCESS, MPI_COMM_WORLD);
        MPI_Bcast(x.data(), systemRank, MPI_DOUBLE, PRIMARY_PROCESS, MPI_COMM_WORLD);

        MPI_Scatterv(tempA.data(), scaledBatchSize.data(), scaledOffset.data(), MPI_DOUBLE,
                     A.data(), scaledBatchSize.at(currentProcess), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    void init_offsets() {
        int partSize = systemRank / totalProcesses;
        int oddProcesses = systemRank % totalProcesses;
        for (int process = 0; process < totalProcesses; process++) {
            int processBatchSize = (process < oddProcesses) ? partSize + 1 : partSize;
            int processOffset = 0;
            if (process > 0) {
                processOffset = offsets[process - 1] + processBatchSize;
            }

            offsets[process] = processOffset;
            scaledOffset[process] = processOffset * systemRank;
            scaledBatchSize[process] = processBatchSize * systemRank;
            batchSizes[process] = processBatchSize;
            if (process == PRIMARY_PROCESS) {
                this->offset = processOffset;
                this->batchSize = processBatchSize;

            } else {
                MPI_Send(&processOffset, 1, MPI_INT, process, 0, MPI_COMM_WORLD);
                MPI_Send(&processBatchSize, 1, MPI_INT, process, 0, MPI_COMM_WORLD);
            }
        }
    };

    void computeIteration() {
        computeStep();
    }

    bool precisionReached() {
        if (xPrev.empty()) {
            return false;
        }
        for (int i = 0; i < systemRank; i++) {
            if (precision < fabs(x[i] - xPrev[i])) {
                return false;
            }
        }
        return true;
    }

    void outputResult() {
        if (currentProcess == PRIMARY_PROCESS) {
            ofstream stream("linear.output");
            for (auto i = 0; i < systemRank; i++) {
                stream << x[i] << " ";
            }
            stream.close();
        }
    }


private:

    void readParameters() {
        ifstream stream("linear.input");
        stream >> systemRank >> precision;
        x.reserve(systemRank);
        b.reserve(systemRank);
        tempA.reserve(systemRank * systemRank);
        double foo = 0;
        for (auto i = 0; i < systemRank; i++) {
            stream >> foo;
            x.push_back(foo);
        }
        for (auto i = 0; i < systemRank; i++) {
            stream >> foo;
            b.push_back(foo);
        }
        for (auto i = 0; i < systemRank; i++) {
            for (auto j = 0; j < systemRank; j++) {
                stream >> foo;
                tempA.push_back(foo);
            }
        }
        stream.close();
    }

    void computeStep() {
        xPrev.resize(0);
        for (int i = 0; i < systemRank; i++) {
            xPrev.push_back(x[i]);
        }
        for (int i = 0; i < batchSize; i++) {
            double sum = 0;
            auto diagonal_element = A[i * systemRank + i + offset];
            for (int j = 0; j < systemRank; j++) {
                if (i + offset != j) {
                    sum += A[i * systemRank + j] * xPrev[j];
                }
            }
            x[i + offset] = (b[i + offset] - sum) / diagonal_element;
        }

        MPI_Allgatherv(x.data() + offset, this->batchSize, MPI_DOUBLE,
                       x.data(), this->batchSizes.data(), this->offsets.data(), MPI_DOUBLE, MPI_COMM_WORLD);


    }
};