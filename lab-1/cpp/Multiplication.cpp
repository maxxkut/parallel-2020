#pragma once

#include "Matrix.cpp"

class Multiplication {
public:
    Matrix *A;
    Matrix *B;

    Multiplication(Matrix *a, Matrix *b) : A(a), B(b) {}

    virtual Matrix *execute() = 0;

    virtual int computeElement(int row, int column) {
        int result = 0;
        for (int i = 0; i < A->width; i++) {
            result += A->matrixValues[row][i] * B->matrixValues[i][column];
        }
        return result;
    }

    Matrix *toMatrix(int **values) {
        auto matrix = new Matrix();
        matrix->height = A->height;
        matrix->width = B->width;
        matrix->matrixValues = values;
        return matrix;
    }
};

class SerialMultiplication : public Multiplication {
public:
    SerialMultiplication(Matrix *a, Matrix *b) : Multiplication(a, b) {}

    Matrix *execute() override {
        int height = A->height;
        int width = B->width;
        int **result = new int *[height];
        for (int row = 0; row < height; row++) {
            result[row] = new int[width];
            for (int column = 0; column < width; column++) {
                result[row][column] = computeElement(row, column);
            }
        }
        return toMatrix(result);
    }
};

class NativeParallelMultiplication : public Multiplication {
public:
    NativeParallelMultiplication(Matrix *a, Matrix *b) : Multiplication(a, b) {}

    Matrix *execute() override {
        int rows = A->height;
        int columns = B->width;
        int **result = new int *[rows];

#pragma omp parallel for schedule(static)
        for (int row = 0; row < rows; row++) {
            result[row] = new int[columns];
            for (int column = 0; column < columns; column++) {
                result[row][column] = computeElement(row, column);
            }
        }

        return toMatrix(result);
    }
};

class ParallelStaticSchedule : public Multiplication {
public:
    ParallelStaticSchedule(Matrix *a, Matrix *b) : Multiplication(a, b) {}

    Matrix *execute() override {
        int rows = A->height;
        int columns = B->width;

        int **result = new int *[rows];
#pragma omp parallel for schedule(static)
        for (int row = 0; row < rows; row++) {
            result[row] = new int[columns];
        }

        int tasks = rows * columns;
#pragma omp parallel for schedule(static)
        for (int task = 0; task < tasks; task++) {
            int row = task / columns;
            int column = task % columns;
            result[row][column] = computeElement(row, column);
        }

        return toMatrix(result);
    }
};

class ParallelDynamicSchedule : public Multiplication {
public:
    ParallelDynamicSchedule(Matrix *a, Matrix *b) : Multiplication(a, b) {}

    Matrix *execute() override {
        int rows = A->height;
        int columns = B->width;

        int **result = new int *[rows];
#pragma omp parallel for schedule(static)
        for (int row = 0; row < rows; row++) {
            result[row] = new int[columns];
        }

        int tasks = rows * columns;
#pragma omp parallel for schedule(dynamic, 1024)
        for (int task = 0; task < tasks; task++) {
            int row = task / columns;
            int column = task % columns;
            result[row][column] = computeElement(row, column);
        }

        return toMatrix(result);
    }
};

class ParallelTasksGuidedScheduleMultiplication : public Multiplication {
public:
    ParallelTasksGuidedScheduleMultiplication(Matrix *a, Matrix *b) : Multiplication(a, b) {}

    Matrix *execute() override {
        int rows = A->height;
        int columns = B->width;

        int **result = new int *[rows];
#pragma omp parallel for schedule(static)
        for (int row = 0; row < rows; row++) {
            result[row] = new int[columns];
        }

        int tasks = rows * columns;
#pragma omp parallel for schedule(guided, 1024)
        for (int task = 0; task < tasks; task++) {
            int row = task / columns;
            int column = task % columns;
            result[row][column] = computeElement(row, column);
        }

        return toMatrix(result);
    }
};
