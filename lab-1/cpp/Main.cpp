#include <chrono>
#include "Matrix.cpp"
#include "Multiplication.cpp"

using namespace std;
using namespace std::chrono;

void executeMultiplication(int height, int width, int bound, int mode) {
    auto A = new Matrix(height, bound);
    auto B = new Matrix(bound, width);

    Multiplication *multiplication;
    switch (mode) {
        case 1:
            multiplication = new NativeParallelMultiplication(A, B);
            break;
        case 2:
            multiplication = new ParallelStaticSchedule(A, B);
            break;
        case 3:
            multiplication = new ParallelDynamicSchedule(A, B);
            break;
        case 4:
            multiplication = new ParallelTasksGuidedScheduleMultiplication(A, B);
            break;
        default:
            multiplication = new SerialMultiplication(A, B);
    }

    auto start = high_resolution_clock::now();
    multiplication->execute();
    auto finish = high_resolution_clock::now();
    cout << duration<double>(finish - start).count() << endl;
}

int main(int argc, char **argv) {
    srand(time(nullptr));
    if (argc != 5) {
        return 1;
    }
    executeMultiplication(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4]));
    return 0;
}
