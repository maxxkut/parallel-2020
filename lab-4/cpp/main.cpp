#include "iostream"
#include <string>
#include <chrono>
#include "Dijkstra.cpp"
#include <omp.h>
using namespace std;
using namespace std::chrono;

int main(int argc, char **argv)
{
    if (argc != 3) {
        return 1;
    }

    try
	{
        auto mode = atoi(argv[1]);
        auto threads = atoi(argv[2]);
		Dijkstra graph;
		if (threads != -1) {
		    omp_set_dynamic(0);
		    omp_set_num_threads(8);
		}
        auto start = high_resolution_clock::now();
        switch (mode) {
            case 0:
                graph.sequentalDijkstra();
                break;
            case 1:
                graph.parallelStaticDijkstra();
                break;
            case 2:
                graph.parallelDynamicDijkstra();
                break;
            default:
                return -1;
        }
        auto finish = high_resolution_clock::now();
        graph.writeResult();

        cout << duration<double>(finish - start).count() << endl;
	}
	catch (exception &ex)
	{
		return -1;
	}
		
	return 0;
}
