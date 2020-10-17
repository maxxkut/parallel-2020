#include <omp.h>
#include "vector"
#include <algorithm>
#include <fstream>

class Dijkstra {
public:

    std::vector<std::vector<int>> graph;
    std::vector<int> distances;
    int size;
    int start;

    Dijkstra() {
        std::ifstream in;
        in.open("dijkstra.input");

        in >> this->size;
        in >> this->start;
        auto foo = 0;
        this->graph.resize(size);
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                in >> foo;
                this->graph[i].push_back(foo);
            }
        }

        in.close();
        this->distances.resize(size);
    }

    void writeResult() {
        std::ofstream out;
        out.open("dijkstra.output");

        for (auto i = 0; i < distances.size(); i++) {
            if (this->distances[i] == std::numeric_limits<int>::max())
                out << "inf ";
            else
                out << this->distances[i] << " ";
        }

        out.close();
    }

    int min_distance(const std::vector<int> &dist, const std::vector<bool> &used_set) {
        int min_value = std::numeric_limits<int>::max();
        int min_index = 0;

        for (int v = 0; v < dist.size(); ++v) {
            if (!used_set[v] && dist[v] <= min_value) {
                min_value = dist[v];
                min_index = v;
            }
        }
        return min_index;
    }

    void sequentalDijkstra() {
        std::vector<int> distances(size, std::numeric_limits<int>::max());
        std::vector<bool> used(size);
        distances[start] = 0;

        for (int step = 0; step < size - 1; ++step) {
            int from = min_distance(distances, used);
            if (distances[from] == std::numeric_limits<int>::max()) {
                break;
            }
            used[from] = true;

            for (auto to = 0; to < this->graph[from].size(); to++) {

                auto weight = this->graph[from][to];
                if (weight == 0) { continue; }

                auto relaxed_dist = distances[from] + weight;
                if (!used[to] && relaxed_dist < distances[to])
                    distances[to] = relaxed_dist;
            }
        }

        std::copy(distances.begin(), distances.end(), this->distances.begin());

    }

    void parallelDynamicDijkstra() {
        std::vector<int> distances(size, std::numeric_limits<int>::max());
        std::vector<bool> used(size);
        auto batch = 256;
        if (size < 2000) {
            batch = 64;
        }
        distances[start] = 0;
        for (int step = 0; step < size - 1; ++step) {
            auto min_value_index_pair = std::make_pair<>(std::numeric_limits<int>::max(), 0);
#pragma omp declare reduction \
        (min_index_reduction : decltype(min_value_index_pair) : omp_out = std::min(omp_out, omp_in)) \
        initializer(omp_priv = { std::numeric_limits<int>::max(), 0 })

#pragma omp parallel for schedule(dynamic, batch) reduction(min_index_reduction: min_value_index_pair)
            for (auto j = 0; j < distances.size(); j++) {
                if (!used[j] && (distances[j] <= min_value_index_pair.first)) {
                    min_value_index_pair = {distances[j], j};
                }
            }
            auto min_index = min_value_index_pair.second;
            if (distances[min_index] == std::numeric_limits<int>::max()) {
                break;
            }
            used[min_index] = true;
#pragma omp parallel for schedule(dynamic, batch)
            for (auto j = 0; j < this->graph[min_index].size(); j++) {

                auto weight = this->graph[min_index][j];
                if (weight == 0) { continue; }
                auto relaxed_dist = distances[min_index] + weight;
                if (!used[j] && relaxed_dist < distances[j])
                    distances[j] = relaxed_dist;
            }
        }
        std::copy(distances.begin(), distances.end(), this->distances.begin());
    }

    void parallelStaticDijkstra() {
        std::vector<int> distances(size, std::numeric_limits<int>::max());
        std::vector<bool> used(size);
        distances[start] = 0;

        for (int step = 0; step < size - 1; ++step) {
            auto min_value_index_pair = std::make_pair<>(std::numeric_limits<int>::max(), 0);
#pragma omp declare reduction \
        (min_index_reduction : decltype(min_value_index_pair) : omp_out = std::min(omp_out, omp_in)) \
        initializer(omp_priv = { std::numeric_limits<int>::max(), 0 })

#pragma omp parallel for schedule(static) reduction(min_index_reduction: min_value_index_pair)
            for (auto j = 0; j < distances.size(); j++) {
                if (!used[j] && (distances[j] <= min_value_index_pair.first)) {
                    min_value_index_pair = {distances[j], j};
                }
            }
            auto min_index = min_value_index_pair.second;
            if (distances[min_index] == std::numeric_limits<int>::max()) {
                break;
            }
            used[min_index] = true;
#pragma omp parallel for schedule(static)
            for (auto j = 0; j < this->graph[min_index].size(); j++) {

                auto weight = this->graph[min_index][j];
                if (weight == 0) { continue; }
                auto relaxed_dist = distances[min_index] + weight;
                if (!used[j] && relaxed_dist < distances[j])
                    distances[j] = relaxed_dist;
            }
        }
        std::copy(distances.begin(), distances.end(), this->distances.begin());

    }
};
