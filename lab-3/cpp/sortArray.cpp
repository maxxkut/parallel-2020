#pragma once

#include <vector>
#include <random>
#include <algorithm>

using namespace std;

class splittableArray {
public:
    int *content;
    int size;
    int pivotType;


    splittableArray(const int *buffer, int size, int pivotType) : size(size), pivotType(pivotType) {
        content = new int[size];
        for (int i = 0; i < size; i++) {
            content[i] = buffer[i];
        }
    }

    splittableArray(splittableArray *low, splittableArray *high) : size(low->size + high->size) {
        content = new int[size];
        pivotType = low->pivotType;
        for (int i = 0; i < low->size; i++) {
            content[i] = low->content[i];
        }
        for (int i = low->size; i < size; i++) {
            content[i] = high->content[i - low->size];
        }
    }

    int firstPivot() {
        return content[0];
    }

    int fastMedianPivot() {
        int a = content[0], b = content[size / 2], c = content[size - 1];
        if ((b <= a && a <= c) || (c <= a && a <= b)) {
            return a;
        }
        if ((a <= b && b <= c) || (c <= b && b <= a)) {
            return b;
        }
        return c;
    }

    int subsetMedianPivot() {
        auto iters = min(size, 50);
        vector<int> subset(iters);
        mt19937 mt{random_device{}()};
        uniform_int_distribution<uint32_t> dist(0, size - 1);
        vector<int> random_samples{};
        int n_samples = 50;
        random_samples.reserve(n_samples + 3);
        for (int i = 0; i < n_samples; ++i) {
            random_samples.push_back(content[dist(mt)]);
        }
        sort(random_samples.begin(), random_samples.end());
        return random_samples.at(random_samples.size() / 2);
    }

    int pivot() {
        switch (pivotType) {
            case 0:
                return firstPivot();
            case 1:
                return fastMedianPivot();
            default:
                return subsetMedianPivot();
        }
    }

    void partition(int pivot, splittableArray **low, splittableArray **high) {
        int i = -1, j = size;
        while (true) {
            do {
                i++;
            } while (content[i] < pivot);
            do {
                j--;
            } while (content[j] > pivot);
            if (i >= j) break;
            swap(content[i], content[j]);
        }
        j++;
        *low = new splittableArray(content, j, pivotType);
        *high = new splittableArray(&content[j], size - j, pivotType);
    }

    splittableArray *getPart(int part, int totalParts) {
        int partSize = size / totalParts;
        int oddSize = size % totalParts;
        int begin = part * partSize;
        begin += min(part, oddSize);
        return new splittableArray(&content[begin], part < oddSize ? partSize + 1: partSize, pivotType);
    }

};
