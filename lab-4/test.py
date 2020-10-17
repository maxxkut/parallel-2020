import csv
import numpy as np
import subprocess
import matplotlib.pyplot as plt
import random
EXECUTABLE = "./cpp/cmake-build-debug/lab-4"

sizes = (500, 1000, 5_000, 10_000, 20000)


iterations = 10

launch_types = {
    0: "serial",
    1: "static schedule",
    2: "dynamic schedule static batch"
}


benchmarks = {
    launch_type: {size: None for size in sizes}
    for launch_type in launch_types
}


def generate_full_graph(size):
    lst = [[0 for _ in range(size)] for _ in range(size)]
    for i in range(size):
        for j in range(i + 1, size):
            lst[j][i] = lst[i][j] = random.randint(1, 1000)
    return lst


def write_matrix(path, matrix):
    with open(path, 'w') as output:
        output.write(f'{len(matrix)}\n')
        for row in matrix:
            output.write(' '.join(map(str, row)) + '\n')


def benchmark():
    for size in sizes:
        benchmark_system(size)


def benchmark_system(size):
    print(f"Benchmarking system of size {size}")

    graph = generate_full_graph(size)
    write_graph(graph, size, 0)
    for mode in launch_types.keys():
        duration = benchmark_system_with_type(mode)
        benchmarks[mode][size] = duration


def create_graph(size):
    print(f"generating graph size {size}")
    start_el = int(np.random.rand() * size)
    graph = np.zeros((size, size))
    for i in range(size):
        for j in range(i + 1, size):
            el = np.random.rand() * 100 + 1
            graph[i, j] = el
            graph[j, i] = el
    graph = graph.astype(int)
    write_graph(graph, size, start_el)


def write_graph(graph, size, start):
    with open("dijkstra.input", "w") as out_file:
        out_file.write(f"{size}\n{start}\n")
        for row in graph:
            out_file.write("{}\n".format(" ".join(map(str, row))))


def benchmark_system_with_type(mode) -> float:
    print(f"Benchmarking {launch_types[mode]} mode")

    duration = 0
    for i in range(iterations):
        process = subprocess.Popen([EXECUTABLE, str(mode), str(100)], stdout=subprocess.PIPE)
        line = process.stdout.readline()
        duration += float(line)
        process.communicate()[0]
        assert (process.returncode == 0)
    duration = duration / iterations
    if duration < 10e-6:
        duration = 10e-6
    print("Duration: %f seconds" % duration)
    return duration


def plot_results():
    serial_durations = tuple(benchmarks[0].values())
    fig = plt.figure()
    ax = plt.subplot(111)
    plot_shapes = [f"[{size}]" for size
                   in sizes]
    for mode in benchmarks:
        durations = tuple(benchmarks[mode].values())
        accelerations = tuple(sd / d for sd, d in zip(serial_durations, durations))
        ax.plot(plot_shapes, accelerations, marker='o', label=launch_types[mode])

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.8])

    # Put a legend below current axis

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=1)
    plt.show()


def save_results():
    records = []
    for process_count in benchmarks:
        for system_size in benchmarks[process_count]:
            records.append((system_size, process_count, benchmarks[process_count][system_size]))
    records.sort(key=lambda record: record[0])
    with open("results.csv", "w") as file:
        writer = csv.writer(file)
        writer.writerow(["length", "processes", "time"])
        for record in records:
            writer.writerow(record)
    print(records)


if __name__ == '__main__':
    np.random.seed(1337)
    benchmark()
    plot_results()
    save_results()
    # create_graph(500)
