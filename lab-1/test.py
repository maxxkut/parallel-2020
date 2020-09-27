import subprocess
import matplotlib.pyplot as plt

EXECUTABLE = "./cpp/cmake-build-release/matrix_mult"

calculation_modes = {
    0: "Sequential",
    1: "Parallel (outer loop)",
    2: "Parallel calculations, static schedule",
    3: "Parallel calculations, dynamic schedule",
    4: "Parallel calculations, guided schedule",
}

matrix_sizes = (
    (200, 200, 200),
    (500, 500, 500),
    (500, 1000, 500),
    (1000, 1000, 500),
    (1000, 500, 500),
    (1000, 500, 250),
    (1000, 500, 1000),
    (1, 1_000_000, 1),
    (1_000_000, 100, 1),
)

benchmarks = {
    mode: {size: None for size in matrix_sizes}
    for mode in calculation_modes
}


def benchmark():
    for matrix_size in matrix_sizes:
        benchmark_matrix(matrix_size)


def benchmark_matrix(matrix_shape=()):
    print(f"Benchmarking matrix {matrix_shape[0]}x{matrix_shape[1]} * "
          f"{matrix_shape[1]}x{matrix_shape[2]}")
    size = matrix_shape
    for mode in calculation_modes:
        duration = benchmark_matrix_mode(matrix_shape, mode)
        benchmarks[mode][size] = duration


def benchmark_matrix_mode(matrix_shape, mode):
    print("Benchmarking mode %d" % mode)
    iterations = 50
    iterations_time = 0
    for i in range(iterations):
        process = subprocess.Popen(
            [EXECUTABLE, str(matrix_shape[0]), str(matrix_shape[1]), str(matrix_shape[2]), str(mode)],
            stdout=subprocess.PIPE)
        line = process.stdout.readline()
        iterations_time += float(line)
    duration = iterations_time / iterations
    print("Duration: %.2f seconds" % duration)
    return duration


def plot_results():
    serial_durations = tuple(benchmarks[0].values())
    fig = plt.figure()
    ax = plt.subplot(111)
    plot_shapes = [f"[{size[0]} x {size[1]}]\n[{size[1]} x {size[2]}]" for size
                   in matrix_sizes]
    for mode in benchmarks:
        durations = tuple(benchmarks[mode].values())
        accelerations = tuple(sd / d for sd, d in zip(serial_durations, durations))
        ax.plot(plot_shapes, accelerations, marker='o', label=calculation_modes[mode])

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.8])

    # Put a legend below current axis

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=1)
    plt.show()



if __name__ == '__main__':
    benchmark()
    print("Results", benchmarks)
    plot_results()
