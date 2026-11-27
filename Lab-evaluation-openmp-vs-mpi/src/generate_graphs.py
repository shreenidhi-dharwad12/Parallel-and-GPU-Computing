import csv
import os
import matplotlib.pyplot as plt

input_file = "results/performance.csv"

os.makedirs("graphs", exist_ok=True)

data = []

with open(input_file, "r") as file:
    reader = csv.DictReader(file)

    for row in reader:
        data.append({
            "Dataset": row["Dataset"],
            "Method": row["Method"],
            "Workers": int(row["Workers"]),
            "ExecutionTime": float(row["ExecutionTime"]),
            "Speedup": float(row["Speedup"]),
            "Efficiency": float(row["Efficiency"])
        })


datasets = ["100K", "1M", "5M"]
methods = ["OpenMP", "MPI"]


# --------------------------------------------------
# 1. Execution Time Graphs
# --------------------------------------------------

for dataset in datasets:

    plt.figure(figsize=(8, 5))

    for method in methods:

        rows = [
            row for row in data
            if row["Dataset"] == dataset
            and row["Method"] == method
        ]

        workers = [row["Workers"] for row in rows]
        times = [row["ExecutionTime"] for row in rows]

        plt.plot(
            workers,
            times,
            marker="o",
            label=method
        )

    plt.xlabel("Threads / Processes")
    plt.ylabel("Execution Time (seconds)")
    plt.title(f"Execution Time - {dataset} Dataset")
    plt.xticks([1, 2, 4, 8])
    plt.legend()
    plt.grid(True)

    plt.savefig(
        f"graphs/execution_time_{dataset}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# --------------------------------------------------
# 2. Speedup Graphs
# --------------------------------------------------

for dataset in datasets:

    plt.figure(figsize=(8, 5))

    for method in methods:

        rows = [
            row for row in data
            if row["Dataset"] == dataset
            and row["Method"] == method
        ]

        workers = [row["Workers"] for row in rows]
        speedups = [row["Speedup"] for row in rows]

        plt.plot(
            workers,
            speedups,
            marker="o",
            label=method
        )

    plt.xlabel("Threads / Processes")
    plt.ylabel("Speedup")
    plt.title(f"Speedup - {dataset} Dataset")
    plt.xticks([1, 2, 4, 8])
    plt.legend()
    plt.grid(True)

    plt.savefig(
        f"graphs/speedup_{dataset}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# --------------------------------------------------
# 3. Efficiency Graphs
# --------------------------------------------------

for dataset in datasets:

    plt.figure(figsize=(8, 5))

    for method in methods:

        rows = [
            row for row in data
            if row["Dataset"] == dataset
            and row["Method"] == method
        ]

        workers = [row["Workers"] for row in rows]
        efficiency = [row["Efficiency"] for row in rows]

        plt.plot(
            workers,
            efficiency,
            marker="o",
            label=method
        )

    plt.xlabel("Threads / Processes")
    plt.ylabel("Efficiency")
    plt.title(f"Efficiency - {dataset} Dataset")
    plt.xticks([1, 2, 4, 8])
    plt.legend()
    plt.grid(True)

    plt.savefig(
        f"graphs/efficiency_{dataset}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


print("All graphs generated successfully.")
print("Graphs saved in the graphs/ folder.")
