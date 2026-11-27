import csv

input_file = "results/results.csv"
output_file = "results/performance.csv"

rows = []

with open(input_file, "r") as file:
    reader = csv.DictReader(file)

    for row in reader:
        rows.append(row)

# Get sequential execution time for each dataset
sequential_times = {}

for row in rows:
    if row["Method"] == "Sequential":
        sequential_times[row["Dataset"]] = float(row["ExecutionTime"])

with open(output_file, "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "Dataset",
        "Method",
        "Workers",
        "ExecutionTime",
        "Speedup",
        "Efficiency"
    ])

    for row in rows:

        dataset = row["Dataset"]
        method = row["Method"]
        workers = int(row["Workers"])
        execution_time = float(row["ExecutionTime"])

        if method == "Sequential":
            speedup = 1.0
            efficiency = 1.0

        else:
            speedup = sequential_times[dataset] / execution_time
            efficiency = speedup / workers

        writer.writerow([
            dataset,
            method,
            workers,
            f"{execution_time:.6f}",
            f"{speedup:.4f}",
            f"{efficiency:.4f}"
        ])

print("Performance analysis generated successfully.")
print("Output:", output_file)
