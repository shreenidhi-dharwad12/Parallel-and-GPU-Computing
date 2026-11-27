
"""
Experiment 7: GPU Trace Analyzer

Reads PyTorch Chrome trace JSON files and reports:
- GPU kernel busy time
- Estimated GPU utilization
- Idle gaps between kernels
- Memory-copy time
- Copy and compute overlap
- Kernel classification hints
"""

import collections
import json
import sys

GAP_US = 50

COMPUTE_HINTS = (
    "gemm",
    "sgemm",
    "cutlass",
    "cublas",
    "conv",
    "wmma"
)


def merge(intervals):
    """Merge overlapping time intervals."""
    intervals = sorted(intervals)
    output = []

    for start, end in intervals:
        if output and start <= output[-1][1]:
            output[-1][1] = max(output[-1][1], end)
        else:
            output.append([start, end])

    return output


def total(intervals):
    return sum(end - start for start, end in intervals)


def overlap(first, second):
    """Calculate the time shared by two sets of intervals."""
    i = 0
    j = 0
    shared_time = 0

    while i < len(first) and j < len(second):
        start = max(first[i][0], second[j][0])
        end = min(first[i][1], second[j][1])

        if end > start:
            shared_time += end - start

        if first[i][1] < second[j][1]:
            i += 1
        else:
            j += 1

    return shared_time


def label_kernel(name):
    """Provide a name-based classification hint."""
    name_lower = name.lower()

    if any(hint in name_lower for hint in COMPUTE_HINTS):
        return "compute-bound"

    return "memory-bound"


def analyze(path):
    with open(path, encoding="utf-8") as file:
        data = json.load(file)

    events = (
        data["traceEvents"]
        if isinstance(data, dict)
        else data
    )

    kernels = []
    copies = []

    kernel_names = collections.defaultdict(lambda: [0, 0.0])
    copy_names = collections.defaultdict(lambda: [0, 0.0])

    for event in events:
        if event.get("ph") != "X":
            continue

        category = event.get("cat", "")
        start = event.get("ts", 0)
        duration = event.get("dur", 0)
        name = event.get("name", "Unknown")

        if category == "kernel":
            kernels.append([start, start + duration])

            kernel_names[name[:60]][0] += 1
            kernel_names[name[:60]][1] += duration

        elif category in ("gpu_memcpy", "gpu_memset"):
            copies.append([start, start + duration])

            copy_names[name[:60]][0] += 1
            copy_names[name[:60]][1] += duration

    if not kernels:
        print(f"{path}: No GPU kernels found in trace.")
        return None

    merged_kernels = merge(kernels)
    merged_copies = merge(copies)

    start_time = min(
        merged_kernels[0][0],
        merged_copies[0][0] if merged_copies else merged_kernels[0][0]
    )

    end_time = max(
        merged_kernels[-1][1],
        merged_copies[-1][1] if merged_copies else merged_kernels[-1][1]
    )

    span = end_time - start_time
    busy = total(merged_kernels)

    gaps = [
        merged_kernels[i + 1][0] - merged_kernels[i][1]
        for i in range(len(merged_kernels) - 1)
    ]

    large_gaps = [gap for gap in gaps if gap > GAP_US]

    copy_time = total(merged_copies)
    overlap_time = overlap(merged_kernels, merged_copies)

    utilization = 100 * busy / span if span > 0 else 0

    print("=" * 72)
    print(path)
    print("=" * 72)

    print(f"Profiled window             : {span / 1000:9.2f} ms")
    print(f"Kernel busy time            : {busy / 1000:9.2f} ms")
    print(f"GPU utilization (busy/span) : {utilization:9.1f} %")
    print(f"Copy busy time              : {copy_time / 1000:9.2f} ms")
    print(f"Copy overlapped with kernels: {overlap_time / 1000:9.2f} ms")

    print(
        f"Idle gaps > {GAP_US} us       : "
        f"{len(large_gaps):9d} "
        f"(total {sum(large_gaps) / 1000:.2f} ms)"
    )

    print("\nKernels (name-based classification hint)")
    print(f"{'Calls':>6} {'Total ms':>10}  {'Label':<14} Name")

    for name, (count, duration) in sorted(
        kernel_names.items(),
        key=lambda item: -item[1][1]
    ):
        print(
            f"{count:6d} "
            f"{duration / 1000:10.2f}  "
            f"{label_kernel(name):<14} "
            f"{name}"
        )

    print("\nMemory copies")
    print(f"{'Calls':>6} {'Total ms':>10}  Name")

    for name, (count, duration) in sorted(
        copy_names.items(),
        key=lambda item: -item[1][1]
    ):
        print(
            f"{count:6d} "
            f"{duration / 1000:10.2f}  "
            f"{name}"
        )

    print()

    return (
        span,
        busy,
        copy_time,
        overlap_time,
        len(large_gaps)
    )


def main():
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: python analyze_trace.py trace_baseline.json "
            "trace_optimized.json"
        )

    results = []

    for path in sys.argv[1:]:
        result = analyze(path)

        if result:
            results.append((path, result))

    if len(results) > 1:
        print("SUMMARY")
        print(
            f"{'File':<24}"
            f"{'Window ms':>10}"
            f"{'Util %':>8}"
            f"{'Overlap ms':>12}"
            f"{'Gaps':>6}"
        )

        for path, values in results:
            span, busy, _, overlap_time, gap_count = values

            utilization = 100 * busy / span if span > 0 else 0

            print(
                f"{path:<24}"
                f"{span / 1000:10.1f}"
                f"{utilization:8.1f}"
                f"{overlap_time / 1000:12.1f}"
                f"{gap_count:6d}"
            )


if __name__ == "__main__":
    main()
