
"""
Experiment 7: Profiling and GPU Utilization

Baseline:
    Pageable host memory, one stream, blocking copies.

Optimized:
    Pinned host memory, two CUDA streams and double buffering.
    Copy of the next batch can overlap with computation.
"""

import argparse
import math
import time
import torch


# ---------------------------------------------------------
# Workload setup
# ---------------------------------------------------------

def make_weights(n):
    generator = torch.Generator().manual_seed(1)
    weights = torch.randn(n, n, generator=generator) / math.sqrt(n)
    return weights.cuda()


def make_host_batches(rows, cols, pinned, unique=4):
    """Create a few reusable host batches."""
    generator = torch.Generator().manual_seed(0)
    batches = []

    for _ in range(unique):
        tensor = torch.randn(rows, cols, generator=generator)

        if pinned:
            tensor = tensor.pin_memory()

        batches.append(tensor)

    return batches


def compute(x, weights, reps):
    """GPU computation for one batch."""
    y = x

    for _ in range(reps):
        y = y @ weights
        y = torch.relu(y)
        y = y.mul(0.5).add(1.0)

    return y.sum(dim=1)


# ---------------------------------------------------------
# Baseline: pageable memory and blocking copies
# ---------------------------------------------------------

def run_baseline(host, weights, reps, nbatches):
    results = []

    for i in range(nbatches):
        x = host[i % len(host)].cuda()
        result = compute(x, weights, reps)
        results.append(result.cpu())

    torch.cuda.synchronize()
    return torch.stack(results)


# ---------------------------------------------------------
# Optimized: pinned memory and two CUDA streams
# ---------------------------------------------------------

def run_optimized(host, weights, reps, nbatches):
    rows, cols = host[0].shape

    copy_stream = torch.cuda.Stream()
    compute_stream = torch.cuda.Stream()

    # Two device buffers for double buffering
    buffers = [
        torch.empty(rows, cols, device="cuda")
        for _ in range(2)
    ]

    # Events track buffer availability and copy completion
    copied = [torch.cuda.Event() for _ in range(2)]
    freed = [torch.cuda.Event() for _ in range(2)]

    # Pinned output memory supports asynchronous D2H copies
    output_host = torch.empty(
        nbatches, rows, pin_memory=True
    )

    torch.cuda.synchronize()

    for event in freed:
        event.record(compute_stream)

    def issue_copy(i):
        k = i % 2

        # Wait until the device buffer is free
        copy_stream.wait_event(freed[k])

        with torch.cuda.stream(copy_stream):
            buffers[k].copy_(
                host[i % len(host)],
                non_blocking=True
            )
            copied[k].record(copy_stream)

    # Start the first batch transfer
    issue_copy(0)

    for i in range(nbatches):
        k = i % 2

        # Prefetch the next batch
        if i + 1 < nbatches:
            issue_copy(i + 1)

        # Wait until the current batch is copied
        compute_stream.wait_event(copied[k])

        with torch.cuda.stream(compute_stream):
            result = compute(buffers[k], weights, reps)

            # Mark the input buffer free after computation
            freed[k].record(compute_stream)

            # Copy result to pinned host memory
            output_host[i].copy_(
                result,
                non_blocking=True
            )

    torch.cuda.synchronize()
    return output_host


# ---------------------------------------------------------
# Main program
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["baseline", "optimized"],
        required=True
    )
    parser.add_argument("--batches", type=int, default=200)
    parser.add_argument("--rows", type=int, default=16384)
    parser.add_argument("--cols", type=int, default=1024)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-batches", type=int, default=10)

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA GPU not available. Check the driver and PyTorch installation."
        )

    pinned = args.mode == "optimized"

    if pinned:
        host = make_host_batches(args.rows, args.cols, True)
        run_function = run_optimized
    else:
        host = make_host_batches(args.rows, args.cols, False)
        run_function = run_baseline

    weights = make_weights(args.cols)

    print(f"GPU        : {torch.cuda.get_device_name(0)}")
    print(f"Mode       : {args.mode}")
    print(
        f"Batch size : "
        f"{args.rows * args.cols * 4 / 1e6:.0f} MB"
    )

    # Warm-up
    run_function(host, weights, args.reps, 3)
    torch.cuda.synchronize()

    # Timed execution
    start_time = time.perf_counter()

    result = run_function(
        host,
        weights,
        args.reps,
        args.batches
    )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time

    print(f"Batches    : {args.batches}")
    print(f"Total time : {elapsed:.3f} s")
    print(
        f"Per batch  : "
        f"{elapsed / args.batches * 1000:.2f} ms"
    )
    print(
        f"Throughput : "
        f"{args.batches / elapsed:.1f} batches/s"
    )
    print(f"Checksum   : {result.double().sum().item():.6e}")

    # GPU profiling
    if args.profile:
        from torch.profiler import profile, ProfilerActivity

        with profile(
            activities=[
                ProfilerActivity.CPU,
                ProfilerActivity.CUDA
            ]
        ) as profiler:

            run_function(
                host,
                weights,
                args.reps,
                args.profile_batches
            )

            torch.cuda.synchronize()

        trace_name = f"trace_{args.mode}.json"
        profiler.export_chrome_trace(trace_name)

        print(f"\nTrace saved: {trace_name}")

        print(
            profiler.key_averages().table(
                sort_by="cuda_time_total",
                row_limit=10
            )
        )


if __name__ == "__main__":
    main()
