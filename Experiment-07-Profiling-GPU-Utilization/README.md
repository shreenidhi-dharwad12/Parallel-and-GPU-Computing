

# Experiment 07: Profiling and GPU Utilization



## 1. Aim



To profile a GPU workload, identify where execution time is lost, and improve GPU utilization by overlapping data transfer with computation using pinned host memory and two CUDA streams.



## 2. Learning Outcomes



- Capture and understand the timeline of a GPU program.

- Identify idle GPU time and host-to-device data transfers.

- Classify kernels as compute-bound or memory-bound using arithmetic intensity.

- Use pinned host memory for faster asynchronous data transfers.

- Apply two CUDA streams and double buffering to overlap data transfer with computation.

- Compare GPU utilization and total execution time before and after optimization.



## 3. Background Theory



### 3.1 GPU Utilization



GPU utilization indicates how much time the GPU spends performing useful work. A GPU may remain idle while waiting for data transfers, CPU operations, or previous tasks to finish.



Profiling helps identify these idle periods and improve GPU performance.



### 3.2 Where Time Is Lost



In a basic GPU program, each batch usually follows these steps:



1. Copy input data from host memory to GPU memory.

2. Execute GPU kernels.

3. Copy the result back to host memory.



When every step waits for the previous step, data transfer and computation happen one after another. This leaves parts of the GPU idle.



### 3.3 Pageable and Pinned Memory



**Pageable memory:**

- Normal host memory managed by the operating system.

- May require an additional staging-buffer copy before GPU transfer.

- Does not directly support asynchronous transfers in the same way as pinned memory.



**Pinned memory:**

- Host memory locked in physical RAM.

- Can be accessed directly by the GPU DMA engine.

- Supports asynchronous data transfers.

- Created in PyTorch using `pin_memory()`.



### 3.4 CUDA Streams



A CUDA stream is a sequence of GPU operations.



- Operations in one stream execute in order.

- Operations in different streams may execute concurrently when hardware and dependencies allow it.

- A copy stream handles data transfers.

- A compute stream handles GPU calculations.



Using two streams can allow the next batch to transfer while the current batch is being processed.



### 3.5 Double Buffering



Double buffering uses two GPU memory buffers.



While one buffer is used for computation, the other can receive the next batch. CUDA events ensure that a buffer is not overwritten while it is still being used.



### 3.6 Compute-Bound and Memory-Bound Kernels



Arithmetic intensity measures the number of floating-point operations performed per byte transferred.



**Formula:**



Arithmetic Intensity = FLOPs / Bytes moved



The ridge point is calculated as:



Ridge Point = Peak FP32 Performance / Peak Memory Bandwidth



- If arithmetic intensity is above the ridge point, the kernel is generally compute-bound.

- If arithmetic intensity is below the ridge point, the kernel is generally memory-bound.



For the sample GPU:



- Peak FP32 performance = 12.7 TFLOP/s

- Memory bandwidth = 360 GB/s

- Ridge point = 12700 / 360 = approximately 35 FLOP/byte.



## 4. Software and Hardware Requirements



### Hardware



- NVIDIA GPU

- Updated NVIDIA driver

- Approximately 2 GB or more free GPU memory



### Software



- Python 3.9 or later

- PyTorch with CUDA support

- CUDA-enabled NVIDIA driver

- NVIDIA System Management Interface (`nvidia-smi`)

- Perfetto or Chrome tracing for viewing profiling timelines

- Nsight Systems (optional)



## 5. Folder Structure



```text

Experiment-07-Profiling-GPU-Utilization/

â”‚

â”œâ”€â”€ README.md

â”‚

â”œâ”€â”€ code/

â”‚   â”œâ”€â”€ gpu_workload.py

â”‚   â””â”€â”€ analyze_trace.py

â”‚

â””â”€â”€ screenshots/

```



## 6. Program Details



### Program 1: gpu_workload.py



This program performs GPU computation in two modes.



**Baseline mode:**

- Uses pageable host memory.

- Uses the default CUDA stream.

- Performs blocking data transfers.

- Executes copying and computation one after another.



**Optimized mode:**

- Uses pinned host memory.

- Uses two CUDA streams.

- Uses double buffering.

- Performs asynchronous data transfers.

- Overlaps data transfer with computation where supported.



The workload includes:



1. Matrix multiplication (GEMM).

2. ReLU activation.

3. Element-wise multiplication and addition.

4. Sum reduction.



### Program 2: analyze_trace.py



This program reads PyTorch profiler trace JSON files and reports:



- GPU kernel busy time.

- Estimated GPU utilization.

- Memory-copy busy time.

- Copy and computation overlap.

- Idle gaps between kernels.

- Kernel execution time and name-based classification hints.



The automatic kernel classification is only a hint. Final classification must be supported by arithmetic-intensity calculations.



## 7. Procedure



### Step 1: Environment Setup



Check the NVIDIA GPU:



```bash

nvidia-smi

```



Check Python:



```bash

python --version

```



Install PyTorch with CUDA support. For the CUDA 12.1 example:



```bash

pip install torch --index-url https://download.pytorch.org/whl/cu121

```



Check CUDA availability:



```bash

python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

```



The command should display `True` and the GPU name.



Use the PyTorch installation command that matches the system's supported CUDA version.



### Step 2: Run the Baseline Workload



Start GPU utilization logging in a second terminal:



```bash

nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -lms 100 > smi_baseline.csv

```



Run the baseline program:



```bash

python code/gpu_workload.py --mode baseline --profile

```



Stop the GPU logging with `Ctrl + C` after the run.



Record:

- Total execution time.

- Time per batch.

- Throughput.

- Checksum.



The program generates:



```text

trace_baseline.json

```



### Step 3: Analyze the Baseline Timeline



Open `trace_baseline.json` using:



- https://ui.perfetto.dev

- Or Chrome tracing.



Identify:

- Memory-copy operations.

- GPU kernel execution.

- Idle gaps between kernels.



Run the analyzer:



```bash

python code/analyze_trace.py trace_baseline.json

```



Record the GPU utilization, kernel busy time, memory-copy time, and idle gaps.



### Step 4: Classify the Kernels



Calculate arithmetic intensity:



```text

Arithmetic Intensity = FLOPs / Bytes moved

```



For the GEMM operation:



```text

FLOPs = 2 x 16384 x 1024 x 1024

&#x20;     = approximately 34.4 billion



Bytes = (16384 x 1024 + 1024 x 1024

&#x20;        + 16384 x 1024) x 4



&#x20;     = approximately 138 MB



Arithmetic Intensity = 34.4 billion / 138 million

&#x20;                    = approximately 249 FLOP/byte

```



Compare the calculated arithmetic intensity with the GPU ridge point.



### Step 5: Run the Optimized Workload



Start utilization logging:



```bash

nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -lms 100 > smi_optimized.csv

```



Run the optimized program:



```bash

python code/gpu_workload.py --mode optimized --profile

```



Stop logging using `Ctrl + C`.



The program generates:



```text

trace_optimized.json

```



Check that the optimized checksum matches the baseline checksum. Small differences in the last digits may occur due to floating-point operations.



### Step 6: Compare Both Traces



Run:



```bash

python code/analyze_trace.py trace_baseline.json trace_optimized.json

```



Compare:

- GPU utilization.

- Kernel busy time.

- Idle gaps.

- Copy and computation overlap.

- Total execution time.



Open `trace_optimized.json` in Perfetto and observe the overlap between data transfers and computation.



## 8. Sample Results



**Note:** The following values are illustrative reference results from the lab manual, not measured results from the student's system. Actual results depend on the GPU and environment.



### 8.1 GPU Utilization and Execution Time



| Metric | Baseline | Optimized |

|---|---:|---:|

| Total time | 4.76 s | 2.62 s |

| Time per batch | 23.8 ms | 13.1 ms |

| Throughput | 42 batches/s | 76 batches/s |

| Average GPU utilization | 55% | 96% |

| Kernel busy time / window | 52% | 94% |

| Idle gaps above 50 us | 9 | 0 |

| Copy and kernel overlap | 0 ms | Approximately 50 ms |



Sample speedup:



```text

Speedup = Baseline Time / Optimized Time



&#x20;       = 4.76 / 2.62



&#x20;       = approximately 1.82x

```



### 8.2 Kernel Classification



| Kernel | Arithmetic Intensity | Classification |

|---|---:|---|

| GEMM | Approximately 249 FLOP/byte | Compute-bound |

| ReLU | Approximately 0.125 FLOP/byte | Memory-bound |

| Multiplication | Approximately 0.125 FLOP/byte | Memory-bound |

| Addition | Approximately 0.125 FLOP/byte | Memory-bound |

| Sum reduction | Approximately 0.25 FLOP/byte | Memory-bound |



These are sample classifications based on the lab manual's assumptions and example GPU.



### 8.3 Observation Table



Fill this table using the actual experiment results.



| Metric | Baseline | Optimized |

|---|---|---|

| GPU model | | |

| Total time (s) | | |

| Time per batch (ms) | | |

| Average GPU utilization (%) | | |

| Kernel busy time (%) | | |

| Idle gaps (count / ms) | | |

| Copy and kernel overlap (ms) | | |

| Checksum | | |



## 9. Bottlenecks and Optimizations



| Bottleneck | Optimization |

|---|---|

| Pageable host memory slows data transfer | Use pinned host memory |

| Copy and computation run one after another | Use two CUDA streams |

| A buffer may be reused too early | Use CUDA events and double buffering |

| CPU waits for every result copy | Use asynchronous D2H transfer and synchronize at the end |

| Element-wise kernels repeatedly access memory | Consider kernel fusion as a future improvement |



## 10. Important Observations



- Pageable memory can introduce extra data-copy overhead.

- Pinned memory supports asynchronous host-to-device transfers.

- CUDA streams can overlap data transfer and computation.

- Double buffering helps process one batch while transferring another.

- CUDA events help maintain correct execution order.

- GPU utilization and execution time should be compared before and after optimization.

- Actual performance improvement depends on the GPU hardware and driver.



## 11. Result



The GPU workload was profiled in baseline and optimized modes. The baseline execution was analyzed to identify idle GPU time and data-transfer overhead. Pinned host memory, two CUDA streams, and double buffering were used to improve GPU utilization by overlapping data transfers with computation.



The actual performance improvement was evaluated using execution time, GPU utilization, kernel activity, and checksum comparison.



## 12. Conclusion



This experiment demonstrated how GPU profiling helps identify performance bottlenecks. The baseline workload showed the effect of sequential data transfers and computation. The optimized workload used pinned memory, two CUDA streams, and double buffering to improve the overlap between data transfer and computation.



The experiment helped understand GPU utilization, arithmetic intensity, kernel classification, CUDA streams, and performance optimization.



## 13. Troubleshooting



| Problem | Possible Solution |

|---|---|

| CUDA is unavailable | Check NVIDIA drivers and install CUDA-enabled PyTorch |

| CUDA out of memory | Reduce batch rows using `--rows 8192` or `--rows 4096` |

| No kernels found in trace | Update the driver and retry profiling |

| Optimized mode is not faster | Check pinned memory, GPU load, and stream overlap |

| Copies and kernels do not overlap | Check GPU concurrency support and profiler timeline |

| Checksums differ significantly | Check the program logic and buffer synchronization |



## 14. References



- NVIDIA CUDA C++ Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/

- PyTorch Documentation: https://pytorch.org/docs/stable/

- PyTorch Profiler: https://pytorch.org/docs/stable/profiler.html

- Perfetto Trace Viewer: https://ui.perfetto.dev/




