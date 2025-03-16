import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging
from torch.cuda.nvtx import range_push, range_pop
from torch.utils.benchmark import Timer
from torch.profiler import profile, record_function, ProfilerActivity

# Configure logging
logging.basicConfig(
    filename='cuda_kernel_profiling.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Define the custom CUDA kernel for element-wise addition
elementwise_add_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void elementwise_add_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b) {
    auto size = a.numel();
    auto out = torch::zeros_like(a);
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    elementwise_add_kernel<<<num_blocks, block_size>>>(
        a.data_ptr<float>(), 
        b.data_ptr<float>(), 
        out.data_ptr<float>(), 
        size
    );
    
    return out;
}
"""

elementwise_add_cpp_source = (
    "torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b);"
)

# Compile the inline CUDA code for element-wise addition
elementwise_add = load_inline(
    name="elementwise_add",
    cpp_sources=elementwise_add_cpp_source,
    cuda_sources=elementwise_add_source,
    functions=["elementwise_add_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.elementwise_add = elementwise_add
    
    def forward(self, a, b):
        return self.elementwise_add.elementwise_add_cuda(a, b)

def collect_metrics(model, input_sizes, num_runs=100):
    metrics = {}
    
    for size in input_sizes:
        logging.info(f"Testing with input size: {size}")
        a = torch.randn(size, device='cuda')
        b = torch.randn(size, device='cuda')
        
        # Warm-up runs
        for _ in range(10):
            _ = model.forward(a, b)
            torch.cuda.synchronize()
        
        # 1. Basic timing
        run_times = []
        for _ in range(num_runs):
            start_time = time.time()
            _ = model.forward(a, b)
            torch.cuda.synchronize()
            end_time = time.time()
            run_times.append((end_time - start_time) * 1000000)  # Convert to μs
        
        # 2. CUDA events timing (more accurate)
        cuda_times = []
        for _ in range(num_runs):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            _ = model.forward(a, b)
            end_event.record()
            torch.cuda.synchronize()
            cuda_times.append(start_event.elapsed_time(end_event) * 1000)  # Convert ms to μs
        
        # 3. Benchmark using torch.utils.benchmark
        benchmark_timer = Timer(
            stmt="model.forward(a, b)",
            globals={"model": model, "a": a, "b": b}
        )
        benchmark_result = benchmark_timer.timeit(num_runs)
        benchmark_time_us = benchmark_result.mean * 1000000  # Convert to μs
        
        # 4. Detailed profiling using torch.profiler
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            with record_function("model_inference"):
                _ = model.forward(a, b)
                torch.cuda.synchronize()
        
        # 5. Memory usage
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        _ = model.forward(a, b)
        torch.cuda.synchronize()
        memory_allocated = torch.cuda.max_memory_allocated() / 1024**2  # MB
        memory_reserved = torch.cuda.max_memory_reserved() / 1024**2    # MB
        
        # 6. Calculate throughput
        data_size_bytes = a.numel() * 4 * 3  # 2 inputs + 1 output, 4 bytes per float32
        avg_time_seconds = sum(cuda_times) / len(cuda_times) / 1000000  # Convert μs to seconds
        throughput_gb_s = data_size_bytes / 1024**3 / avg_time_seconds
        
        # Collect metrics
        metrics[size] = {
            "mean_time_us": np.mean(run_times),
            "std_time_us": np.std(run_times),
            "median_time_us": np.median(run_times),
            "min_time_us": min(run_times),
            "max_time_us": max(run_times),
            "cuda_mean_time_us": np.mean(cuda_times),
            "cuda_std_time_us": np.std(cuda_times),
            "benchmark_mean_us": benchmark_time_us,
            "memory_allocated_mb": memory_allocated,
            "memory_reserved_mb": memory_reserved,
            "throughput_gb_s": throughput_gb_s,
            "profiler_data": prof.key_averages().table(sort_by="cuda_time_total", row_limit=10),
        }
        
        # Log results
        logging.info(f"Mean execution time: {metrics[size]['mean_time_us']:.4f} μs")
        logging.info(f"CUDA event time: {metrics[size]['cuda_mean_time_us']:.4f} μs")
        logging.info(f"Benchmark time: {metrics[size]['benchmark_mean_us']:.4f} μs")
        logging.info(f"Memory allocated: {metrics[size]['memory_allocated_mb']:.2f} MB")
        logging.info(f"Memory reserved: {metrics[size]['memory_reserved_mb']:.2f} MB")
        logging.info(f"Throughput: {metrics[size]['throughput_gb_s']:.2f} GB/s")
        logging.info(f"Profiler output:\n{metrics[size]['profiler_data']}")
    
    return metrics

def visualize_metrics(metrics):
    sizes = list(metrics.keys())
    
    # Create a figure with multiple subplots for different metrics
    plt.figure(figsize=(15, 10))
    
    # 1. Execution Times
    plt.subplot(2, 2, 1)
    plt.plot(sizes, [metrics[size]["mean_time_us"] for size in sizes], 'o-', label='Python time')
    plt.plot(sizes, [metrics[size]["cuda_mean_time_us"] for size in sizes], 's-', label='CUDA events')
    plt.plot(sizes, [metrics[size]["benchmark_mean_us"] for size in sizes], '^-', label='Benchmark')
    plt.xlabel('Input Size (elements)')
    plt.ylabel('Time (μs)')
    plt.title('Execution Time vs Input Size')
    plt.legend()
    plt.grid(True)
    plt.xscale('log', base=2)
    plt.yscale('log', base=10)
    
    # 2. Memory Usage
    plt.subplot(2, 2, 2)
    plt.plot(sizes, [metrics[size]["memory_allocated_mb"] for size in sizes], 'o-', label='Allocated')
    plt.plot(sizes, [metrics[size]["memory_reserved_mb"] for size in sizes], 's-', label='Reserved')
    plt.xlabel('Input Size (elements)')
    plt.ylabel('Memory (MB)')
    plt.title('Memory Usage vs Input Size')
    plt.legend()
    plt.grid(True)
    plt.xscale('log', base=2)
    
    # 3. Throughput
    plt.subplot(2, 2, 3)
    plt.plot(sizes, [metrics[size]["throughput_gb_s"] for size in sizes], 'o-', color='green')
    plt.xlabel('Input Size (elements)')
    plt.ylabel('Throughput (GB/s)')
    plt.title('Memory Throughput vs Input Size')
    plt.grid(True)
    plt.xscale('log', base=2)
    
    # 4. Min/Max Execution Times - Fixed to avoid negative values
    plt.subplot(2, 2, 4)
    means = [metrics[size]["cuda_mean_time_us"] for size in sizes]
    mins = [metrics[size]["min_time_us"] for size in sizes]
    maxs = [metrics[size]["max_time_us"] for size in sizes]
    
    plt.plot(sizes, means, 'o-', label='Mean time')
    plt.fill_between(sizes, mins, maxs, alpha=0.3, label='Min-Max Range')
    plt.xlabel('Input Size (elements)')
    plt.ylabel('Time (μs)')
    plt.title('Execution Time Range (Min/Max)')
    plt.legend()
    plt.grid(True)
    plt.xscale('log', base=2)
    
    plt.tight_layout()
    plt.savefig('cuda_kernel_metrics.png')
    logging.info("Saved visualization to cuda_kernel_metrics.png")

def main():
    logging.info("Starting CUDA kernel profiling")
    model = ModelNew()
    logging.info("Model created successfully")
    
    # Test with power-of-2 input sizes
    input_sizes = [2**i for i in range(10, 21)]  # 1024 to 2,097,152
    logging.info(f"Testing input sizes: {input_sizes}")
    
    metrics = collect_metrics(model, input_sizes, num_runs=100)
    logging.info("Metrics collection complete")
    
    # Visualize the results
    visualize_metrics(metrics)
    logging.info("Visualization complete")
    
    # Export metrics to CSV
    metrics_df = pd.DataFrame({
        'size': [],
        'metric': [],
        'value': []
    })
    
    for size in metrics:
        for metric, value in metrics[size].items():
            if metric != "profiler_data":  # Skip profiler table
                metrics_df = pd.concat([metrics_df, pd.DataFrame({
                    'size': [size],
                    'metric': [metric],
                    'value': [value]
                })])
    
    metrics_df.to_csv('cuda_kernel_metrics.csv', index=False)
    logging.info("Metrics saved to cuda_kernel_metrics.csv")

if __name__ == "__main__":
    main()
