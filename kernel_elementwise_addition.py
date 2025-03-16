import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import time
import matplotlib.pyplot as plt
import numpy as np
import logging
import os
from datetime import datetime, timezone
from tqdm import tqdm
import pytz
import shutil

# Get the current timestamp
current_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# Define the base directory and new directory path
base_directory = os.getcwd()
output_directory = os.path.join(base_directory, current_timestamp)

# Create the new directory
os.makedirs(output_directory, exist_ok=True)

# Set the timezone to US/Pacific
os.environ['TZ'] = 'US/Pacific'
time.tzset()

# Configure logging
log_file_path = os.path.join(output_directory, "cuda_kernel_logs.log")
logging.basicConfig(
    filename=log_file_path,
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S.%f %Z"
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

    elementwise_add_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), size);

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
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.elementwise_add = elementwise_add

    def forward(self, a, b):
        return self.elementwise_add.elementwise_add_cuda(a, b)


def visualize_run_times(run_times, output_dir):
    """Visualize run times with statistics."""
    mean_time = np.mean(run_times)
    std_time = np.std(run_times)

    plt.figure(figsize=(10, 6))
    plt.plot(run_times, label='Run times')
    plt.axhline(mean_time, color='r', linestyle='--', label=f'Mean: {mean_time:.6f}s')
    plt.fill_between(
        range(len(run_times)),
        mean_time - std_time,
        mean_time + std_time,
        color='gray',
        alpha=0.2,
        label=f"±1 Std Dev: {std_time:.6f}s"
    )
    
    plt.xlabel('Run')
    plt.ylabel('Time (seconds)')
    plt.title('CUDA Kernel Execution Time Over 100 Runs')
    plt.legend()
    plt.grid(True)
    
    plot_file = os.path.join(output_dir, "run_times_visualization.png")
    plt.savefig(plot_file)  # Save plot to file
    plt.close()


def visualize_tensor_distributions(a, b, output_dir):
    """Visualize tensor distributions."""
    plt.figure(figsize=(10, 6))
    
    # Histogram for tensor 'a'
    plt.hist(a.cpu().numpy(), bins=50, alpha=0.5, label='Tensor A')
    
    # Histogram for tensor 'b'
    plt.hist(b.cpu().numpy(), bins=50, alpha=0.5, label='Tensor B')
    
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Tensor Value Distributions')
    plt.legend()
    plt.grid(True)
    
    plot_file = os.path.join(output_dir, "tensor_distributions.png")
    plt.savefig(plot_file)  # Save plot to file
    plt.close()


def main():
    model = ModelNew()
    
    # Generate random tensors on GPU
    a = torch.randn(1024, device='cuda')
    b = torch.randn(1024, device='cuda')

    logging.info("Generated input tensors 'a' and 'b' on CUDA device.")
    
    run_times = []

    # Use tqdm for progress tracking during 100 runs
    for i in tqdm(range(100), desc="Executing CUDA Kernel"):
        start_time = time.time()
        
        # Log tensor shapes and device info before kernel execution
        logging.debug(f"Run {i+1}: Tensor A shape: {a.shape}, Tensor B shape: {b.shape}, Device: {a.device}")
        
        _ = model.forward(a, b)
        
        torch.cuda.synchronize()  # Wait for the CUDA kernel to finish
        
        end_time = time.time()
        run_time = (end_time - start_time) * 1e6 # Convert to microseconds
        
        run_times.append(run_time)
        
        # Log execution time for each run
        logging.info(f"Run {i+1}: Execution time: {run_time:.6f} microseconds")

    
    # Log summary statistics after all runs
    mean_run_time = np.mean(run_times)
    std_run_time = np.std(run_times)
    
    logging.info(f"Mean execution time over 100 runs: {mean_run_time:.6f} microseconds")
    logging.info(f"Standard deviation of execution time over 100 runs: {std_run_time:.6f} microseconds")

    
    # Visualize results
    visualize_run_times(run_times, output_directory)
    
    # Visualize tensor distributions
    visualize_tensor_distributions(a, b, output_directory)


if __name__ == "__main__":
    main()
