import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import time
import matplotlib.pyplot as plt
import numpy as np

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


def main():
    model = ModelNew()
    a = torch.randn(1024, device="cuda")
    b = torch.randn(1024, device="cuda")

    run_times = []

    for _ in range(100):
        start_time = time.time()
        _ = model.forward(a, b)
        torch.cuda.synchronize()  # Wait for the CUDA kernel to finish
        end_time = time.time()
        run_times.append(end_time - start_time)

    # Log the run times
    for i, run_time in enumerate(run_times):
        print(f"Run {i + 1}: {run_time:.6f} seconds")

    # Visualize the run times
    plt.plot(run_times, label="Run times")
    plt.xlabel("Run")
    plt.ylabel("Time (seconds)")
    plt.title("CUDA Kernel Execution Time Over 100 Runs")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
