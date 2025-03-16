import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
from torch.profiler import (
    profile,
    record_function,
    ProfilerActivity,
    tensorboard_trace_handler,
)

# Import NVTX for custom annotations
import torch.cuda.nvtx as nvtx


# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        # First convolution block with NVTX
        nvtx.range_push("conv_block_1")  # Start NVTX range
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        nvtx.range_pop()  # End NVTX range

        # Second convolution block with NVTX
        nvtx.range_push("conv_block_2")
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        nvtx.range_pop()

        # Third convolution block with NVTX
        nvtx.range_push("conv_block_3")
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)
        nvtx.range_pop()

        # Fully connected layers with NVTX
        nvtx.range_push("fc_layers")
        x = x.view(-1, 256 * 4 * 4)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        nvtx.range_pop()

        return x


# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize the model and move it to GPU
model = SimpleModel().to(device)

# Create random input data
batch_size = 64
input_data = torch.randn(batch_size, 3, 32, 32, device=device)
target = torch.randint(0, 10, (batch_size,), device=device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)


# Function to run a single training step with NVTX annotations
def train_step():
    nvtx.range_push("zero_grad")  # Annotate optimizer zero_grad
    optimizer.zero_grad()
    nvtx.range_pop()

    nvtx.range_push("forward")  # Annotate forward pass
    output = model(input_data)
    nvtx.range_pop()

    nvtx.range_push("loss_calculation")  # Annotate loss calculation
    loss = criterion(output, target)
    nvtx.range_pop()

    nvtx.range_push("backward")  # Annotate backward pass
    loss.backward()
    nvtx.range_pop()

    nvtx.range_push("optimizer_step")  # Annotate optimizer step
    optimizer.step()
    nvtx.range_pop()

    return loss


# Create a custom NVTX domain for grouping related events
nvtx_domain = nvtx.Domain("PyTorch_Training")

# Warmup with NVTX markers
nvtx.range_push("Warmup")
print("Warming up...")
for i in range(10):
    # Add NVTX markers with different colors for each warmup iteration
    with nvtx_domain.range(f"warmup_iter_{i}", color=i % 8):
        train_step()
nvtx.range_pop()

# Basic timing without profiler
nvtx.range_push("Basic_Timing")
start_time = time.time()
train_step()
end_time = time.time()
print(f"Basic timing: {(end_time - start_time) * 1000:.2f} ms")
nvtx.range_pop()

# Create directories for output files
os.makedirs("./nsight_profile", exist_ok=True)
os.makedirs("./log/pytorch_profiler", exist_ok=True)

# Profile with PyTorch Profiler and NVTX
print("\nRunning with PyTorch Profiler...")
nvtx.range_push("PyTorch_Profiling")
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for step in range(5):
        # NVTX annotation with custom color and payload
        nvtx_domain.range_push(f"training_step_{step}", color=step % 8)
        loss = train_step()
        print(f"Step {step}, Loss: {loss.item():.4f}")
        nvtx_domain.range_pop()
nvtx.range_pop()

# Print profiler results
print("\nProfiler Results:")
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# Export the profiler trace to a file
prof.export_chrome_trace("./nsight_profile/pytorch_gpu_trace.json")
print("\nExported chrome trace to ./nsight_profile/pytorch_gpu_trace.json")

# Export profiler data in formats useful for analysis with other tools
prof.export_stacks("./nsight_profile/profiler_stacks.txt", "self_cuda_time_total")
print("Exported stacks to ./nsight_profile/profiler_stacks.txt")

# Export detailed kernel information
print(
    prof.key_averages(group_by_input_shape=True).table(
        sort_by="cuda_time_total", row_limit=20
    )
)
print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))

# Custom NVTX metrics for more detailed analysis
nvtx.range_push("Custom_Metrics")
# Register custom metrics with NVTX
for i in range(5):
    # Adding numeric payload to ranges to represent metrics
    # This can be useful for annotating batch sizes, loss values, etc.
    nvtx_domain.range_push(f"custom_metric_{i}")
    nvtx.mark(f"metric_{i}_value: {i * 10}")  # Add a point marker with a value
    time.sleep(0.01)  # Simulate work
    nvtx_domain.range_pop()
nvtx.range_pop()

print("\nProfiling completed successfully!")

# Add information on how to use nsight with this script
print("\nTo profile with Nsight Systems, run this script with:")
print(
    "nsys profile -t cuda,nvtx,osrt,cudnn,cublas -o nsight_profile/nsys_report python enhanced_pytorch_profiling.py"
)
print("\nTo profile with Nsight Compute, run this script with:")
print(
    "ncu --set full -o nsight_profile/ncu_report python enhanced_pytorch_profiling.py"
)
