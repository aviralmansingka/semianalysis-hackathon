import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.profiler import profile, record_function, ProfilerActivity


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
        # First convolution block
        with record_function("conv_block_1"):
            x = self.conv1(x)
            x = self.relu(x)
            x = self.pool(x)

        # Second convolution block
        with record_function("conv_block_2"):
            x = self.conv2(x)
            x = self.relu(x)
            x = self.pool(x)

        # Third convolution block
        with record_function("conv_block_3"):
            x = self.conv3(x)
            x = self.relu(x)
            x = self.pool(x)

        # Fully connected layers
        with record_function("fc_layers"):
            x = x.view(-1, 256 * 4 * 4)
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)

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


# Function to run a single training step
def train_step():
    optimizer.zero_grad()
    output = model(input_data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    return loss


# Warmup
print("Warming up...")
for _ in range(10):
    train_step()

# Basic timing without profiler
start_time = time.time()
train_step()
end_time = time.time()
print(f"Basic timing: {(end_time - start_time) * 1000:.2f} ms")

# Profile with PyTorch Profiler
print("\nRunning with PyTorch Profiler...")
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for step in range(5):
        with record_function(f"training_step_{step}"):
            loss = train_step()
            print(f"Step {step}, Loss: {loss.item():.4f}")

# Print profiler results
print("\nProfiler Results:")
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# Export the profiler trace to a file
prof.export_chrome_trace("pytorch_gpu_trace.json")
print("\nExported chrome trace to pytorch_gpu_trace.json")

# Optional: Export to tensorboard
try:
    prof.export_stacks("/tmp/profiler_stacks.txt", "self_cuda_time_total")
    print("Exported stacks to /tmp/profiler_stacks.txt")

    from torch.profiler import tensorboard_trace_handler

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
        on_trace_ready=tensorboard_trace_handler("./log/pytorch_profiler"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for step in range(5):
            train_step()
            prof.step()
    print("Exported tensorboard logs to ./log/pytorch_profiler")
except Exception as e:
    print(f"Tensorboard export failed: {e}")

print("\nProfiling completed successfully!")
