import os
print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))




print('NEW line ')

import torch

# Check if CUDA is available
if torch.cuda.is_available():
    print(f"CUDA is available. Device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("CUDA is not available.")

import torch

# Check if CUDA is available
if torch.cuda.is_available():
    print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")

    # Create a tensor and move it to the GPU
    x = torch.rand(3, 3)
    x_gpu = x.to('cuda')
    print(f"Tensor on GPU: {x_gpu}")

    # Perform a simple operation on the GPU
    y_gpu = x_gpu * 2
    print(f"Result of operation on GPU: {y_gpu}")

else:
    print("CUDA is not available. Running on CPU.")
