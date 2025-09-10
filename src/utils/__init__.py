import psutil
import torch 

def fix_seed(seed=42):
    import os
    env_vars = {
        'PYTHONHASHSEED': str(seed),
        'CUBLAS_WORKSPACE_CONFIG': ':4096:8'
    }
    for key, value in env_vars.items():
        os.environ[key] = value
    import random
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

def print_gpu_memory(text):
    print(text, flush=True)
    # Get the memory usage details
    memory_info = psutil.virtual_memory()

    # Convert bytes to GB (1 GB = 1024^3 bytes)
    used_gb = memory_info.used / (1024 ** 3)
    free_gb = memory_info.available / (1024 ** 3)

    # Print the RAM usage in GB
    print(f"RAM used: {used_gb:.2f} GB", flush=True)
    print(f"RAM available: {free_gb:.2f} GB", flush=True)
    if torch.cuda.is_available():
        # Get the allocated memory (used)
        allocated_memory = torch.cuda.memory_allocated()
        # Get the cached memory (reserved)
        cached_memory = torch.cuda.memory_reserved()
        # Get the total memory on the GPU
        total_memory = torch.cuda.get_device_properties(0).total_memory
        # Calculate free memory
        free_memory = total_memory - allocated_memory - cached_memory
        print(text, flush=True)
        print(f"GPU Memory Allocated: {allocated_memory / (1024 ** 2):.2f} MB", flush=True)
        print(f"GPU Memory Cached: {cached_memory / (1024 ** 2):.2f} MB", flush=True)
        print(f"GPU Memory Free: {free_memory / (1024 ** 2):.2f} MB", flush=True)
        print(f"Total GPU Memory: {total_memory / (1024 ** 2):.2f} MB", flush=True)
    else:
        print("No GPU available.", flush=True)