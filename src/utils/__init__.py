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
