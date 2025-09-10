import random
import torch 
import numpy as np
import networkx as nx
import dgl
from atsp_utils import tour_cost
from algorithms import nearest_neighbor
import psutil

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

def calculate_statistics(tensors):
    stats = {}
    for key, values in tensors.items():
        stats[key] = {
            'mean': torch.mean(values).item(),
            'min': torch.min(values).item(),
            'max': torch.max(values).item(),
            'q1': torch.quantile(values, 0.25).item(),
            'median': torch.median(values).item(),
            'q3': torch.quantile(values, 0.75).item()
        }
    return stats

def atsp_results(model, args, val_set):
    result2 = dict()
    keys = ['avg_corr', 'avg_corr_cosin', 'avg_init_cost', 'avg_opt_cost', 'avg_gap']
    for key in keys:
        result2.setdefault(key, 0.)
    for idx in range(args.n_samples_result_train):
        G = nx.read_gpickle(f'{args.data_dir}/{val_set.instances[idx]}')
        H = val_set.get_scaled_features(G).to(args.device)
        x = H.ndata['weight']
        y = H.ndata['regret']
        with torch.no_grad():
            y_pred = model(H, x)
        
        regret_pred = val_set.scalers['regret'].inverse_transform(y_pred.cpu().numpy())
        
        for e, regret_pred_i in zip(val_set.es, regret_pred):
            G.edges[e]['regret_pred'] = np.maximum(regret_pred_i.item(), 0)
        
        opt_cost = gnngls.optimal_cost(G, weight='weight')
        
        init_tour = nearest_neighbor(G, 0, weight='regret_pred')
        init_cost = tour_cost(G, init_tour)
        result2['avg_corr'] += correlation_matrix(y_pred.cpu(), H.ndata['regret'].cpu())
        result2['avg_corr_cosin'] += cosine_similarity(y_pred.cpu().flatten(), H.ndata['regret'].cpu().flatten())
        result2['avg_init_cost'] += init_cost
        result2['avg_opt_cost'] += opt_cost
        result2['avg_gap'] += (init_cost / opt_cost - 1) * 100
    
    return result2

def tsp_to_atsp_instance(G1):
    num_nodes = G1.number_of_nodes() // 2
    G2 = nx.DiGraph()
    G2.add_nodes_from(range(num_nodes))
    G2.add_edges_from([(u, v) for u in range(num_nodes) for v in range(num_nodes) if u != v])

    first_edge = list(G1.edges)[0]

    # Get the attribute names of the first edge
    attribute_names = G1[first_edge[0]][first_edge[1]].keys()
    attribute_names_list = list(attribute_names)
    for attribute_name in attribute_names_list:
        attribute, _ = nx.attr_matrix(G1, attribute_name)
        attribute = attribute[num_nodes:, :num_nodes]
        for u, v in G2.edges():
            G2[u][v][attribute_name] = attribute[u, v]
    
    return G2

def fix_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
    dgl.random.seed(seed)

def cosine_similarity(A, B):
    # Calculate the dot product and norms
    dot_product = np.dot(A, B)
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)

    # Handle the case where norm_A or norm_B is zero
    if norm_A == 0 or norm_B == 0:
        return 0.0

    # Calculate the similarity
    similarity = dot_product / (norm_A * norm_B)
    
    return similarity

def correlation_matrix(tensor1, tensor2):
    # Flatten tensors into 1D arrays
    flat_tensor1 = tensor1.flatten().numpy()
    flat_tensor2 = tensor2.flatten().numpy()
    # Compute the correlation matrix
    corr_matrix = np.corrcoef(flat_tensor1, flat_tensor2)[0, 1]
    return corr_matrix

def save(model, optimizer, epoch, train_loss, val_loss, save_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': train_loss,
        'val_loss': val_loss
    }, save_path)

def add_diag(num_nodes, t1):
    n = num_nodes
    t2 = torch.zeros(n, n, dtype=torch.float32)
    cnt = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            t2[i][j] = t1[cnt]
            cnt += 1
    return t2

# # loss function using the cosince that showed that does not work well
# def loss3(y_pred, y, batch_size = 32):
#     num_edges = int(y.shape[0]//batch_size)
#     y = y.view(batch_size, num_edges) ** 2
#     y = F.normalize(y, p=2, dim=1)
    
#     y_pred = y_pred.view(batch_size, num_edges) ** 2
#     y_pred = F.normalize(y_pred, p=2, dim=1)
    
#     cos_similarities = F.cosine_similarity(y, y_pred, dim=1)

#     return 1 - cos_similarities.mean()