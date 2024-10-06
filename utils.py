import random
import torch 
import numpy as np
import networkx as nx
import dgl

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
    dot_product = np.dot(A, B)
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)
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

def cosine_similarity(A, B):
    dot_product = np.dot(A, B)
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)
    similarity = dot_product / (norm_A * norm_B)
    return similarity

def correlation_matrix(tensor1, tensor2):
    
    # Flatten tensors into 1D arrays
    flat_tensor1 = tensor1.flatten().numpy()
    flat_tensor2 = tensor2.flatten().numpy()

    # Concatenate flattened tensors along the second axis

    # Compute the correlation matrix
    corr_matrix = np.corrcoef(flat_tensor1, flat_tensor2)[0, 1]
    
    return corr_matrix

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