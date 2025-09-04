import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

class SparseGATConv(nn.Module):
    src_nodes_dim = 0  # position of source nodes in edge index
    trg_nodes_dim = 1  # position of target nodes in edge index

    def __init__(self, in_feats, out_feats, num_heads, dropout_prob=0.6):
        super().__init__()
        self.num_heads = num_heads
        self.out_feats = out_feats
        self.dropout_prob = dropout_prob

        self.W = nn.Parameter(torch.Tensor(in_feats, num_heads * out_feats))
        self.attn_l = nn.Parameter(torch.Tensor(1, num_heads, out_feats))
        self.attn_r = nn.Parameter(torch.Tensor(1, num_heads, out_feats))
        
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.attn_l)
        nn.init.xavier_uniform_(self.attn_r)

        self.leakyReLU = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, adj, feat):
        N = feat.size(0)
        feat = self.dropout(feat)
        h = torch.mm(feat, self.W).view(N, self.num_heads, self.out_feats)
        h = self.dropout(h)

        el = (h * self.attn_l).sum(dim=-1).unsqueeze(-1)
        er = (h * self.attn_r).sum(dim=-1).unsqueeze(-1)
        
        scores_source = el.squeeze(-1)
        scores_target = er.squeeze(-1)

        indices = adj._indices()
        src_index = indices[self.src_nodes_dim]
        trg_index = indices[self.trg_nodes_dim]

        scores_source_lifted = scores_source.index_select(0, src_index)
        scores_target_lifted = scores_target.index_select(0, trg_index)
        scores_per_edge = self.leakyReLU(scores_source_lifted + scores_target_lifted)

        attentions_per_edge = self.neighborhood_aware_softmax(scores_per_edge, trg_index, N)
        attentions_per_edge = self.dropout(attentions_per_edge)

        h_lifted = h.index_select(0, src_index)
        h_prime = torch.zeros_like(h)
        print(h.shape)
        print(h_lifted.shape)
        print(attentions_per_edge.shape)
        for i in range(self.num_heads):
            h_lifted_i = h_lifted[:, i, :] * attentions_per_edge[:, i].unsqueeze(-1)
            h_prime[:, i, :] = torch.sparse.mm(adj, h_lifted_i)

        return h_prime.view(N, -1)

    def neighborhood_aware_softmax(self, scores_per_edge, trg_index, num_of_nodes):
        scores_per_edge = scores_per_edge - scores_per_edge.max()
        exp_scores_per_edge = scores_per_edge.exp()

        trg_index_broadcasted = trg_index.unsqueeze(-1).expand_as(exp_scores_per_edge)
        size = [num_of_nodes, self.num_heads]
        neighborhood_sums = torch.zeros(size, dtype=exp_scores_per_edge.dtype, device=exp_scores_per_edge.device)

        neighborhood_sums.scatter_add_(0, trg_index_broadcasted, exp_scores_per_edge)
        return (exp_scores_per_edge / (neighborhood_sums.index_select(0, trg_index) + 1e-16)).unsqueeze(-1)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linears = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.Linear(hidden_dim, output_dim, bias=False)
        ])
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.relu(self.batch_norm(self.linears[0](x)))
        return self.linears[1](h)

class MLP2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linears = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.Linear(hidden_dim, output_dim, bias=False)
        ])
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.relu(self.batch_norm(self.linears[0](x)))
        return self.linears[1](h)

class RGCN4(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names, num_layers=4, n_heads=8):
        super().__init__()
        self.rel_names = rel_names
        self.num_layers = num_layers

        self.embed_layer = MLP2(in_feats, hid_feats, hid_feats)
        self.gnn_layers = nn.ModuleList([
            nn.ModuleDict({
                rel: SparseGATConv(hid_feats, hid_feats // n_heads, n_heads)
                for rel in rel_names
            }) for _ in range(num_layers)
        ])
        self.decision_layer = MLP(hid_feats * len(rel_names), hid_feats, out_feats)

    def forward(self, adjs, inputs):
        inputs = self.embed_layer(inputs)
        h = inputs
        
        for layer_idx in range(self.num_layers):
            new_h = []
            for rel in self.rel_names:
                adj = adjs[rel]
                h_rel = self.gnn_layers[layer_idx][rel](adj, h)
                new_h.append(h_rel.view(h.size(0), -1))
            h = F.leaky_relu(torch.cat(new_h, dim=1))
            h = h + inputs
        
        out = self.decision_layer(h)
        return out

# Example usage:
num_nodes = 10
in_feats = 16
hid_feats = 16
out_feats = 5
rel_names = ['rel1', 'rel2', 'rel3']
num_layers = 4
n_heads = 8
num_edges = 20

# Random sparse adjacency matrices for each relation
adjs = {rel: torch.sparse_coo_tensor(
    indices=torch.randint(0, num_nodes, (2, num_edges)), 
    values=torch.ones(num_edges), 
    size=(num_nodes, num_nodes)
) for rel in rel_names}

print(adjs['rel1'])
# Random node features
inputs = torch.randn(num_nodes, in_feats)

model = RGCN4(in_feats, hid_feats, out_feats, rel_names, num_layers, n_heads)
output = model(adjs, inputs)
print(output.shape)  # should be [num_nodes, out_feats]
