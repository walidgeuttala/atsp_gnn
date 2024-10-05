import dgl.nn as dglnn
import torch.nn as nn
import dgl
import torch
import torch.nn.functional as F

class MLP(nn.Module):
    """Construct two-layer MLP-type aggreator for GIN model"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linears = nn.ModuleList()
        # two-layer MLP
        self.linears.append(nn.Linear(input_dim, hidden_dim, bias=False))
        self.linears.append(nn.Linear(hidden_dim, output_dim, bias=False))
        self.batch_norm = nn.BatchNorm1d((hidden_dim))
        self.relu = nn.ReLU()
    def forward(self, x):
        h = x
        h = self.relu(self.batch_norm(self.linears[0](h)))
        return self.linears[1](h)
    
class MLP2(nn.Module):
    """Construct two-layer MLP-type aggreator for GIN model"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linears = nn.ModuleList()
        # two-layer MLP
        self.linears.append(nn.Linear(input_dim, hidden_dim, bias=False))
        self.linears.append(nn.Linear(hidden_dim, output_dim, bias=False))
        self.batch_norm = nn.BatchNorm1d((hidden_dim))
        self.relu = nn.ReLU()
    def forward(self, x):
        h = x
        h = self.relu(self.batch_norm(self.linears[0](h)))
        return self.linears[1](h) + h

class RGCN4(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names, num_layers = 4, n_heads = 16):
        super().__init__()
        self.rel_names = rel_names

        self.embed_layer = MLP2(in_feats, hid_feats, hid_feats)

        self.gnn_layers = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.gnn_layers.append(
                    dglnn.HeteroGraphConv({
                rel: dgl.nn.GATConv(hid_feats, hid_feats // n_heads, n_heads)
                for rel in rel_names}, aggregate='sum')
            )
        self.decision_layer = MLP(hid_feats, hid_feats, out_feats)
    # graph: hetro grpah with 5 type edges and 1 type node
    # inputs (n, 1) tensor shape
    def forward(self, graph, inputs):
        with graph.local_scope():
            inputs = self.embed_layer(inputs)
            h1 = {graph.ntypes[0]: inputs}
            for gnn_layer in self.gnn_layers:
                h2 = gnn_layer(graph, h1)
                h2 = {k: F.leaky_relu(v).flatten(1) for k, v in h2.items()}
                h2[graph.ntypes[0]] += h1[graph.ntypes[0]]
                
                h1 = h2
                
            h2 = self.decision_layer(torch.cat([x for x in list(h2.values())], dim=1))
            return h2