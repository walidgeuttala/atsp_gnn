import sys
import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn

from gnngls.model_utils import *


class HetroGATSum(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, relation_types, num_gnn_layers=4, num_heads=16):
        super().__init__()
        self.relation_types = relation_types
        self.embed_layer = MLP(input_dim, hidden_dim, hidden_dim, skip=True)
        self.num_edge_types = len(relation_types.split(' '))
        self.gnn_layers = torch.nn.ModuleList()
        for _ in range(num_gnn_layers):
            self.gnn_layers.append(
                dglnn.HeteroGraphConv({
                    rel: dgl.nn.GATConv(hidden_dim, hidden_dim // num_heads, num_heads)
                    for rel in relation_types.split(' ')
                }, aggregate='sum')
            )
        self.decision_layer = MLP(hidden_dim, hidden_dim, output_dim)
        

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

class HetroGATConcat(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, relation_types, num_gnn_layers=4, num_heads=16):
        super().__init__()
        self.relation_types = relation_types
        self.embed_layer = MLP(input_dim, hidden_dim, hidden_dim, skip=True)
        self.num_edge_types = len(relation_types.split(' '))
        self.gnn_layers = torch.nn.ModuleList()
        self.mlp_layers = nn.ModuleList()

        for _ in range(num_gnn_layers):
            self.gnn_layers.append(
                dglnn.HeteroGraphConv({
                    rel: dgl.nn.GATConv(hidden_dim, hidden_dim // num_heads, num_heads)
                    for rel in relation_types.split(' ')
                }, aggregate='stack')
            )
            self.mlp_layers.append(MLP(hidden_dim*self.num_edge_types, hidden_dim*self.num_edge_types//2, hidden_dim))

        self.decision_layer = MLP(hidden_dim, hidden_dim, output_dim)
        
    def forward(self, graph, inputs):
        with graph.local_scope():
            inputs = self.embed_layer(inputs)
            h1 = {graph.ntypes[0]: inputs}
            for gnn_layer, mlp_layer in zip(self.gnn_layers, self.mlp_layers):
                h2 = gnn_layer(graph, h1)
                h2 = {k: F.leaky_relu(v).flatten(1) for k, v in h2.items()}
                h2 = {k: mlp_layer(v) for k, v in h2.items()}
                h2[graph.ntypes[0]] += h1[graph.ntypes[0]]

                h1 = h2

            h2 = self.decision_layer(torch.cat([x for x in list(h2.values())], dim=1))
            return h2

# Original
class EdgePropertyPredictionModel0(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_gnn_layers=4, num_heads=16):
    
        super().__init__()
        self.embed_layer = MLP(input_dim, hidden_dim, hidden_dim, skip=True)
        self.message_passing_layers = dgl.nn.utils.Sequential(
            *(AttentionLayer(hidden_dim, num_heads, hidden_dim*2) for _ in range(num_gnn_layers))
        )
        
        self.decision_layer = MLP(hidden_dim, hidden_dim, output_dim)
        
    def forward(self, G, x):
        h = self.embed_layer(x)
        for l in self.message_passing_layers:
            h = l(G, h)
        h = self.decision_layer(h)
        return h


# Original+SC
class EdgePropertyPredictionModel1(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_gnn_layers=4, num_heads=16):
    
        super().__init__()
        self.embed_layer = MLP(input_dim, hidden_dim, hidden_dim, skip=True)
        self.message_passing_layers = dgl.nn.utils.Sequential(
            *(AttentionLayer(hidden_dim, num_heads, hidden_dim*2) for _ in range(num_gnn_layers))
        )
        
        self.decision_layer = MLP(hidden_dim, hidden_dim, output_dim)
        
    def forward(self, G, x):
        h = self.embed_layer(x)
        for l in self.message_passing_layers:
            h = l(G, h) + h
        h = self.decision_layer(h)
        return h


import dgl
import dgl.nn.pytorch as dglnn
import torch
import torch.nn as nn

class GCNmodel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNmodel, self).__init__()
        self.conv1 = dgl.nn.GraphConv(input_dim, hidden_dim)
        self.conv2 = dgl.nn.GraphConv(hidden_dim, output_dim)

    def forward(self, G, x):
        h = F.relu(self.conv1(G, x))  # First layer with ReLU activation
        h = self.conv2(G, h)           # Second layer (no activation)
        return h  # Return the output embeddings



# Original+kj
class EdgePropertyPredictionModel2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_gnn_layers=4, num_heads=16, jk='cat'):
    
        super().__init__()
        self.embed_layer = MLP(input_dim, hidden_dim, hidden_dim, skip=True)

        self.message_passing_layers = dgl.nn.utils.Sequential(
            *(AttentionLayer(hidden_dim, num_heads, hidden_dim*2) for _ in range(num_gnn_layers))
        )
        self.jk = dgl.nn.pytorch.utils.JumpingKnowledge(jk)
        if jk == "cat":
            self.decision_layer = MLP(hidden_dim*(num_gnn_layers+1), hidden_dim, output_dim)
        else:
            self.decision_layer = MLP(hidden_dim, hidden_dim, output_dim)
        
    def forward(self, G, x):
        h = self.embed_layer(x)
        xs = []
        xs.append(h)
        for l in self.message_passing_layers:
            h = l(G, h)
            xs.append(h)
        h = self.jk(xs)
        h = self.decision_layer(h)
        return h

# Original+kj+SC
class EdgePropertyPredictionModel3(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_gnn_layers=4, num_heads=16, jk='cat'):
    
        super().__init__()
        self.embed_layer = MLP(input_dim, hidden_dim, hidden_dim, skip=True)

        self.message_passing_layers = dgl.nn.utils.Sequential(
            *(AttentionLayer(hidden_dim, num_heads, hidden_dim*2) for _ in range(num_gnn_layers))
        )
        self.jk = dgl.nn.pytorch.utils.JumpingKnowledge(jk)
        if jk == "cat":
            self.decision_layer = MLP(hidden_dim*(num_gnn_layers+1), hidden_dim, output_dim)
        else:
            self.decision_layer = MLP(hidden_dim, hidden_dim, output_dim)
        
    def forward(self, G, x):
        h = self.embed_layer(x)
        xs = []
        xs.append(h)
        for l in self.message_passing_layers:
            h = l(G, h) + h
            xs.append(h)
        h = self.jk(xs)
        h = self.decision_layer(h)
        return h


def get_model(args):
    try:
        model = getattr(sys.modules[__name__], args.model)
    except AttributeError:
        raise ValueError(f"Model '{args.model}' not recognized")
    
    # Get the constructor's signature for the model class
    model_signature = inspect.signature(model)

    # Extract only the arguments that are required by the model's constructor
    model_args = {
        param: getattr(args, param) for param in model_signature.parameters if hasattr(args, param)
    }

    return model(**model_args)