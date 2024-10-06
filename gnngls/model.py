import sys
import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn

from gnngls.model_utils import *

class HetroGAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, relation_types, num_gnn_layers=4, num_heads=16, aggregate='sum'):
        super().__init__()
        self.relation_types = relation_types

        self.embed_layer = MLP2(input_dim, hidden_dim, hidden_dim)

        self.gnn_layers = torch.nn.ModuleList()
        for _ in range(num_gnn_layers):
            self.gnn_layers.append(
                dglnn.HeteroGraphConv({
                    rel: dgl.nn.GATConv(hidden_dim, hidden_dim // num_heads, num_heads)
                    for rel in relation_types.split(' ')
                }, aggregate=aggregate)
            )
        if aggregate == 'sum':
            self.decision_layer = MLP(hidden_dim, hidden_dim, output_dim)
        elif aggregate == 'cat':
            self.decision_layer = MLP(hidden_dim*num_gnn_layers, hidden_dim, output_dim)

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