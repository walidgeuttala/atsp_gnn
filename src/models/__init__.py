from .gnn_model import *
from .hetero_gat import *
from .rgcn4 import *
from .model_utils import *

def get_model(name, **kwargs):
    if name.lower() == "gnn":
        from .gnn_model import GNNModel
        return GNNModel(**kwargs)
    elif name.lower() == "rgcn4":
        from .rgcn4 import RGCN4
        return RGCN4(**kwargs)
    elif name.lower().startswith("hetero"):
        from .hetero_gat import get_hetero_model
        return get_hetero_model(name, **kwargs)
    else:
        raise ValueError(f"Unknown model: {name}")
