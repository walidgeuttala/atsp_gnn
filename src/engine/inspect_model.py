#!/usr/bin/env python3
"""
Estimate memory usage layer-by-layer for HetroGAT (analytical, no forward pass).

Usage: edit ckpt_path and graph_path at bottom or pass via small wrapper.
"""
import torch
import dgl
import math
from pathlib import Path
from ..utils import print_gpu_memory

# adapt these two paths to your environment
CKPT_PATH = "/project/c_gnn_001/code/tsp/atsp_gnn/jobs/runs/Sep10_23-05-57_HetroGATconcat_ATSP50_Comboss_st_tt_pp/trial_0/best_model.pt"
GRAPH_PATH = "/project/c_gnn_001/code/tsp/atsp_gnn/saved_dataset/ATSP_30x500/templates/all/template_500_pp_ss_st_tt.dgl"

def bytes_to_mb(b): return b / (1024 ** 2)
def bytes_to_gb(b): return b / (1024 ** 3)

def dtype_size(dtype: torch.dtype):
    return torch.tensor([], dtype=dtype).element_size()

class ArgsFromDict:
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)

def load_model_from_ckpt(get_model_fn, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    args_dict = ckpt.get('args', None)
    if args_dict is None:
        raise RuntimeError("Checkpoint has no 'args' entry. Provide args to reconstruct the model.")
    args = ArgsFromDict(args_dict)
    model = get_model_fn(args)   # should construct the same architecture
    model.load_state_dict(ckpt['model_state_dict'])
    return model, args

def is_gatconv_module(m):
    # very soft check for DGL GATConv-like object: has 'fc' Linear attribute and probably 'num_heads'
    return hasattr(m, 'fc') and hasattr(m.fc, 'in_features') and hasattr(m.fc, 'out_features')

def analyze_model(model, g, dtype=torch.float32, edge_buffer_factor=2.0):
    """
    edge_buffer_factor: multiplicative factor to account for extra temporary buffers
      (gathered messages, softmax buffers, intermediate copies). Default 2x conservative.
    """
    dsize = dtype_size(dtype)
    total_param_count = sum(p.numel() for p in model.parameters())
    param_mem = total_param_count * dsize

    num_nodes = g.num_nodes()
    # map etype name -> edge count
    edges_per_etype = {etype: g.num_edges(etype) for etype in g.etypes}
    total_edges = sum(edges_per_etype.values())

    report_lines = []
    running_total = param_mem

    report_lines.append(f"Nodes: {num_nodes}, edges by etype: {edges_per_etype}")
    report_lines.append(f"Model total parameters: {total_param_count} -> {bytes_to_mb(param_mem):.3f} MB")

    # --- Embed MLP (assume applied per-node) ---
    # Find embed_layer if present
    if hasattr(model, 'embed_layer'):
        embed = model.embed_layer
        # linear layers inside
        for name, module in embed.named_children():
            if isinstance(module, torch.nn.Linear):
                pcount = sum(p.numel() for p in module.parameters())
                pbytes = pcount * dsize
                # Activation is per-node: output dimension is module.out_features
                act_bytes = num_nodes * module.out_features * dsize
                running_total += pbytes + act_bytes
                report_lines.append(f"EMBED - {name}: params={pcount} ({bytes_to_mb(pbytes):.3f} MB), "
                                    f"activation_per_node_out={module.out_features}, activation={bytes_to_mb(act_bytes):.3f} MB")
        # fc_skip if present (params already counted above if it's a child)
    else:
        report_lines.append("No embed_layer detected.")

    # --- GNN layers (hetero) ---
    # model.gnn_layers is ModuleList of HeteroGraphConv (each having ModuleDict of rel->GATConv)
    if hasattr(model, 'gnn_layers'):
        for layer_idx, hetero in enumerate(model.gnn_layers):
            report_lines.append(f"\nGNN layer {layer_idx}:")
            # Try to access ModuleDict of relation modules
            # DGL's HeteroGraphConv typically stores submodules in .mods or ._modules['mods']
            mods = None
            if hasattr(hetero, 'mods'):
                mods = hetero.mods  # ModuleDict
            else:
                # try scanning children to find ModuleDict-like objects
                for nm, ch in hetero.named_children():
                    if isinstance(ch, dict) or 'mods' in nm:
                        mods = ch
                        break
                if mods is None:
                    # fallback: try hetero._modules
                    mods = getattr(hetero, '_modules', None)

            # If mods is ModuleDict, iterate
            if hasattr(mods, 'items'):
                for rel, conv in mods.items():
                    # conv is likely a GATConv
                    if is_gatconv_module(conv):
                        # param memory
                        pcount = sum(p.numel() for p in conv.parameters())
                        pbytes = pcount * dsize

                        in_feats = conv.fc.in_features
                        out_feats = conv.fc.out_features  # this is total out = out_per_head * num_heads
                        # find num_heads if available
                        num_heads = getattr(conv, 'num_heads', None) or getattr(conv, '_num_heads', None) \
                                    or getattr(conv, 'heads', None) or getattr(conv, 'num_heads', None)
                        # fallback: try to infer from args or out_feats
                        if num_heads is None or num_heads == 0:
                            # try to infer from out_feats and model.hidden_dim if available
                            num_heads = getattr(model, 'n_heads', None) or getattr(model, 'n_heads', None) or 1

                        # edges for this relation
                        ecount = edges_per_etype.get(rel, g.num_edges(rel))

                        # Message tensor: when GAT projects node features and gathers per-edge,
                        # it materializes (num_edges, out_feats) (or (num_edges, num_heads, out_per_head))
                        msg_bytes = ecount * out_feats * dsize

                        # Attention logits: roughly (num_edges * num_heads)
                        attn_bytes = ecount * int(num_heads) * dsize

                        # Account for extra temporaries (softmax buffers, duplicates) with factor:
                        edge_total_bytes = (msg_bytes + attn_bytes) * edge_buffer_factor

                        running_total += pbytes + edge_total_bytes

                        report_lines.append(
                            f"  REL '{rel}': GATConv in={in_feats}, out={out_feats}, heads={num_heads}, edges={ecount}\n"
                            f"     params={pcount} ({bytes_to_mb(pbytes):.3f} MB), "
                            f"msg_mem={bytes_to_mb(msg_bytes):.3f} MB, attn_mem={bytes_to_mb(attn_bytes):.3f} MB, "
                            f"buffered_edge_mem={bytes_to_mb(edge_total_bytes):.3f} MB"
                        )
                    else:
                        # unknown conv module: fallback to counting its params
                        pcount = sum(p.numel() for p in conv.parameters())
                        pbytes = pcount * dsize
                        running_total += pbytes
                        report_lines.append(f"  REL '{rel}': conv type {type(conv)} params={pcount} ({bytes_to_mb(pbytes):.3f} MB)")
            else:
                report_lines.append("  Could not find relation modules inside hetero conv. Skipping details.")
    else:
        report_lines.append("No gnn_layers found on model.")

    # --- MLP layers (node-wise) ---
    if hasattr(model, 'mlp_layers'):
        for i, mlp in enumerate(model.mlp_layers):
            # mlp has fc1 and fc2
            fc1 = mlp.fc1
            fc2 = mlp.fc2
            pcount = sum(p.numel() for p in mlp.parameters())
            pbytes = pcount * dsize
            # activations: fc1 output (num_nodes x fc1.out_features), fc2 output (num_nodes x fc2.out_features)
            act1 = num_nodes * fc1.out_features * dsize
            act2 = num_nodes * fc2.out_features * dsize
            running_total += pbytes + act1 + act2
            report_lines.append(f"MLP layer {i}: params={pcount} ({bytes_to_mb(pbytes):.3f} MB), "
                                f"act_fc1={bytes_to_mb(act1):.3f} MB (out={fc1.out_features}), "
                                f"act_fc2={bytes_to_mb(act2):.3f} MB (out={fc2.out_features})")
    else:
        report_lines.append("No mlp_layers found.")

    # --- decision layer (node-wise) ---
    if hasattr(model, 'decision_layer'):
        dec = model.decision_layer
        pcount = sum(p.numel() for p in dec.parameters())
        pbytes = pcount * dsize
        # fc1, fc2 expected
        if hasattr(dec, 'fc1') and hasattr(dec, 'fc2'):
            act1 = num_nodes * dec.fc1.out_features * dsize
            act2 = num_nodes * dec.fc2.out_features * dsize
            running_total += pbytes + act1 + act2
            report_lines.append(f"DECISION LAYER: params={pcount} ({bytes_to_mb(pbytes):.3f} MB), "
                                f"act_fc1={bytes_to_mb(act1):.3f} MB, act_fc2={bytes_to_mb(act2):.3f} MB")
        else:
            running_total += pbytes
            report_lines.append(f"Decision module of unknown internals: params={pcount} ({bytes_to_mb(pbytes):.3f} MB)")
    else:
        report_lines.append("No decision_layer found.")

    # Final numbers
    report_lines.append("\n=== Totals (approximate, inference-time activations only) ===")
    report_lines.append(f"Estimated peak bytes (params + activations + edge buffers): {running_total:,} bytes "
                        f"= {bytes_to_mb(running_total):.3f} MB = {bytes_to_gb(running_total):.3f} GB")

    # training multiplier
    # Conservative: training needs ~ (params + activations)* (1 + gradient + optimizer states)
    # optimizer (Adam) ~ 2x params; gradients ~ same size as params or activations depending.
    training_estimate = running_total + 2 * total_param_count * dsize  # add Adam approx
    report_lines.append(f"Rough training-time lower-bound (add optimizer states): {bytes_to_gb(training_estimate):.3f} GB")

    return "\n".join(report_lines)

if __name__ == "__main__":
    # Lazy import model factory here so script can be placed in repo root
    import importlib
    models_mod = importlib.import_module("src.models.models_dgl")
    get_dgl_model = getattr(models_mod, "get_dgl_model")
    print_gpu_memory('before')

    model, args = load_model_from_ckpt(get_dgl_model, CKPT_PATH)
    graphs, _ = dgl.load_graphs(GRAPH_PATH)
    g = graphs[0]
    print_gpu_memory('after')
    # choose dtype float32 or float64
    out = analyze_model(model, g, dtype=torch.float32, edge_buffer_factor=2.0)
    print(out)
