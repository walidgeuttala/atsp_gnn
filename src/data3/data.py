import pathlib
import pickle
import dgl
import networkx as nx
import torch
import torch.utils.data

class TSPDataset(torch.utils.data.Dataset):
    def __init__(self, instances_file, args, scalers_file=None):
        if not isinstance(instances_file, pathlib.Path):
            instances_file = pathlib.Path(instances_file)
        self.root_dir = instances_file.parent
        self.num_edges = args.atsp_size * (args.atsp_size-1)
        self.instances = [line.strip() for line in open(instances_file)]

        if scalers_file is None:
            scalers_file = self.root_dir / 'scalers.pkl'
        scalers = pickle.load(open(scalers_file, 'rb'))
        if 'edges' in scalers: # for backward compatability
            self.scalers = scalers['edges']
        else:
            self.scalers = scalers
        if 'st' in args.relation_types:
            if args.half_st:
                graphs, _ = dgl.load_graphs(f"../tsp_input/graph_{args.atsp_size}_half_st.dgl")
            else:
                graphs, _ = dgl.load_graphs(f"../tsp_input/graph_{args.atsp_size}_full_st.dgl")
        else:
            graphs, _ = dgl.load_graphs(f"../tsp_input/graph_{args.atsp_size}_none_st.dgl")

        self.G = graphs[0]
        self.es = self.G.ndata['e'].cpu().numpy()
        # Transfer the Hetro to Homo
        if args.to_homo:
            self.G = dgl.to_homogeneous(self.G, ndata=['e'])

        self.etypes = self.G.etypes

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, i):
        if torch.is_tensor(i):
            i = i.tolist()

        G = nx.read_gpickle(self.root_dir / self.instances[i])
        H = self.get_scaled_features(G)
        return H

    def get_scaled_features(self, G):
        
        
        features = torch.empty(self.num_edges, dtype=torch.float32)
        regret = torch.empty(self.num_edges, dtype=torch.float32)
        in_solution = torch.empty(self.num_edges, dtype=torch.float32)
        
        for idx, e in enumerate(self.es):
            features[idx] = G.edges[e]['weight']
            regret[idx] = G.edges[e]['regret']
            in_solution[idx] = G.edges[e]['in_solution']

        features_transformed = self.scalers['weight'].transform(features.reshape(-1, 1))
        regret_transformed = self.scalers['regret'].transform(regret.reshape(-1, 1))
        
        H = self.G
        H.ndata['weight'] = torch.tensor(features_transformed, dtype=torch.float32)
        H.ndata['regret'] = torch.tensor(regret_transformed, dtype=torch.float32)
        H.ndata['in_solution'] = in_solution.reshape(-1, 1)

        return H
