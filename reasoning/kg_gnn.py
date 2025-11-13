"""
A compact Knowledge Graph + lightweight GNN-like message-passing implementation.
- Builds a small KG of departments and attributes
- Encodes detection params into node features
- Does a few message passing rounds using adjacency and an MLP (PyTorch)
- Outputs scores per department (0..1)
This is deliberately small and dependency-light (networkx + torch + numpy).
"""

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Departments we care about
DEPARTMENTS = ["Waste Management", "Construction", "Municipality", "Roads", "Electricity", "Water", "Ward Office"]

class SimpleGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc_msg = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, out_dim)

    def forward(self, features, adj, steps=2):
        # features: (N, F)
        h = F.relu(self.fc1(features))
        for _ in range(steps):
            # message: adj @ h
            m = torch.matmul(adj, h)
            h = F.relu(self.fc_msg(m))
        out = torch.sigmoid(self.fc_out(h))  # (N, out_dim)
        return out

class KnowledgeGraphReasoner:
    def __init__(self):
        # Build simple KG: nodes = departments + attributes
        self.G = nx.DiGraph()
        # add department nodes
        for d in DEPARTMENTS:
            self.G.add_node(d, type='dept')
        # add attribute nodes (example)
        attrs = ["large_waste", "hazardous_waste", "deep_pothole", "large_pothole", "near_electric", "near_water"]
        for a in attrs:
            self.G.add_node(a, type='attr')
        # add edges: attr -> dept with simple prior weights (used only for structure)
        self.G.add_edge("large_waste", "Waste Management")
        self.G.add_edge("hazardous_waste", "Waste Management")
        self.G.add_edge("large_waste", "Ward Office")
        self.G.add_edge("deep_pothole", "Roads")
        self.G.add_edge("large_pothole", "Roads")
        self.G.add_edge("near_electric", "Electricity")
        self.G.add_edge("near_water", "Water")
        # For GNN we convert to adjacency later

        # create mapping node->index
        self.nodes = list(self.G.nodes())
        self.node_index = {n: i for i, n in enumerate(self.nodes)}

        # feature dim: we'll create simple one-hot for attrs + scalar numeric features for depts
        self.in_dim = 8  # small
        self.hidden_dim = 16
        self.out_dim = len(DEPARTMENTS)
        self.model = SimpleGNN(self.in_dim, self.hidden_dim, self.out_dim)

    def build_feature_matrix(self, detection_record):
        """
        Builds a feature matrix for all nodes.
        For attr nodes: binary 1/0 if the detection triggers them.
        For dept nodes: start with zeros or simple priors.
        Returns torch.FloatTensor shape (N, in_dim)
        """
        N = len(self.nodes)
        features = np.zeros((N, self.in_dim), dtype=np.float32)

        # simple detection -> attr mapping heuristics
        params = detection_record.get('params', {})
        t = detection_record.get('type')

        attr_map = {
            "large_waste": False,
            "hazardous_waste": False,
            "large_pothole": False,
            "deep_pothole": False,
            "near_electric": False,
            "near_water": False
        }

        if t == 'waste':
            primary = params.get('primary')
            if primary:
                # consider large if area_pct > 0.02
                if primary.get('area_pct', 0) > 0.02:
                    attr_map['large_waste'] = True
                # hazardous heuristic: class_name contains 'battery' or 'chemical'
                clsname = primary.get('class_name', '').lower() if primary else ''
                if 'battery' in clsname or 'chemical' in clsname:
                    attr_map['hazardous_waste'] = True
        elif t == 'pothole':
            primary = params.get('primary')
            if primary:
                if primary.get('area_pct', 0) > 0.01:
                    attr_map['large_pothole'] = True
                if (primary.get('est_depth_m') or 0) > 0.05:
                    attr_map['deep_pothole'] = True
                # proximity to edges might map to near_water/electric based on metadata in future

        # fill attr nodes
        for attr in [n for n in self.nodes if self.G.nodes[n].get('type') == 'attr']:
            idx = self.node_index[attr]
            # place binary value in feature[0], and duplicate into couple dims
            val = 1.0 if attr_map.get(attr, False) else 0.0
            features[idx, 0] = val
            features[idx, 1] = val  # duplicate small signals

        # dept nodes priors: small bias toward some departments based on type
        for dept in [n for n in self.nodes if self.G.nodes[n].get('type') == 'dept']:
            idx = self.node_index[dept]
            # set small one-hot indicating dept identity (not needed but helps)
            # map dept index modulo dims
            features[idx, 2 + (idx % (self.in_dim - 2))] = 0.1

        # convert to torch
        return torch.from_numpy(features)

    def build_adj_matrix(self):
        # undirected adjacency for message passing between nodes
        N = len(self.nodes)
        adj = np.zeros((N, N), dtype=np.float32)
        for u, v in self.G.edges():
            ui = self.node_index[u]; vi = self.node_index[v]
            adj[ui, vi] = 1.0
            adj[vi, ui] = 1.0  # make symmetric
        # add self loops
        for i in range(N):
            adj[i, i] = 1.0
        # normalize adjacency rows
        row_sums = adj.sum(axis=1, keepdims=True)
        row_sums[row_sums==0] = 1.0
        adj = adj / row_sums
        return torch.from_numpy(adj)

    def reason(self, detection_record):
        """
        Returns dict { department: score } where score in 0..1
        """
        features = self.build_feature_matrix(detection_record)  # (N, in_dim)
        adj = self.build_adj_matrix()  # (N, N)
        with torch.no_grad():
            out = self.model(features, adj, steps=2)  # (N, out_dim)
            # out per node; we want department nodes aggregated
            # department indices:
            dept_scores = np.zeros((len(self.nodes), self.out_dim), dtype=np.float32)
            out_np = out.numpy()
            # For simplicity, sum outputs across all nodes and take department columns
            # out has shape N x out_dim where out_dim == number of departments
            agg = out_np.sum(axis=0)  # shape (out_dim,)
            # normalize
            agg = (agg - agg.min()) / (agg.max() - agg.min() + 1e-8)
            # map to DEPARTMENTS
            res = {}
            for i, dept in enumerate(DEPARTMENTS):
                res[dept] = float(agg[i])
            return res
