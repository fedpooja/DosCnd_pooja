import os
import torch
import random
import numpy as np
from torch_geometric.data import Data, Batch
from torch_geometric.utils import dense_to_sparse, to_dense_adj, to_dense_batch
from collections import Counter



def condense(args, packed_data):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    device = f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'
    torch.cuda.set_device(args.gpu_id)
    
    agent = GraphAgent(data=packed_data, args=args, device=device, nnodes_syn=get_mean_nodes(args))
    agent.train()
    
    if args.save:
        torch.save([agent.adj_syn, agent.feat_syn], f'saved/{args.dataset}_ipc{args.ipc}_s{args.seed}_lra{args.lr_adj}_lrf{args.lr_feat}.pt')
    
    return agent.adj_syn, agent.feat_syn

def condense_A_dataset(data, coarsen_params, cr_ratio, args, packed_data):
    condensed_results = {}
    
    args.dataset = data
    args.reduction_rate = cr_ratio
    
    print(f"Condensing dataset: {data} with coarsening ratio: {cr_ratio}")
    
    adj_syn, feat_syn = condense(args, packed_data)
    print(f"Completed condensation for dataset: {data} with coarsening ratio: {cr_ratio}")
    
    condensed_data = []
    for i in range(len(adj_syn)):
        edge_index, edge_attr = dense_to_sparse(adj_syn[i])
        g = Data(x=feat_syn[i], edge_index=edge_index, edge_attr=edge_attr)
        condensed_data.append(g)
    
    condensed_results[cr_ratio] = condensed_data
    
    return condensed_results


class GraphAgent:
    def __init__(self, data, args, device, nnodes_syn=75):
        self.data = data
        self.args = args
        self.device = device
        labels_train = [x.y.item() for x in data[0]]

        print('training size:', len(labels_train))
        nfeat = data[0][0].num_features
        nclass = len(set(labels_train))

        self.prepare_train_indices()

        self.labels_syn = self.get_labels_syn(labels_train)
        if args.ipc == 0:
            n = int(len(labels_train) * args.reduction_rate)
        else:
            self.labels_syn = torch.LongTensor([[i] * args.ipc for i in range(nclass)]).to(device).view(-1)
            self.syn_class_indices = {i: [i * args.ipc, (i + 1) * args.ipc] for i in range(nclass)}
            n = args.ipc * nclass

        self.adj_syn = torch.rand(size=(n, nnodes_syn, nnodes_syn), dtype=torch.float, requires_grad=True, device=device)
        self.feat_syn = torch.rand(size=(n, nnodes_syn, nfeat), dtype=torch.float, requires_grad=True, device=device)

        if args.init == 'real':
            for c in range(nclass):
                ind = self.syn_class_indices[c]
                feat_real, adj_real = self.get_graphs(c, batch_size=ind[1] - ind[0], max_node_size=nnodes_syn, to_dense=True)
                self.feat_syn.data[ind[0]: ind[1]] = feat_real[:, :nnodes_syn].detach().data
                self.adj_syn.data[ind[0]: ind[1]] = adj_real[:, :nnodes_syn, :nnodes_syn].detach().data
            self.sparsity = self.adj_syn.mean().item()
            if args.stru_discrete:
                self.adj_syn.data.copy_(self.adj_syn * 10 - 5)
        else:
            if args.stru_discrete:
                adj_init = torch.log(self.adj_syn) - torch.log(1 - self.adj_syn)
                adj_init = adj_init.clamp(-10, 10)
                self.adj_syn.data.copy_(adj_init)

        print('adj.shape:', self.adj_syn.shape, 'feat.shape:', self.feat_syn.shape)
        self.optimizer_adj = torch.optim.Adam([self.adj_syn], lr=args.lr_adj)
        self.optimizer_feat = torch.optim.Adam([self.feat_syn], lr=args.lr_feat)
        self.weights = []

    def prepare_train_indices(self):
        dataset = self.data[0]
        indices_class = {}
        nnodes_all = []
        for ix, single in enumerate(dataset):
            c = single.y.item()
            if c not in indices_class:
                indices_class[c] = [ix]
            else:
                indices_class[c].append(ix)
            nnodes_all.append(single.num_nodes)

        self.nnodes_all = np.array(nnodes_all)
        self.real_indices_class = indices_class

    def get_labels_syn(self, labels_train):
        counter = Counter(labels_train)
        num_class_dict = {}
        n = len(labels_train)

        sorted_counter = sorted(counter.items(), key=lambda x: x[1])
        sum_ = 0
        labels_syn = []
        self.syn_class_indices = {}

        for ix, (c, num) in enumerate(sorted_counter):
            if ix == len(sorted_counter) - 1:
                num_class_dict[c] = int(n * self.args.reduction_rate) - sum_
                self.syn_class_indices[c] = [len(labels_syn), len(labels_syn) + num_class_dict[c]]
                labels_syn += [c] * num_class_dict[c]
            else:
                num_class_dict[c] = max(int(num * self.args.reduction_rate), 1)
                sum_ += num_class_dict[c]
                self.syn_class_indices[c] = [len(labels_syn), len(labels_syn) + num_class_dict[c]]
                labels_syn += [c] * num_class_dict[c]

        self.num_class_dict = num_class_dict
        return torch.LongTensor(labels_syn).to(self.device)

    def get_graphs(self, c, batch_size, max_node_size=None, to_dense=False, idx_selected=None):
        if idx_selected is None:
            if max_node_size is None:
                idx_shuffle = np.random.permutation(self.real_indices_class[c])[:batch_size]
                sampled = self.data[4][idx_shuffle]
            else:
                indices = np.array(self.real_indices_class[c])[self.nnodes_all[self.real_indices_class[c]] <= max_node_size]
                idx_shuffle = np.random.permutation(indices)[:batch_size]
                sampled = self.data[4][idx_shuffle]
        else:
            sampled = self.data[4][idx_selected]
        
        data = Batch.from_data_list(sampled)
        if to_dense:
            x, edge_index, batch = data.x, data.edge_index, data.batch
            x, mask = to_dense_batch(x, batch=batch, max_num_nodes=max_node_size)
            adj = to_dense_adj(edge_index, batch=batch, max_num_nodes=max_node_size)
            return x.to(self.device), adj.to(self.device)
        else:
            return data.to(self.device)