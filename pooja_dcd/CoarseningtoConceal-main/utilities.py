

import random
import numpy as np
import torch
from torch_geometric.datasets import TUDataset
#from torch_geometric.transforms import Compose, Complete, OneHotDegree
from torch_geometric.data import DataLoader, Data
from torch_geometric.utils import dense_to_sparse, degree, to_networkx
from sklearn.model_selection import train_test_split
from collections import Counter
import os.path as osp
import pandas as pd
#coarsen_A_dataset from privcy folder inside coasrsening.py
from privacy.coarsening import condense_A_dataset

import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.transforms import OneHotDegree
import torch
from torch_geometric.utils import to_networkx, degree
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from models.GCN import GCN
from models.serverGCN import serverGCN
from server.default import Server
from client.default import Client_GC


from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import OneHotDegree
import os.path as osp



import random
import numpy as np
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import OneHotDegree
from torch_geometric.data import DataLoader
from torch_geometric.utils import degree, to_networkx
from sklearn.model_selection import train_test_split
import pandas as pd
import os.path as osp

from privacy.coarsening import condense_A_dataset
from models.GCN import GCN
from models.serverGCN import serverGCN
from server.default import Server
from client.default import Client_GC

def prepare_packed_data(datapath, data, convert_x=False, new_param=None):
    if data in ["COLLAB", "IMDB-BINARY", "IMDB-MULTI"]:
        pre_transform = OneHotDegree(491 if data == "COLLAB" else (135 if data == "IMDB-BINARY" else 88), cat=False)
        tudataset = TUDataset(root=datapath, name=data, pre_transform=pre_transform)
    elif data in ['REDDIT-BINARY', 'REDDIT-MULTI-5K']:
        tudataset = TUDataset(root=datapath, name=data)
        maxdegree = get_maxDegree(tudataset)
        tudataset = TUDataset(root=datapath, name=data, transform=OneHotDegree(maxdegree, cat=False))
        num_nodes = min(200, max([d.num_nodes for d in tudataset]))
        indices = [i for i, d in enumerate(tudataset) if d.num_nodes <= num_nodes]
        tudataset = tudataset[torch.tensor(indices)]
    elif data in ["MUTAG", "BZR", "COX2", "DHFR", "PTC_MR", "AIDS", "NCI1"]:
        tudataset = TUDataset(root=datapath, name=data)
        if convert_x:
            maxdegree = get_maxDegree(tudataset)
            tudataset = TUDataset(root=datapath, name=data, transform=OneHotDegree(maxdegree, cat=False))
    elif data in ["ENZYMES", "DD", "PROTEINS"]:
        tudataset = TUDataset(root=datapath, name=data)
    else:
        raise ValueError(f"Dataset {data} not supported")

    # Example usage of new_param (modify according to your specific use case)
    if new_param is not None:
        # Example: Perform some operation with new_param
        print(f"New parameter value: {new_param}")

    graphs = [x for x in tudataset]
    print(f"  ** {data}: {len(graphs)} graphs")
    
    return tudataset, graphs



def _randChunk(graphs, num_client, overlap, seed=None):
    random.seed(seed)
    np.random.seed(seed)

    totalNum = len(graphs)
    minSize = min(50, int(totalNum / num_client))
    graphs_chunks = []
    if not overlap:
        for i in range(num_client):
            graphs_chunks.append(graphs[i * minSize:(i + 1) * minSize])
        for g in graphs[num_client * minSize:]:
            idx_chunk = np.random.randint(low=0, high=num_client, size=1)[0]
            graphs_chunks[idx_chunk].append(g)
    else:
        sizes = np.random.randint(low=50, high=150, size=num_client)
        for s in sizes:
            graphs_chunks.append(random.choices(graphs, k=s))
    return graphs_chunks

def prepareData_oneDS(datapath, data, num_client, batchSize, convert_x=False, seed=None, overlap=False, cr=False, cr_ratio=0):
    tudataset, graphs = prepare_packed_data(datapath, data, convert_x)
    graphs_chunks = _randChunk(graphs, num_client, overlap, seed=seed)
    splitedData = {}
    df = pd.DataFrame()
    num_node_features = graphs[0].num_node_features
    if cr:
        coarsen_params = define_coarsen_params(data)
        packed_data = (tudataset, graphs)
    for idx, chunks in enumerate(graphs_chunks):
        ds = f'{idx}-{data}'
        ds_tvt = chunks
        ds_train, ds_vt = split_data(ds_tvt, train=0.8, test=0.2, shuffle=True, seed=seed)
        ds_val, ds_test = split_data(ds_vt, train=0.5, test=0.5, shuffle=True, seed=seed)
        if cr:
            ds_train = condense_A_dataset(ds_train, coarsen_params=coarsen_params, cr_ratio=cr_ratio, args=args, packed_data=packed_data)
        dataloader_train = DataLoader(ds_train, batch_size=batchSize, shuffle=True)
        dataloader_val = DataLoader(ds_val, batch_size=batchSize, shuffle=True)
        dataloader_test = DataLoader(ds_test, batch_size=batchSize, shuffle=True)
        num_graph_labels = get_numGraphLabels(ds_train)
        splitedData[ds] = ({'train': dataloader_train, 'val': dataloader_val, 'test': dataloader_test},
                           num_node_features, num_graph_labels, len(ds_train))
        df = get_stats(df, ds, ds_train, graphs_val=ds_val, graphs_test=ds_test)

    return splitedData, df

def prepareData_multiDS(datapath, group='small', batchSize=32, convert_x=False, seed=None, cr=False, cr_ratio=0):
    datasets = {
        'molecules': ["MUTAG", "BZR", "COX2", "DHFR", "PTC_MR", "AIDS", "NCI1"],
        'proteins': ["ENZYMES", "DD", "PROTEINS"],
        'social': ["COLLAB", "IMDB-BINARY", "IMDB-MULTI", "REDDIT-BINARY", "REDDIT-MULTI-5K"],
        'md1': ["MUTAG", "BZR", "COX2", "DHFR", "PTC_MR", "ENZYMES", "DD", "PROTEINS"],
        'md2': ["MUTAG", "BZR", "COX2", "DHFR", "PTC_MR", "COLLAB", "IMDB-BINARY", "IMDB-MULTI", "REDDIT-BINARY", "REDDIT-MULTI-5K"],
        'mol': ["MUTAG", "BZR", "COX2", "DHFR", "PTC_MR", "AIDS", "NCI1"],
        'small': ["MUTAG", "BZR", "COX2", "DHFR", "PTC_MR", "ENZYMES", "DD", "PROTEINS"],
        'mix': ["MUTAG", "BZR", "COX2", "DHFR", "PTC_MR", "AIDS", "NCI1", "ENZYMES", "DD", "PROTEINS", "COLLAB", "IMDB-BINARY", "IMDB-MULTI"],
        'biochem': ["MUTAG", "BZR", "COX2", "DHFR", "PTC_MR", "AIDS", "NCI1", "ENZYMES", "DD", "PROTEINS"]
    }[group]
    if group.endswith('tiny'):
        datasets = datasets[:5]

    splitedData = {}
    df = pd.DataFrame()
    for data in datasets:
        tudataset, graphs = prepare_packed_data(datapath, data, convert_x)

        graphs_train, graphs_valtest = split_data(graphs, test=0.2, shuffle=True, seed=seed)
        graphs_val, graphs_test = split_data(graphs_valtest, train=0.5, test=0.5, shuffle=True, seed=seed)
        if cr:
            coarsen_params = define_coarsen_params(data)
            packed_data = (tudataset, graphs)
            graphs_train = condense_A_dataset(graphs_train, coarsen_params=coarsen_params, cr_ratio=cr_ratio, args=args, packed_data=packed_data)
        if group.endswith('tiny'):
            graphs, _ = split_data(graphs, train=150, shuffle=True, seed=seed)
            graphs_train, graphs_valtest = split_data(graphs, test=0.2, shuffle=True, seed=seed)
            graphs_val, graphs_test = split_data(graphs_valtest, train=0.5, test=0.5, shuffle=True, seed=seed)

        num_node_features = graphs[0].num_node_features
        dataloader_train = DataLoader(graphs_train, batch_size=batchSize, shuffle=True)
        dataloader_val = DataLoader(graphs_val, batch_size=batchSize, shuffle=True)
        dataloader_test = DataLoader(graphs_test, batch_size=batchSize, shuffle=True)
        num_graph_labels = get_numGraphLabels(graphs_train)
        splitedData[data] = ({'train': dataloader_train, 'val': dataloader_val, 'test': dataloader_test},
                             num_node_features, num_graph_labels, len(graphs_train))
        df = get_stats(df, data, graphs_train, graphs_val=graphs_val, graphs_test=graphs_test)

    return splitedData, df
# def prepare_packed_data(datapath, data, convert_x=False):
#     if data in ["COLLAB", "IMDB-BINARY", "IMDB-MULTI"]:
#         pre_transform = OneHotDegree(491 if data == "COLLAB" else (135 if data == "IMDB-BINARY" else 88), cat=False)
#         tudataset = TUDataset(root=datapath, name=data, pre_transform=pre_transform)
#     elif data in ['REDDIT-BINARY', 'REDDIT-MULTI-5K']:
#         tudataset = TUDataset(root=datapath, name=data)
#         maxdegree = get_maxDegree(tudataset)
#         tudataset = TUDataset(root=datapath, name=data, transform=OneHotDegree(maxdegree, cat=False))
#         num_nodes = min(200, max([d.num_nodes for d in tudataset]))
#         indices = [i for i, d in enumerate(tudataset) if d.num_nodes <= num_nodes]
#         tudataset = tudataset[torch.tensor(indices)]
#     elif data in ["MUTAG", "BZR", "COX2", "DHFR", "PTC_MR", "AIDS", "NCI1"]:
#         tudataset = TUDataset(root=datapath, name=data)
#         if convert_x:
#             maxdegree = get_maxDegree(tudataset)
#             tudataset = TUDataset(root=datapath, name=data, transform=OneHotDegree(maxdegree, cat=False))
#     elif data in ["ENZYMES", "DD", "PROTEINS"]:
#         tudataset = TUDataset(root=datapath, name=data)
#     else:
#         raise ValueError(f"Dataset {data} not supported")

#     graphs = [x for x in tudataset]
#     print(f"  ** {data}: {len(graphs)} graphs")
    
#     return tudataset, graphs


# def _randChunk(graphs, num_client, overlap, seed=None):
#     random.seed(seed)
#     np.random.seed(seed)

#     totalNum = len(graphs)
#     minSize = min(50, int(totalNum / num_client))
#     graphs_chunks = []
#     if not overlap:
#         for i in range(num_client):
#             graphs_chunks.append(graphs[i * minSize:(i + 1) * minSize])
#         for g in graphs[num_client * minSize:]:
#             idx_chunk = np.random.randint(low=0, high=num_client, size=1)[0]
#             graphs_chunks[idx_chunk].append(g)
#     else:
#         sizes = np.random.randint(low=50, high=150, size=num_client)
#         for s in sizes:
#             graphs_chunks.append(random.choices(graphs, k=s))
#     return graphs_chunks


# def prepareData_oneDS(datapath, data, num_client, batchSize, convert_x=False, seed=None, overlap=False, cr=False, cr_ratio=0):
#     tudataset, graphs = prepare_packed_data(datapath, data, convert_x)
#     graphs_chunks = _randChunk(graphs, num_client, overlap, seed=seed)
#     splitedData = {}
#     df = pd.DataFrame()
#     num_node_features = graphs[0].num_node_features
#     if cr:
#         coarsen_params = define_coarsen_params(data)
#         packed_data = (tudataset, graphs)
#     for idx, chunks in enumerate(graphs_chunks):
#         ds = f'{idx}-{data}'
#         ds_tvt = chunks
#         ds_train, ds_vt = split_data(ds_tvt, train=0.8, test=0.2, shuffle=True, seed=seed)
#         ds_val, ds_test = split_data(ds_vt, train=0.5, test=0.5, shuffle=True, seed=seed)
#         if cr:
#             ds_train = condense_A_dataset(ds_train, coarsen_params=coarsen_params, cr_ratio=cr_ratio, args=args, packed_data=packed_data)
#         dataloader_train = DataLoader(ds_train, batch_size=batchSize, shuffle=True)
#         dataloader_val = DataLoader(ds_val, batch_size=batchSize, shuffle=True)
#         dataloader_test = DataLoader(ds_test, batch_size=batchSize, shuffle=True)
#         num_graph_labels = get_numGraphLabels(ds_train)
#         splitedData[ds] = ({'train': dataloader_train, 'val': dataloader_val, 'test': dataloader_test},
#                            num_node_features, num_graph_labels, len(ds_train))
#         df = get_stats(df, ds, ds_train, graphs_val=ds_val, graphs_test=ds_test)

#     return splitedData, df


# def prepareData_multiDS(datapath, group='small', batchSize=32, convert_x=False, seed=None, cr=False, cr_ratio=0):
#     datasets = {
#         'molecules': ["MUTAG", "BZR", "COX2", "DHFR", "PTC_MR", "AIDS", "NCI1"],
#         'proteins': ["ENZYMES", "DD", "PROTEINS"],
#         'social': ["COLLAB", "IMDB-BINARY", "IMDB-MULTI", "REDDIT-BINARY", "REDDIT-MULTI-5K"],
#         'md1': ["MUTAG", "BZR", "COX2", "DHFR", "PTC_MR", "ENZYMES", "DD", "PROTEINS"],
#         'md2': ["MUTAG", "BZR", "COX2", "DHFR", "PTC_MR", "COLLAB", "IMDB-BINARY", "IMDB-MULTI", "REDDIT-BINARY", "REDDIT-MULTI-5K"],
#         'mol': ["MUTAG", "BZR", "COX2", "DHFR", "PTC_MR", "AIDS", "NCI1"],
#         'small': ["MUTAG", "BZR", "COX2", "DHFR", "PTC_MR", "ENZYMES", "DD", "PROTEINS"],
#         'mix': ["MUTAG", "BZR", "COX2", "DHFR", "PTC_MR", "AIDS", "NCI1", "ENZYMES", "DD", "PROTEINS", "COLLAB", "IMDB-BINARY", "IMDB-MULTI"],
#         'biochem': ["MUTAG", "BZR", "COX2", "DHFR", "PTC_MR", "AIDS", "NCI1", "ENZYMES", "DD", "PROTEINS"]
#     }[group]
#     if group.endswith('tiny'):
#         datasets = datasets[:5]

#     splitedData = {}
#     df = pd.DataFrame()
#     for data in datasets:
#         tudataset, graphs = prepare_packed_data(datapath, data, convert_x)

#         graphs_train, graphs_valtest = split_data(graphs, test=0.2, shuffle=True, seed=seed)
#         graphs_val, graphs_test = split_data(graphs_valtest, train=0.5, test=0.5, shuffle=True, seed=seed)
#         if cr:
#             coarsen_params = define_coarsen_params(data)
#             packed_data = (tudataset, graphs)
#             graphs_train = condense_A_dataset(graphs_train, coarsen_params=coarsen_params, cr_ratio=cr_ratio, args=args, packed_data=packed_data)
#         if group.endswith('tiny'):
#             graphs, _ = split_data(graphs, train=150, shuffle=True, seed=seed)
#             graphs_train, graphs_valtest = split_data(graphs, test=0.2, shuffle=True, seed=seed)
#             graphs_val, graphs_test = split_data(graphs_valtest, train=0.5, test=0.5, shuffle=True, seed=seed)

#         num_node_features = graphs[0].num_node_features
#         num_graph_labels = get_numGraphLabels(graphs_train)

#         dataloader_train = DataLoader(graphs_train, batch_size=batchSize, shuffle=True)
#         dataloader_val = DataLoader(graphs_val, batch_size=batchSize, shuffle=True)
#         dataloader_test = DataLoader(graphs_test, batch_size=batchSize, shuffle=True)

#         splitedData[data] = ({'train': dataloader_train, 'val': dataloader_val, 'test': dataloader_test},
#                              num_node_features, num_graph_labels, len(graphs_train))

#         df = get_stats(df, data, graphs_train, graphs_val=graphs_val, graphs_test=graphs_test)
#     return splitedData, df





def setup_devices(splitedData, args):
    idx_clients = {}
    clients = []
    for idx, ds in enumerate(splitedData.keys()):
        idx_clients[idx] = ds
        dataloaders, num_node_features, num_graph_labels, train_size = splitedData[ds]
        cmodel_gc = GCN(num_node_features, args.hidden, num_graph_labels, args.nlayer, args.dropout)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, cmodel_gc.parameters()), lr=args.lr, weight_decay=args.weight_decay)
        clients.append(Client_GC(cmodel_gc, idx, ds, train_size, dataloaders, optimizer, args))

    smodel = serverGCN(nlayer=args.nlayer, nhid=args.hidden)
    server = Server(smodel, args.device)
    return clients, server, idx_clients


def convert_to_nodeDegreeFeatures(graphs):
    graph_infos = []
    maxdegree = 0
    for i, graph in enumerate(graphs):
        g = to_networkx(graph, to_undirected=True)
        gdegree = max(dict(g.degree).values())
        if gdegree > maxdegree:
            maxdegree = gdegree
        graph_infos.append((graph, g.degree, graph.num_nodes))  # (graph, node_degrees, num_nodes)

    new_graphs = []
    for i, tuple in enumerate(graph_infos):
        idx, x = tuple[0].edge_index[0], tuple[0].x
        deg = degree(idx, tuple[2], dtype=torch.long)
        deg = F.one_hot(deg, num_classes=maxdegree + 1).to(torch.float)

        new_graph = tuple[0].clone()
        new_graph.__setitem__('x', deg)
        new_graphs.append(new_graph)

    return new_graphs

def get_maxDegree(graphs):
    maxdegree = 0
    for i, graph in enumerate(graphs):
        g = to_networkx(graph, to_undirected=True)
        gdegree = max(dict(g.degree).values())
        if gdegree > maxdegree:
            maxdegree = gdegree

    return maxdegree

def use_node_attributes(graphs):
    num_node_attributes = graphs.num_node_attributes
    new_graphs = []
    for i, graph in enumerate(graphs):
        new_graph = graph.clone()
        new_graph.__setitem__('x', graph.x[:, :num_node_attributes])
        new_graphs.append(new_graph)
    return new_graphs

def split_data(graphs, train=None, test=None, shuffle=True, seed=None):
    y = torch.cat([graph.y for graph in graphs])
    graphs_tv, graphs_test = train_test_split(graphs, train_size=train, test_size=test, stratify=y, shuffle=shuffle, random_state=seed)
    return graphs_tv, graphs_test


def get_numGraphLabels(dataset):
    s = set()
    for g in dataset:
        s.add(g.y.item())
    return len(s)


def _get_avg_nodes_edges(graphs):
    numNodes = 0.
    numEdges = 0.
    numGraphs = len(graphs)
    for g in graphs:
        numNodes += g.num_nodes
        numEdges += g.num_edges / 2.  # undirected
    return numNodes / numGraphs, numEdges / numGraphs


def get_stats(df, ds, graphs_train, graphs_val=None, graphs_test=None):
    df.loc[ds, "#graphs_train"] = len(graphs_train)
    avgNodes, avgEdges = _get_avg_nodes_edges(graphs_train)
    df.loc[ds, 'avgNodes_train'] = avgNodes
    df.loc[ds, 'avgEdges_train'] = avgEdges

    if graphs_val:
        df.loc[ds, '#graphs_val'] = len(graphs_val)
        avgNodes, avgEdges = _get_avg_nodes_edges(graphs_val)
        df.loc[ds, 'avgNodes_val'] = avgNodes
        df.loc[ds, 'avgEdges_val'] = avgEdges

    if graphs_test:
        df.loc[ds, '#graphs_test'] = len(graphs_test)
        avgNodes, avgEdges = _get_avg_nodes_edges(graphs_test)
        df.loc[ds, 'avgNodes_test'] = avgNodes
        df.loc[ds, 'avgEdges_test'] = avgEdges

    return df

class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data


def get_dataset_node(name, sparse=True, dirname=None):
    if dirname is None:
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    else:
        path = osp.join(dirname, name)
    if name == "QM9":
        dataset = QM9(path)
    elif name == "QM7b":
        dataset = QM7b(path)
    else:
        dataset = TUDataset(path, name)
    dataset.data.edge_attr = None

    if dataset.data.x is None:
        max_degree = 0
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())

        if max_degree < 1000:
            dataset.transform = OneHotDegree(max_degree)
        else:
            deg = torch.cat(degs, dim=0).to(torch.float)
            mean, std = deg.mean().item(), deg.std().item()
            dataset.transform = NormalizedDegree(mean, std)


    num_nodes = max_num_nodes = 0
    for data in dataset:
        num_nodes += data.num_nodes
        max_num_nodes = max(data.num_nodes, max_num_nodes)

    if name == 'REDDIT-BINARY':
        num_nodes = min(int(num_nodes / len(dataset) * 1.5), max_num_nodes)
    else:
        num_nodes = min(int(num_nodes / len(dataset) * 5), max_num_nodes)

    indices = []
    for i, data in enumerate(dataset):
        if data.num_nodes <= num_nodes:
            indices.append(i)
    dataset = dataset[torch.tensor(indices)]

    if not sparse:
        if dataset.transform is None:
            dataset.transform = ToDense(num_nodes)
        else:
            dataset.transform = Compose(
                [dataset.transform, ToDense(num_nodes)])
    return dataset

def define_coarsen_params(data):
    if data == 'PROTEINS':
        return [0.01, 0.1, 0.1, 0.1]
    elif data == 'MUTAG':
        return [1, 10, 10, 0.01]
    elif data == 'AIDS':
        return [0.1, 10, 1, 10]
    elif data == 'NCI1':
        return [0.01, 10, 10, 10]
    elif data == 'PTC_MR':
        return [10, 1, 0.01, 0.01]
    elif data == 'ENZYMES':
        return [10, 0.1, 10, 0.01]
    elif data == 'BZR':
        return [0.1, 1, 1, 10]
    elif data == 'DLHR':
        return [0.01, 0.01, 0.01, 0.01]
    elif data == 'DFHR':
        return [1, 10, 1, 0.1]
    elif data == 'COX2':
        return [0.01, 0.01, 0.01, 0.01]
    elif data == 'DD':
        return [0.01, 0.1, 1, 10]
    elif 'REDDIT-BINARY' in data:
        return [1, 1, 0.1, 0.1]
    elif data == 'REDDIT-MULTI-5K':
        return [0.01, 0.1, 0.1, 10]
    else:
        return [0.01, 0.01, 0.01, 0.01]

