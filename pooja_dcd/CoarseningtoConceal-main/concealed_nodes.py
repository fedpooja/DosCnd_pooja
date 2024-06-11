import torch
from torch_geometric.datasets import TUDataset
from utilities import get_dataset_node as get_dataset

import matplotlib.pyplot as plt
import seaborn as sns


import random
from privacy.coarsening import coarsening
import csv
import numpy as np
from tqdm import tqdm
dataset_name=['PROTEINS']
datasets_all=[]
for d in dataset_name:
    dataset = get_dataset(name=d,sparse=False,dirname=None)
    datasets_all.append(dataset)
    print()
    print(f'Dataset: {dataset}:')
    print('====================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')

    data = dataset[110]  # Get the first graph object.

    print()
    print(data)
    print('=============================================================')

    # Gather some statistics about the first graph.
    print(f'Number of nodes: {data.num_nodes}')
    # print(f'Number of edges: {data.num_edges}')
    # print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    Total_Num = 0
    for data in dataset:
        Total_Num+= data.num_nodes

    fields= ['dataset','ratio','average','std','percent']
    filename = f"outputs/{d}_pure_nodes_percent.csv"

    cnt = random.randint(0,188)
            # Load the graph
    # for ratio in [0.3,0.4,0.5,0.6,0.7]:
    refer_dict = []
    all_distirbutions = []
    for ratio in [0.9,0.7,0.5,0.3,0.1]:
        num_of_pure_nodes = []
        distribution = []
        for cnt in tqdm(range(len(dataset))):

            G = dataset[cnt]
            adj_matrix = G.adj
            X=G.x
            adj = torch.Tensor(adj_matrix)

            # print(adj.shape,X.shape)
            # adj_coarsen,Xc,C = coarsening(ratio,adj,X,0.01,0.01,0.01,0.01)
            adj_coarsen,Xc,C = coarsening(adj,X,0.01,0.01,0.01,0.01, ratio, return_c=True)
            
            # print(C.shape,adj.shape)
            C[C<0.01]=0
            C[C>=0.01]=1
            src_rln = torch.count_nonzero(C,dim=0)

            pure_nodes = src_rln.tolist().count(1)
            num_of_pure_nodes.append(pure_nodes)
            src_rln = src_rln.tolist()
            while 0 in src_rln:
                src_rln.remove(0)
            distribution = distribution+src_rln
            
        all_distirbutions.append(distribution)
        
        # plt.figure(figsize=(10, 6))
        # sns.histplot(distribution, kde=False, bins=range(min(distribution), max(distribution) + 2), discrete=True)
        # plt.show()

        refer_dict.append({"dataset":f"{d}","ratio":ratio,"average":np.average(num_of_pure_nodes),"std":np.std(num_of_pure_nodes),'percent': np.sum(num_of_pure_nodes)/Total_Num*100})
        
        print("at ratio : ",ratio," Average Number of pure Nodes ", np.average(num_of_pure_nodes), "std :", np.std(num_of_pure_nodes),'percent: ', np.sum(num_of_pure_nodes)/Total_Num*100 )

    plt.figure(figsize=(10, 6))
    plt.hist(all_distirbutions, bins=20, label=['0.9', '0.7', '0.5', '0.3','0.1'], alpha=0.7)
    plt.legend(loc='upper right')
    plt.show()

    with open(filename, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
    
        writer.writeheader()
        writer.writerows(refer_dict)