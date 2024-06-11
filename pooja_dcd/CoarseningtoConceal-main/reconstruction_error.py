
import torch
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import OneHotDegree
from utilities import get_maxDegree
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_dir = "./data"
os.makedirs(data_dir, exist_ok=True)

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
import seaborn as sns
from torch_geometric.utils import to_dense_adj
import os
import torch

import pandas as pd
import numpy as np
from torch.utils import data
from scipy.stats import rv_continuous
from privacy.coarsening import get_laplacian, coarsening
torch.manual_seed(124)


class CustomDistribution(rv_continuous):
        def _rvs(self,  size=None, random_state=None):
            return random_state.standard_normal(size)


def Drichlet_Energy(l1,l2):
    diff = l1 - l2
    return np.linalg.norm(diff, ord='fro')
def preprocess(X, edge_index):
  adj = to_dense_adj(edge_index)
  adj = adj[0]
  # X = x
  X = X.to_dense()
  # N = X.shape[0]
  N = min(X.shape[0], adj.shape[0])

  nn = int(1*N)
  X = X[:nn,:]
  adj = adj[:nn,:nn]
  features = X.numpy()
  return X,adj


def run(data,ratio):
  if data=='PROTEINS':
      c_param=[0.01,0.1,0.1,0.1]
  elif data=='MUTAG':
      # c_param=[10,0.1,0.01,1]
      c_param=[1,10,10,0.01]

  elif data=='AIDS':
      c_param=[0.1,10,1,10]
  elif data=='NCI1':
      c_param=[0.01,10,10,10]
  elif data=='PTC_MR':
      c_param=[10,1,0.01,0.01]
  elif data=='ENZYMES':
      c_param=[10,0.1,10,0.01]
  elif data=='BZR':
      c_param=[0.1,1,1,10]
  elif data=='DLHR':
      c_param=[0.01,0.01,0.01,0.01]
  elif data=='DFHR':
      c_param=[1,10,1,0.1]
  elif data=='COX2':
      c_param=[0.01,0.01,0.01,0.01]
  elif data=='DD':
      c_param=[0.01, 0.1, 1, 10]
  elif 'REDDIT-BINARY' in data:
      c_param= [1, 1, 0.1, 0.1]
  elif data=='REDDIT-MULTI-5K':
      c_param=[0.01, 0.1, 0.1, 10]
  else:
      c_param=[0.01,0.01,0.01,0.01]

  if data == "COLLAB":
      dataset1 = TUDataset(f"data/TUDataset", data, pre_transform=OneHotDegree(491, cat=False))
  elif data == "IMDB-BINARY":
      dataset1 = TUDataset(f"data/TUDataset", data, pre_transform=OneHotDegree(135, cat=False))
  elif data == "IMDB-MULTI":
      dataset1 = TUDataset(f"data/TUDataset", data, pre_transform=OneHotDegree(88, cat=False))
  elif data == 'REDDIT-BINARY' or data == 'REDDIT-MULTI-5K':
      dataset1 = TUDataset(f"data/TUDataset", data)
      maxdegree = get_maxDegree(dataset1)
      dataset1 = TUDataset(f"data/TUDataset", data, transform=OneHotDegree(maxdegree, cat=False))
      num_nodes = max_num_nodes = 0
      for d in dataset1:
          num_nodes += d.num_nodes
          max_num_nodes = max(d.num_nodes, max_num_nodes)
      num_nodes = min(int(num_nodes / len(dataset1) * 1.5), max_num_nodes)
      indices = []
      num_nodes=200
      for i, d in enumerate(dataset1):
          if d.num_nodes <= num_nodes:
              indices.append(i)
      dataset1 = dataset1[torch.tensor(indices)]
  else:
      dataset1 = TUDataset(f"data/TUDataset", data)

  """##Error"""

  energies=[]
  for i in tqdm(range(len(dataset1))):
    cora=dataset1[i]
    adj=to_dense_adj(cora.edge_index)[0]
    X = cora.x
    adj=adj.to('cpu')
    L_o=get_laplacian(adj)
    try:
      adj, X, C=coarsening(adj, X, c_param[0], c_param[1], c_param[2], c_param[3], ratio, return_c=True)
    except:
      continue
    adj=adj.to('cpu')
    try:
        L = get_laplacian(adj)
        P = np.linalg.pinv(C)
        P = torch.from_numpy(P)
        L_lift = P.T@L@P
    except:
        continue
    en=Drichlet_Energy(L_o, L_lift)
    if en<=100:
        energies.append(en)
  with open(f"errors/{data}_{ratio}.csv", 'w') as f:
      for i in range(len(energies)):
          f.write(f"{energies[i]}\n")

  return sum(energies)/len(energies)
      

if __name__ == '__main__':
    datasets=["MUTAG", "BZR", "COX2", "DHFR", "PTC_MR", "ENZYMES", "DD", "PROTEINS", "IMDB-BINARY", "IMDB-MULTI","REDDIT-BINARY", "REDDIT-MULTI-5K"]
    ratios=[0.1,0.2,0.3,0.5]
    results=[]

    for data in datasets:
        out=[]
        for ratio in ratios:
            print("----Running-----",data, '--', ratio)
            out.append(str(run(data, ratio)))
        with open(f"errors/errors.csv", 'a') as f:
            f.write(f"{data},{','.join(out)}\n")
        

