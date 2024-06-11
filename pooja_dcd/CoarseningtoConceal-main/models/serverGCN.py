import torch
import torch.nn.functional as F
# from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.nn import global_add_pool

from torch_geometric.nn import GCNConv



class serverGCN(torch.nn.Module):
    def __init__(self, nlayer, nhid):
        super(serverGCN, self).__init__()
        self.graph_convs = torch.nn.ModuleList()
        # self.nn1 = torch.nn.Sequential(torch.nn.Linear(nhid, nhid), torch.nn.ReLU(),
        #                                torch.nn.Linear(nhid, nhid))
        # self.graph_convs.append(GINConv(self.nn1))
        self.graph_convs.append(GCNConv(in_channels=nhid, out_channels=nhid))

        for l in range(nlayer - 1):
            # self.nnk = torch.nn.Sequential(torch.nn.Linear(nhid, nhid), torch.nn.ReLU(),
            #                                torch.nn.Linear(nhid, nhid))
            # self.graph_convs.append(GINConv(self.nnk))
            self.graph_convs.append(GCNConv(in_channels=nhid, out_channels=nhid))

