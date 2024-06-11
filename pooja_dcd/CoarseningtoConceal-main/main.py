import os
import argparse
import random
import copy

import torch
from pathlib import Path

from utilities import *
from strategies.selftrain import run_selftrain_GC
from strategies.fedavg import run_fedavg
from strategies.fedprox import run_fedprox
from strategies.GFCL import run_gcfl
from strategies.GFCLPlus import run_gcflplus
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore", category=UserWarning)



def process_selftrain(clients, server, local_epoch):
    print("Self-training ...")
    df = pd.DataFrame()
    allAccs = run_selftrain_GC(clients, server, local_epoch)
    for k, v in allAccs.items():
        df.loc[k, [f'train_acc', f'val_acc', f'test_acc']] = v
    print(df)
    if args.repeat is None:
        outfile = os.path.join(outpath, f'accuracy_selftrain_GC{suffix}.csv')
    else:
        outfile = os.path.join(outpath, "repeats", f'{args.repeat}_accuracy_selftrain_GC{suffix}.csv')
    df.to_csv(outfile)
    print(f"Wrote to file: {outfile}")


def process_fedavg(clients, server):
    print("\nDone setting up FedAvg devices.")

    print("Running FedAvg ...")
    frame, logs, ti = run_fedavg(clients, server, args.num_rounds, args.local_epoch, samp=None)
    if args.repeat is None:
        outfile = os.path.join(outpath, f'accuracy_fedavg_GC{suffix}.csv')
        outfile_r = os.path.join(outpath, f'accuracy_fedavg_GC{suffix}_r.csv')
        outfile_t = os.path.join(outpath, f'accutacy_fedavg_GC{suffix}_t.csv')


    else:
        outfile = os.path.join(outpath, "repeats", f'{args.repeat}_accuracy_fedavg_GC{suffix}.csv')
        outfile_r = os.path.join(outpath, "repeats", f'{args.repeat}_accuracy_fedavg_GC{suffix}_r.csv')
        outfile_t = os.path.join(outpath, "repeats", f'{args.repeat}_accuracy_fedavg_GC{suffix}_t.csv')

    frame.to_csv(outfile)
    logs.to_csv(outfile_r)
    ti.to_csv(outfile_t)
    print(f"Wrote to file: {outfile}")


def process_fedprox(clients, server, mu):
    print("\nDone setting up FedProx devices.")

    print("Running FedProx ...")
    frame,logs, ti = run_fedprox(clients, server, args.num_rounds, args.local_epoch, mu, samp=None)
    if args.repeat is None:
        outfile = os.path.join(outpath, f'accuracy_fedprox_mu{mu}_GC{suffix}.csv')
        outfile_r = os.path.join(outpath, f'accuracy_fedprox_mu{mu}_GC{suffix}_r.csv')
        outfile_t = os.path.join(outpath, f'accuracy_fedprox_mu{mu}_GC{suffix}_t.csv')


    else:
        outfile = os.path.join(outpath, "repeats", f'{args.repeat}_accuracy_fedprox_mu{mu}_GC{suffix}.csv')
        outfile_r = os.path.join(outpath, "repeats", f'{args.repeat}_accuracy_fedprox_mu{mu}_GC{suffix}_r.csv')
        outfile_t = os.path.join(outpath, "repeats", f'{args.repeat}_accuracy_fedprox_mu{mu}_GC{suffix}_t.csv')


    frame.to_csv(outfile)
    logs.to_csv(outfile_r)
    ti.to_csv(outfile_t)
    print(f"Wrote to file: {outfile}")


def process_gcfl(clients, server):
    print("\nDone setting up GCFL devices.")
    print("Running GCFL ...")

    if args.repeat is None:
        outfile = os.path.join(outpath, f'accuracy_gcfl_GC{suffix}.csv')
        outfile_r = os.path.join(outpath, f'accuracy_gcfl_GC{suffix}_r.csv')
        outfile_t = os.path.join(outpath, f'accuracy_gcfl_GC{suffix}_t.csv')

    else:
        outfile = os.path.join(outpath, "repeats", f'{args.repeat}_accuracy_gcfl_GC{suffix}.csv')
        outfile_r = os.path.join(outpath, "repeats", f'{args.repeat}_accuracy_gcfl_GC{suffix}_r.csv')
        outfile_t = os.path.join(outpath, "repeats", f'{args.repeat}_accuracy_gcfl_GC{suffix}_t.csv')

    frame, logs,ti = run_gcfl(clients, server, args.num_rounds, args.local_epoch, EPS_1, EPS_2)
    frame.to_csv(outfile)
    logs.to_csv(outfile_r)
    ti.to_csv(outfile_t)
    print(f"Wrote to file: {outfile}")


def process_gcflplus(clients, server):
    print("\nDone setting up GCFL devices.")
    print("Running GCFL plus ...")

    if args.repeat is None:
        outfile = os.path.join(outpath, f'accuracy_gcflplus_GC{suffix}.csv')
        outfile_r = os.path.join(outpath, f'accuracy_gcflplus_GC{suffix}_r.csv')
        outfile_t = os.path.join(outpath, f'accuracy_gcflplus_GC{suffix}_t.csv')

    else:
        outfile = os.path.join(outpath, "repeats", f'{args.repeat}_accuracy_gcflplus_GC{suffix}.csv')
        outfile_r = os.path.join(outpath, "repeats", f'{args.repeat}_accuracy_gcflplus_GC{suffix}_r.csv')
        outfile_t = os.path.join(outpath, "repeats", f'{args.repeat}_accuracy_gcflplus_GC{suffix}_t.csv')

    frame,logs,ti = run_gcflplus(clients, server, args.num_rounds, args.local_epoch, EPS_1, EPS_2, args.seq_length, args.standardize)
    frame.to_csv(outfile)
    logs.to_csv(outfile_r)
    ti.to_csv(outfile_t)

    print(f"Wrote to file: {outfile}")


import argparse
import random
import numpy as np
import torch
import os
from pathlib import Path
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.transforms import OneHotDegree, Compose
from collections import Counter
import pandas as pd
import copy

def main():
    parser = argparse.ArgumentParser(description='Consolidated argparse parameters from multiple files.')

    # Add all the arguments from the provided codes
    parser.add_argument('--device', type=str, default='cpu', help='CPU / GPU device.')
    parser.add_argument('--num_repeat', type=int, default=5, help='number of repeating rounds to simulate;')
    parser.add_argument('--num_rounds', type=int, default=200, help='number of rounds to simulate;')
    parser.add_argument('--local_epoch', type=int, default=1, help='number of local epochs;')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate for inner solver;')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--nlayer', type=int, default=3, help='Number of GINconv layers')
    parser.add_argument('--hidden', type=int, default=64, help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for node classification.')
    parser.add_argument('--seed', help='seed for randomness;', type=int, default=123)

    parser.add_argument('--datapath', type=str, default='./data', help='The input path of data.')
    parser.add_argument('--outbase', type=str, default='./outputs', help='The base path for outputting.')
    parser.add_argument('--repeat', help='index of repeating;', type=int, default=None)
    parser.add_argument('--data_group', help='specify the group of datasets', type=str, default='PROTEINS')
    parser.add_argument('--convert_x', help='whether to convert original node features to one-hot degree features', type=bool, default=False)
    parser.add_argument('--overlap', help='whether clients have overlapped data', type=bool, default=False)
    parser.add_argument('--standardize', help='whether to standardize the distance matrix', type=bool, default=False)
    parser.add_argument('--seq_length', help='the length of the gradient norm sequence', type=int, default=10)
    parser.add_argument('--epsilon1', help='the threshold epsilon1 for GCFL', type=float, default=0.01)
    parser.add_argument('--epsilon2', help='the threshold epsilon2 for GCFL', type=float, default=0.1)
    parser.add_argument('--cr', help='Coarsening', type=str, default='False')
    parser.add_argument('--cr_ratio', help='cr_ratio', type=float, default=1)
    parser.add_argument('--dp', help='DP', type=str, default='False')
    parser.add_argument('--priv_budget', help='priv_budget', type=float, default=0)
    parser.add_argument('--strategy', help='strategy', type=str, default='SDMC')
    parser.add_argument('--num_clients', help='number of clients', type=int, default=10)

    parser.add_argument('--gpu_id', type=int, default=0, help='GPU id')
    parser.add_argument('--dataset', type=str, default='DD', help='Dataset name')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--init', type=str, default='real', help='Initialization method')
    parser.add_argument('--lr_adj', type=float, default=0.01, help='Learning rate for adjacency')
    parser.add_argument('--lr_feat', type=float, default=0.01, help='Learning rate for features')
    parser.add_argument('--nconvs', type=int, default=3, help='Number of convolutional layers')
    parser.add_argument('--outer', type=int, default=1, help='Outer loop iterations')
    parser.add_argument('--inner', type=int, default=0, help='Inner loop iterations')
    parser.add_argument('--pooling', type=str, default='mean', help='Pooling method')
    parser.add_argument('--lr_model', type=float, default=0.005, help='Learning rate for the model')
    parser.add_argument('--stru_discrete', type=int, default=1, help='Structure discrete flag')
    parser.add_argument('--ipc', type=int, default=0, help='Number of condensed samples per class')
    parser.add_argument('--reduction_rate', type=float, default=0.1, help='Reduction rate')
    parser.add_argument('--save', type=int, default=0, help='Save flag for condensed graphs')
    parser.add_argument('--dis_metric', type=str, default='mse', help='Distance metric')
    parser.add_argument('--eval_init', type=int, default=1, help='Evaluate initialization flag')
    parser.add_argument('--bs_cond', type=int, default=256, help='Batch size for sampling graphs')
    parser.add_argument('--net_norm', type=str, default='none', help='Network normalization type')
    parser.add_argument('--beta', type=float, default=0.1, help='Coefficient for the regularization term')
    parser.add_argument('--debug', type=int, default=0, help='Debug flag')
    parser.add_argument('--nruns', type=int, default=10, help='Number of runs')
    parser.add_argument('--mlp', type=int, default=0, help='MLP flag')
    parser.add_argument('--num_blocks', type=int, default=1, help='Number of blocks')
    parser.add_argument('--num_bases', type=int, default=0, help='Number of bases')
    parser.add_argument('--filename', type=str, help='Filename')

    args = parser.parse_args()

    seed_dataSplit = 124

    # set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    EPS_1 = args.epsilon1
    EPS_2 = args.epsilon2

    outbase = os.path.join(args.outbase, f'seqLen{args.seq_length}')
    if args.cr == 'True':
        outbase = os.path.join(outbase, 'Coarsen', f'{args.cr_ratio}')
    elif args.dp == 'True':
        outbase = os.path.join(outbase, 'DP', f'{args.priv_budget}')
    else:
        outbase = os.path.join(outbase, 'Standard')

    if args.strategy == 'MDMC':  # Multi Data Multi Client
        if args.overlap and args.standardize:
            outpath = os.path.join(outbase, f"standardizedDTW/multiDS-overlap")
        elif args.overlap:
            outpath = os.path.join(outbase, f"multiDS-overlap")
        elif args.standardize:
            outpath = os.path.join(outbase, f"standardizedDTW/multiDS-nonOverlap")
        else:
            outpath = os.path.join(outbase, f"multiDS-nonOverlap")
        outpath = os.path.join(outpath, args.data_group, f'eps_{EPS_1}_{EPS_2}')
        Path(outpath).mkdir(parents=True, exist_ok=True)
        print(f"Output Path: {outpath}")

        if not args.convert_x:
            suffix = ""
            print("Preparing data (original features) ...")
        else:
            suffix = "_degrs"
            print("Preparing data (one-hot degree features) ...")

        if args.repeat is not None:
            Path(os.path.join(outpath, 'repeats')).mkdir(parents=True, exist_ok=True)

        splitedData, df_stats = prepareData_multiDS(args.datapath, args.data_group, args.batch_size, convert_x=args.convert_x, seed=seed_dataSplit, cr=args.cr, cr_ratio=args.cr_ratio)
        print("Done")
    elif args.strategy == 'SDMC':  # Single Data Multi Client
        if args.overlap and args.standardize:
            outpath = os.path.join(outbase, f"standardizedDTW/oneDS-overlap")
        elif args.overlap:
            outpath = os.path.join(outbase, f"oneDS-overlap")
        elif args.standardize:
            outpath = os.path.join(outbase, f"standardizedDTW/oneDS-nonOverlap")
        else:
            outpath = os.path.join(outbase, f"oneDS-nonOverlap")
        outpath = os.path.join(outpath, f'{args.data_group}-{args.num_clients}clients', f'eps_{EPS_1}_{EPS_2}')
        Path(outpath).mkdir(parents=True, exist_ok=True)
        print(f"Output Path: {outpath}")

        if not args.convert_x:
            suffix = ""
            print("Preparing data (original features) ...")
        else:
            suffix = "_degrs"
            print("Preparing data (one-hot degree features) ...")

        if args.repeat is not None:
            Path(os.path.join(outpath, 'repeats')).mkdir(parents=True, exist_ok=True)

        splitedData, df_stats = prepareData_oneDS(args.datapath, args.data_group, num_client=args.num_clients, batchSize=args.batch_size,
                                                  convert_x=args.convert_x, seed=seed_dataSplit, overlap=args.overlap, cr=args.cr, cr_ratio=args.cr_ratio)

    if args.repeat is None:
        outf = os.path.join(outpath, f'stats_trainData{suffix}.csv')
    else:
        outf = os.path.join(outpath, "repeats", f'{args.repeat}_stats_trainData{suffix}.csv')
    df_stats.to_csv(outf)
    print(f"Wrote to {outf}")

    init_clients, init_server, init_idx_clients = setup_devices(splitedData, args)
    print("\nDone setting up devices.")

    process_selftrain(clients=copy.deepcopy(init_clients), server=copy.deepcopy(init_server), local_epoch=100)
    process_fedavg(clients=copy.deepcopy(init_clients), server=copy.deepcopy(init_server))
    process_fedprox(clients=copy.deepcopy(init_clients), server=copy.deepcopy(init_server), mu=0.01)
    process_gcfl(clients=copy.deepcopy(init_clients), server=copy.deepcopy(init_server))
    process_gcflplus(clients=copy.deepcopy(init_clients), server=copy.deepcopy(init_server))

if __name__ == '__main__':
    main()
