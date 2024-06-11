import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from .utils import log_round_wise
def run_fedavg(clients, server, COMMUNICATION_ROUNDS, local_epoch, samp=None, frac=1.0):
    for client in clients:
        client.download_from_server(server)

    if samp is None:
        sampling_fn = server.randomSample_clients
        frac = 1.0
    logs=pd.DataFrame(columns=[client.name for client in clients])
    ti=pd.DataFrame(columns=['time'])
    for c_round in range(1, COMMUNICATION_ROUNDS + 1):
        if (c_round) % 50 == 0:
            print(f"  > round {c_round}")

        if c_round == 1:
            selected_clients = clients
        else:
            selected_clients = sampling_fn(clients, frac)
        start_time=time.time()
        for client in selected_clients:
            # only get weights of graphconv layers
            client.local_train(local_epoch)
        

        server.aggregate_weights(selected_clients)
        for client in selected_clients:
            client.download_from_server(server)
        end_time=time.time()
        ti.loc[len(ti)]=end_time-start_time

        logs=log_round_wise(log=logs, client=clients)
        
    frame = pd.DataFrame()
    for client in clients:
        loss, acc = client.evaluate()
        frame.loc[client.name, 'test_acc'] = acc

    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: yellow' if v else '' for v in is_max]

    fs = frame.style.apply(highlight_max).data
    print(fs)
    return frame, logs, ti

