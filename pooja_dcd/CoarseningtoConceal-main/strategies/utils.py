
def log_round_wise(log,client):
    res=[]
    for cl in client:
        res.append(cl.evaluate()[1]) 
    # log = log.append(pd.Series(res, index=log.columns), ignore_index=True)
    log.loc[len(log)]=res
    return log