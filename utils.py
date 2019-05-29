import numpy as np
import math

def sample_top(a=[], top_k=10):
    idx = np.argsort(a)[::-1]
    idx = idx[:top_k]
    probs = a[idx]
    probs = probs / np.sum(probs)
    choice = np.random.choice(idx, p=probs)
    return choice

# fajie
def sample_top_k(a=[], top_k=10):
    idx = np.argsort(a)[::-1]
    idx = idx[:top_k]
    # probs = a[idx]
    # probs = probs / np.sum(probs)
    # choice = np.random.choice(idx, p=probs)
    return idx

print("print in utils", sample_top_k(np.array([0.02,0.01,0.01,0.16,0.8]),3))

def cau_recall_mrr_org(preds, labels):
    recall5, recall20 = [], []
    mrr5, mrr20 = [], []
    ndcg5, ndcg20 = [], []

    rank_l = []
    batch_predict=[]

    for batch, b_label in zip(preds, labels):
        batch_predict.append(np.argmax(batch))
        ranks = (batch[b_label] < batch).sum() + 1     # 比label对应的值大的有多少个
        rank_l.append(ranks)

        # if ranks == 1:
        #     f = open('prestosee.txt', "w")
        #     f.write("min pres="+str(min(batch))+'\n')
        #     f.write("max pres="+str(max(batch))+'\n')
        #     f.write("batch[label]="+str(batch[b_label])+'\n')
        #     for p in batch:
        #         f.write(str(p)+'\n')

        recall5.append(ranks <= 5)
        recall20.append(ranks <= 20)
        mrr5.append(1 / ranks if ranks <= 5 else 0.0)
        mrr20.append(1 / ranks if ranks <= 20 else 0.0)
        ndcg5.append(1/math.log(ranks + 1, 2) if ranks <= 5 else 0.0)
        ndcg20.append(1/math.log(ranks + 1, 2) if ranks <= 20 else 0.0)
    return rank_l, batch_predict, recall5, recall20, mrr5, mrr20, ndcg5, ndcg20
