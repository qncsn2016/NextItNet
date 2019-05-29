import math
import time
import utils
import numpy as np

# list1 = ["nima", "daue", "zhangsan"]
# for index, item in enumerate(list1):
#  print index, item
# # dict={"nima":1,"daue333":3}

# print math.log(4,2)



# !/usr/bin/env python
# -*- coding:utf-8 -*-

'''
def procedure():
    time.sleep(2.5)

t0 = time.clock()
procedure()
t1 = time.clock()
print("time.clock(): ", t1 - t0)

t0 = time.time()
procedure()
t1 = time.time()
print("time.time()", t1 - t0)
a=[0,2,3]
print(a[0:-1])
print(a[1:])
'''


def calc1(preds, test_batch):
    batch_out = []
    for line in test_batch:
        batch_out.append(line[-1])

    recall5, recall20 = [], []
    mrr5, mrr20 = [], []
    ndcg5, ndcg20 = [], []

    rank_l = []
    batch_predict = []

    for batch, b_label in zip(preds, batch_out):
        batch_predict.append(np.argmax(batch))
        ranks = (batch[b_label] < batch).sum() + 1  # 比label对应的值大的有多少个
        rank_l.append(ranks)

        recall5.append(ranks <= 5)
        recall20.append(ranks <= 20)
        mrr5.append(1 / ranks if ranks <= 5 else 0.0)
        mrr20.append(1 / ranks if ranks <= 20 else 0.0)
        ndcg5.append(1 / math.log(ranks + 1, 2) if ranks <= 5 else 0.0)
        ndcg20.append(1 / math.log(ranks + 1, 2) if ranks <= 20 else 0.0)
    # 返回的是一个batch的
    return rank_l, batch_predict, recall5, recall20, mrr5, mrr20, ndcg5, ndcg20


def calc2(preds, test_batch):
    _formrr5, _forhit5, _forndcg5, _formrr20, _forhit20, _forndcg20 = [], [], [], [], [], []
    rank_l, batch_predict = [], []
    for bi in range(len(preds)):
        pred_words_5 = utils.sample_top_k(preds[bi], top_k=5)
        pred_words_20 = utils.sample_top_k(preds[bi], top_k=20)

        predictmap_5 = {ch: i for i, ch in enumerate(pred_words_5)}
        pred_words_20 = {ch: i for i, ch in enumerate(pred_words_20)}

        true_word = test_batch[bi][-1]
        batch_predict.append(np.argmax(preds[bi]))

        ranks = (preds[bi][true_word] < preds[bi]).sum() + 1
        rank_l.append(ranks)

        rank_5 = predictmap_5.get(true_word)
        rank_20 = pred_words_20.get(true_word)
        if rank_5 == None:
            _formrr5.append(0.0)
            _forhit5.append(0.0)
            _forndcg5.append(0.0)
        else:
            MRR_5 = 1.0 / (rank_5 + 1)
            Rec_5 = 1.0
            ndcg_5 = 1.0 / math.log(rank_5 + 2, 2)
            _formrr5.append(MRR_5)
            _forhit5.append(Rec_5)
            _forndcg5.append(ndcg_5)
        if rank_20 == None:
            _formrr20.append(0.0)
            _forhit20.append(0.0)
            _forndcg20.append(0.0)
        else:
            MRR_20 = 1.0 / (rank_20 + 1)
            Rec_20 = 1.0
            ndcg_20 = 1.0 / math.log(rank_20 + 2, 2)
            _formrr20.append(MRR_20)
            _forhit20.append(Rec_20)
            _forndcg20.append(ndcg_20)
    return rank_l, batch_predict, _forhit5, _forhit20, _formrr5, _formrr20, _forndcg5, _forndcg20


if __name__ == '__main__':
    preds = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0.5, 0.4, 0.6, 0.3, 0.2, 0.1]])
    test_batch = np.array([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]])
    r1, p1, h51, h201, m51, m201, n51, n201 = calc1(preds, test_batch)
    r2, p2, h52, h202, m52, m202, n52, n202 = calc2(preds, test_batch)

    print("r1", r1)
    print("r2", r2)
    print("\np1", p1)
    print("p2", p2)
    print("\nh51", h51)
    print("h52", h52)
    print("\nh201", h201)
    print("h202", h202)
    print("\nm51", m51)
    print("m52", m52)
    print("\nm201", m201)
    print("m202", m202)
    print("\nn51", n51)
    print("n52", n52)
    print("\nn201", n201)
    print("n202", n202)
