import functools

import numpy as np


class Model(object):
    def __init__(self, name, SDS_acc, real_acc):
        self.name = name
        self.SDS_acc = SDS_acc
        self.real_acc = real_acc


# real_acc = df[0][0]
# print(real_acc)
#
# SDS_acc = acc_list[0][1]
# print(SDS_acc)

def DCG(scores):
    # np.power(2, scores) - 1
    return np.sum(
        np.divide(scores, np.log2(np.arange(scores.shape[0], dtype=np.float32) + 2))
        , dtype=np.float32)


def NDCG(model_list, k, sample_size):
    ndcg = []
    for i in range(sample_size):
        models_rank = sorted(model_list, key=functools.cmp_to_key(
            lambda x, y: y.SDS_acc - x.SDS_acc))

        ranks_ori = np.array(
            [models_rank[i].real_acc for i in range(len(model_list))])  # rels = models_rank[i].real_acc
        ranks_real_ori = -np.sort(-ranks_ori)

        ranks_index = np.arange(k)
        ranks_real_index = np.argsort(-ranks_ori)[0:k]

        ranks = ranks_ori[0:k]
        ranks_real = ranks_real_ori[0:k]

        if k > 1:
            tmp_index = np.unique(np.concatenate((ranks_index, ranks_real_index), axis=0))
            tmp = -np.sort(-ranks_ori[tmp_index])
            index = []
            for x in ranks:
                index.append(np.where(tmp == x)[0][0])
            tmp = np.divide((tmp - np.min(tmp)), np.max(tmp) - np.min(tmp))  # 归一化放大
            # ranks = np.divide( (ranks - np.min(ranks)), np.max(ranks)-np.min(ranks)) # 归一化放大
            # ranks_real = np.divide( (ranks_real - np.min(ranks_real)), np.max(ranks_real)-np.min(ranks_real)) # 归一化放大
            for i in range(len(ranks)):
                ranks_real[i] = tmp[i]
                ranks[i] = tmp[index[i]]
        dcg = DCG(ranks)
        idcg = DCG(ranks_real)
        if dcg == 0.0:
            return 0.0
        ndcg.append(dcg / idcg)

    return np.array(ndcg)


def MRR(model_list, sample_size):
    mrr = []
    for i in range(sample_size):
        models_rank = sorted(model_list, key=functools.cmp_to_key(
            lambda x, y: y.SDS_acc - x.SDS_acc))

        ranks = np.array([models_rank[i].real_acc for i in range(len(model_list))])  # rels = models_rank[i].real_acc
        ranks_real = -np.sort(-ranks)
        rr = np.argwhere(ranks_real[0] == ranks)[0][0] + 1
        mrr.append(1 / rr)
    return mrr


def MAP(model_list, k, sample_size):
    """Compute the average precision (AP) of a list of ranked items
    """
    map = []
    for i in range(sample_size):
        models_rank = sorted(model_list, key=functools.cmp_to_key(
            lambda x, y: y.SDS_acc - x.SDS_acc))

        ranks = np.array([models_rank[i].real_acc for i in range(len(model_list))])  # rels = models_rank[i].real_acc
        ranks_real = -np.sort(-ranks)
        ranks = ranks[0:k]
        ranks_real = ranks_real[0:k]

        hits = 0
        sum_precs = 0

        for n in range(len(ranks)):
            if ranks[n] in ranks_real:
                hits += 1
                sum_precs += hits / (n + 1.0)
        if hits > 0:
            map.append(sum_precs / len(ranks_real))
        else:
            map.append(0)
    return map
