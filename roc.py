import numpy as np
from sklearn.metrics import roc_auc_score

def my_auc_formula(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    # 分开正负样本的分数
    pos_scores = y_score[y_true == 1]
    neg_scores = y_score[y_true == 0]

    n_pos = len(pos_scores)
    n_neg = len(neg_scores)

    if n_pos == 0 or n_neg == 0:
        return 0.0

    # 公式核心：统计满足 pos > neg 的对数
    cnt = 0.0
    for p in pos_scores:
        for n in neg_scores:
            if p > n:
                cnt += 1.0
            elif p == n:
                cnt += 0.5

    auc = cnt / (n_pos * n_neg)
    return auc


# 测试（带重复分数 0.9）
y_true = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 0])
y_score = np.array([0.1, 0.3, 0.2, 0.6, 0.75, 0.5, 0.25, 0.9, 0.2, 0.9])

auc_f = my_auc_formula(y_true, y_score)
auc_sk = roc_auc_score(y_true, y_score)

print("公式法 AUC =", auc_f)
print("sklearn AUC =", auc_sk)