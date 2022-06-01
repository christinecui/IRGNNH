import numpy as np

def zero2eps(x):
    x[x == 0] = 1
    return x

def normalize(affnty):
    col_sum = zero2eps(np.sum(affnty, axis=1)[:, np.newaxis])
    row_sum = zero2eps(np.sum(affnty, axis=0))
    out_affnty = affnty/col_sum
    in_affnty = np.transpose(affnty/row_sum)
    return in_affnty, out_affnty

def L21_norm(X):
    # 2-norm for column
    X_norm2 = np.linalg.norm(X, ord=2, axis=0)
    X_norm2_norm1 = np.linalg.norm(X_norm2, ord=1)
    return X_norm2_norm1

# Check in 2020-6-25(14:26)
def getTrue2(test_label: np.ndarray, train_label: np.ndarray):
    cateTestTrain = np.sign(np.matmul(test_label, train_label.T)) # 0 or 1
    cateTestTrain = cateTestTrain.astype('int16')
    return cateTestTrain

def affinity_tag_multi(tag1: np.ndarray, tag2: np.ndarray):
    aff = np.matmul(tag1, tag2.T)
    affinity_matrix = np.float32(aff)
    in_aff, out_aff = normalize(affinity_matrix)
    return in_aff, out_aff
