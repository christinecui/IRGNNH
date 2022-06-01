import numpy as np
def mAP(cateTrainTest: np.ndarray, IX: np.ndarray, topk = None):

    m, n = cateTrainTest.shape
    cateTrainTest = np.ascontiguousarray(cateTrainTest, np.int32).reshape(m*n)
    IX = np.ascontiguousarray(IX, np.int32).reshape(m*n)

    if topk == None:
        topk = m
    mAPs = np.zeros(n, dtype=np.float64)
    idx = 0
    for i in range(n):
        x = 0.0
        p = 0.0
        for j in range(topk):
            idx = IX[i + n*j]
            if cateTrainTest[i + n*idx] == 1:
                x += 1
                p = p + x/(j * 1.0 + 1.0)

        if p == 0.0:
            mAPs[i] = 0
        else:
            mAPs[i] = p/x
    mAP = 0
    for i in range(n):
        mAP += mAPs[i]
    mAP = mAP / n
    return mAP

def topK(cateTrainTest, IX, topk=500):
    m, n = cateTrainTest.shape
    cateTrainTest = np.ascontiguousarray(cateTrainTest, np.int32).reshape(m * n)
    IX = np.ascontiguousarray(IX, np.int32).reshape(m * n)

    precs = np.zeros(n, dtype=np.float64)
    recs = np.zeros(n, dtype=np.float64)

    if topk == None:
        topk = m

    retrieved_rel = 0
    idx = 0
    real_rel = 0

    for i in range(n):
        retrieved_rel = 0
        for j in range(topk):
            idx = IX[i + n*j]
            retrieved_rel += cateTrainTest[i + n*idx]
        real_rel = 0
        for j in range(m):
            real_rel += cateTrainTest[i + n*j]
        
        precs[i] = retrieved_rel / (topk * 1.0)
        if real_rel != 0:        
            recs[i] = retrieved_rel / (real_rel * 1.0)
        else:
            recs[i] = 0

    return np.mean(precs), np.mean(recs)
