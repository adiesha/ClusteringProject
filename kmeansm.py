import numpy as np
import pandas as pd
from enum import Enum
import logging
import random
import time



def kmeansm(data, k, l, maxiterations, eps, r):
    n = data.shape[0]
    centers = random.sample(np.arange(0,data.shape[0]), k)
    centerV = []
    for i in centers:
        centerV.append(np.array(data.iloc[i, 0:r]))
    logging.info(centers)
    logging.info(centerV)
    i = 1
    dx = np.full(n, np.inf)
    cx = np.full(n, 0, dtype=np.int32)  # cluster center of each point
    updateInitialDistances(data, dx, centerV, cx, r)
    logging.info(dx)
    logging.info(cx)
    while i < maxiterations:
        updateInitialDistances(data, dx, centerV, cx, r)
        ascargsort = np.argsort(dx)
        li = ascargsort[::-1][:l]
        logging.info(li)
        cx[li] = -1
        for j in range(0,k):
            output = [idx for idx, element in enumerate(cx) if element == j]
            mean = np.full(r, 0.0)
            count = 0
            for y in output:
                mean = mean + np.array(data.iloc[y, 0:r])
                count += 1
            centerV[j] = mean/count
        i += 1
    logging.info(dx)
    logging.info(cx)

def updateInitialDistances(data, dx, centers, cx, r):
    for i in data.index:
        basetuple = np.array(data.iloc[i, 0:r])
        index = 0
        for j in centers:
            tempTuple = j
            tempdistance = np.linalg.norm(basetuple - tempTuple)
            if tempdistance < dx[i]:
                dx[i] = tempdistance
                cx[i] = index
            index += 1
    return dx, cx

def main():
    print("hello")
    data = pd.read_csv('data/iris.data', header=None)
    kmeansm(data, 6, 10, 20, 0.1, 4)
    pass



if __name__ == "__main__":
    logging.basicConfig(level='INFO')
    a  = np.arange(10, 1, -2) 
    a[np.argsort(a)] = 1
    print(a)
    print(np.full(4, 0.0))
    # b = np.argsort(a)
    # print(b[::-1][:5])
    main()
    pass