import numpy as np
import pandas as pd
from enum import Enum
import logging
import random
import time


def kmeansm(data, k, l, maxiterations, eps, r):
    n = data.shape[0]
    centers = random.sample(list(np.arange(0, data.shape[0])), k)
    centers.sort()
    centerV = []
    for i in centers:
        centerV.append(np.array(data.iloc[i, 0:r]))
    logging.debug(centers)
    logging.debug(centerV)
    i = 1
    dx = np.full(n, np.inf)
    cx = np.full(n, 0, dtype=np.int32)  # cluster center of each point
    updateInitialDistances(data, dx, centerV, cx, r)
    logging.debug(dx)
    logging.debug(cx)
    while i < maxiterations:
        logging.info("Iteration %d", i)
        updateInitialDistances(data, dx, centerV, cx, r)
        ascargsort = np.argsort(dx)
        li = ascargsort[::-1][:l]
        logging.debug(li)
        cx[li] = -1
        logging.debug(cx)

        error = 0
        for j in range(0, k):
            output = [idx for idx, element in enumerate(cx) if element == j]
            mean = np.full(r, 0.0)
            count = 0
            for y in output:
                mean = mean + np.array(data.iloc[y, 0:r])
                count += 1
            error = error + abs(np.linalg.norm(centerV[j] - mean / count))
            centerV[j] = mean / count
        if error < eps:
            logging.info("Error: %.7f", error)
            logging.info("algorithm stopped because error level was minimized, number of iterations %d", i)
            break
        i += 1
    logging.debug(dx)
    logging.debug(cx)
    return cx


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
    # print("hello")
    # data = pd.read_csv('data/iris.data', header=None)
    # clusterl = kmeansm(data, 3, 3, 100, 0.05, 4)
    # data[data.shape[1]] = clusterl
    # data.to_csv('data/iris.data.kmeansm.result.csv', index=False, header=False)

    data = pd.read_csv('data/shuttle.trn', header=None)
    clusterl = kmeansm(data, 10, 175, 20, 0.1, 9)
    data[data.shape[1]] = clusterl
    data.to_csv('data/shuttle.data.kmeansm.result.csv', index=False, header=False)

    pass


if __name__ == "__main__":
    logging.basicConfig(level='INFO')
    a = np.arange(10, 1, -2)
    a[np.argsort(a)] = 1
    print(a)
    print(np.full(4, 0.0))
    # b = np.argsort(a)
    # print(b[::-1][:5])
    main()
    pass
