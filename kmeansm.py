import numpy as np
import pandas as pd
from enum import Enum
import logging
import random
import time
import scipy.io

def kmeansm(data, k, l, maxiterations, eps, r):
    start_time = time.time()
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
    end_time = time.time()
    logging.info("Time taken %s seconds", str(end_time - start_time))
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
    # Cardio Data set
    mat = scipy.io.loadmat('data/cardio.mat')
    X_car = mat['X']
    y=mat['y']
    X_car = pd.DataFrame(X_car)
    print(X_car.head(4))
    clusterlmat = kmeansm(X_car, 2, 176, 10, 0.05, 21)
    print(clusterlmat)
    X_car[X_car.shape[1]] = clusterlmat
    X_car.to_csv('data/cardio.data.kmeansm.result.csv', index=False, header=False)
    
    # Pima Data set
    datapima = pd.read_csv('data/pimaindiansdiabetes.csv')
    clusterlPim = kmeansm(datapima, 2, 268, 5, 0.2, 8)
    datapima[datapima.shape[1]] = clusterlPim
    print(clusterlPim)
    datapima.to_csv('data/pima.data.kmeansm.result.csv', index=False, header=False)
    
    # Wine Data Set
    win = scipy.io.loadmat('data/wine.mat')
    print(win)
    X_win = win['X']
    y_win=win['y']
    print(y_win)
    X_win = pd.DataFrame(X_win)
    print(X_win.head(4))
    clusterlwin = kmeansm(X_win, 2, 10, 10, 0.2, 13)
    print(clusterlwin)
    X_win[X_win.shape[1]] = clusterlwin
    X_win.to_csv('data/wine.data.kmeansm.result.csv', index=False, header=False)
    
    # Glass Data Set
    
    glass = scipy.io.loadmat('data/glass.mat')
    print(glass)
    X_gla = glass['X']
    y_gla=glass['y']
    print(y_gla)
    X_gla = pd.DataFrame(X_gla)
    print(X_gla.head(4))
    clusterlgla = kmeansm(X_gla, 2, 9, 10, 0.2, 9)
    print(clusterlgla)
    X_gla[X_gla.shape[1]] = clusterlgla
    X_gla.to_csv('data/glass.data.kmeansm.result.csv', index=False, header=False)
    
    # Breast Cancer Data sets
    
    cancer = scipy.io.loadmat('data/breastw.mat')
    print(cancer)
    X_can = cancer['X']
    y_can=cancer['y']
    print(y_gla)
    X_can = pd.DataFrame(X_can)
    print(X_can.head(4))
    clusterlcan = kmeansm(X_can, 2, 239, 10, 0.2, 9)
    print(clusterlcan)
    X_can[X_can.shape[1]] = clusterlcan
    X_can.to_csv('data/breastw.data.kmeansm.result.csv', index=False, header=False)
    
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
