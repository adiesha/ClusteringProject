import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.cluster import DBSCAN
from sklearn.metrics import jaccard_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import f1_score
import logging as logging
import assessments as assess
import dbscanpp as dbscan


def wine_re():
    epss = [25, 30, 35, 38, 72]
    for e in epss:

        mat = loadmat('../data/wine.mat')
        X_car = mat['X']
        y = pd.DataFrame(mat['y'])
        y = y[0].to_numpy()
        X_car = pd.DataFrame(X_car)
        print(X_car.head(12))
        print(y)
        # clusterlmat = kmeansm(X_car, 2, 176, 10, 0.05, 21)
        clusterlmat = dbscan.dbscanp(X_car, 13, eps=e, minpts=4, factor=0.3,
                                     initialization=dbscan.Initialization.UNIFORM)
        # print(clusterlmat[0][13])
        y_t = clusterlmat[0][13].to_numpy()
        print(y_t)

        index = 0
        for i in y_t.copy():
            if i == -1:
                y_t[index] = 1
            else:
                y_t[index] = 0
            index += 1
        print(y_t)

        print(f1_score(y, y_t, average='weighted'))
        print(assess.falsealarmrate(y, [0], y_t, 1))
        print(adjusted_rand_score(y, y_t))
        print(jaccard_score(y, y_t))
        print(e)

        # X_car[X_car.shape[1]] = clusterlmat
        # X_car.to_csv('data/cardio.data.kmeansm.result.csv', index=False, header=False)


if __name__ == '__main__':
    wine_re()
