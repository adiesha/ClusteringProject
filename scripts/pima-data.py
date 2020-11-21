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


def pima_re():
    epss = [(45,0.2,0.9),(35,0.35, 2),(25,0.2,0.0005)]
    for t in epss:

        data = pd.read_csv('../data/pimaindiansdiabetes.csv', header=None)
        print(data.head(12))
        y = data[8].to_numpy()
        print(y)
        # clusterlmat = kmeansm(X_car, 2, 176, 10, 0.05, 21)
        clusterlmat = dbscan.dbscanp(data, 8, eps=t[2], minpts=t[0], factor=t[1], initialization=dbscan.Initialization.UNIFORM, plot=True)
        # print(clusterlmat[0][13])
        y_t = clusterlmat[0][9].to_numpy()
        # print(y_t)

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
        print(t[2])

        # X_car[X_car.shape[1]] = clusterlmat
        # X_car.to_csv('data/cardio.data.kmeansm.result.csv', index=False, header=False)


if __name__ == '__main__':
    pima_re()
