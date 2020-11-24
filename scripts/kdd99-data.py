import logging as logging
# from scipy.io import loadmat
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_score

# import assessments as assess
import dbscanann as dbann
import dbscanpp as dbscan

import mat73 as mttt


def kdd99_re():
    epss = [25, 30, 35, 38, 72]
    for e in epss:


        mat = mttt.loadmat('../data/http.mat')
        X_car = mat['X']
        y = pd.DataFrame(mat['y'])
        y = y[0].to_numpy()
        X_car = pd.DataFrame(X_car)
        print(X_car.head(12))
        print(y)
        # clusterlmat = kmeansm(X_car, 2, 176, 10, 0.05, 21)
        clusterlmat = dbann.dbscanann(X_car, 3, eps=e, minpts=4, factor=0.3,
                                     initialization=dbscan.Initialization.UNIFORM)
        # print(clusterlmat[0][13])
        y_t = clusterlmat[0][3].to_numpy()
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
        # print(assess.falsealarmrate(y, [0], y_t, 1))
        print(adjusted_rand_score(y, y_t))
        print(jaccard_score(y, y_t))
        print(e)

        # X_car[X_car.shape[1]] = clusterlmat
        # X_car.to_csv('data/cardio.data.kmeansm.result.csv', index=False, header=False)


if __name__ == '__main__':
    logging.basicConfig(level="INFO")
    kdd99_re()
