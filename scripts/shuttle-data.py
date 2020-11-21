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
import dbscanann as dbann
import csv as csv
import kmeansm as km

def shuttle_re_db_ann():
    epss = [(4.5, 10, 0.1), (4.8, 10, 0.1), (5, 10, 0.1), (5.3, 10, 0.1), (5.5, 10, 0.1), (5.8, 10, 0.1), (6, 10, 0.1),
            (6.8, 10, 0.1), (7, 10, 0.1), (9, 10, 0.1), (10, 10, 0.1), (28, 10, 0.1), (28.5, 10, 0.1)]

    # epss = [(6.8, 10, 0.1)]
    for t in epss:
        data = pd.read_csv('../data/shuttle-unsupervised-trn.csv', header=None)
        # datafiltered = data[data[9] != 4]
        # datafiltered = pd.DataFrame(datafiltered)
        # print(datafiltered)
        y = data[9].to_numpy()
        print(y)
        # datafiltered[9].hist()
        # plt.title("shuttle data set histogram")
        # plt.show()
        # clusterlmat = kmeansm(X_car, 2, 176, 10, 0.05, 21)
        clusterlmat = dbann.dbscanann(data, 9, eps=t[0], minpts=t[1], factor=1,
                                      initialization=dbann.Initialization.NONE,
                                      plot=False)
        # print(clusterlmat[0][13])
        y_t = clusterlmat[0][10].to_numpy()
        identifiedNoisepoints = np.count_nonzero(y_t == -1)
        print("count of noise points", identifiedNoisepoints)
        # print(y_t)

        a = y != 4
        y = y[y != 4]
        y_t = y_t[a]

        index = 0
        for i in y_t.copy():
            if i == -1:
                y_t[index] = 1
            else:
                y_t[index] = 0
            index += 1
            # print(y_t)
        index = 0
        for i in y.copy():
            if i == 2 or i == 3 or i == 5 or i == 6 or i == 7:
                y[index] = 1
            else:
                y[index] = 0
            index += 1

        print("Actual number of outlier: ", np.count_nonzero(y == 1))
        #2644
        f1_scored = f1_score(y, y_t, average='weighted')
        falarm = assess.falsealarmrate(y, [0], y_t, 1)
        arand = adjusted_rand_score(y, y_t)
        jacc = jaccard_score(y, y_t)

        rr = [t[1], t[0], t[2], f1_scored, falarm, arand, jacc, identifiedNoisepoints]
        with open('../data/ann/dbscan.dbann.shuttle.result.csv', 'a') as fd:
            writer = csv.writer(fd)
            writer.writerow(rr)

        # X_car[X_car.shape[1]] = clusterlmat
        # X_car.to_csv('data/cardio.data.kmeansm.result.csv', index=False, header=False)


def shuttle_re_db_ann_uniform():
    epss = [(4.5, 10, 0.1), (4.8, 10, 0.1), (5, 10, 0.1), (5.3, 10, 0.1), (5.5, 10, 0.1), (5.8, 10, 0.1), (6, 10, 0.1),
            (6.8, 10, 0.1), (7, 10, 0.1), (9, 10, 0.1), (10, 10, 0.1), (28, 10, 0.1), (28.5, 10, 0.1)]

    # epss = [(6.8, 10, 0.1)]
    for t in epss:
        data = pd.read_csv('../data/shuttle-unsupervised-trn.csv', header=None)
        # datafiltered = data[data[9] != 4]
        # datafiltered = pd.DataFrame(datafiltered)
        # print(datafiltered)
        y = data[9].to_numpy()
        print(y)
        # datafiltered[9].hist()
        # plt.title("shuttle data set histogram")
        # plt.show()
        # clusterlmat = kmeansm(X_car, 2, 176, 10, 0.05, 21)
        clusterlmat = dbann.dbscanann(data, 9, eps=t[0], minpts=t[1], factor=t[2],
                                      initialization=dbann.Initialization.UNIFORM,
                                      plot=False)
        # print(clusterlmat[0][13])
        y_t = clusterlmat[0][10].to_numpy()
        identifiedNoisepoints = np.count_nonzero(y_t == -1)
        print("count of noise points", identifiedNoisepoints)
        # print(y_t)

        a = y != 4
        y = y[y != 4]
        y_t = y_t[a]

        index = 0
        for i in y_t.copy():
            if i == -1:
                y_t[index] = 1
            else:
                y_t[index] = 0
            index += 1
            # print(y_t)
        index = 0
        for i in y.copy():
            if i == 2 or i == 3 or i == 5 or i == 6 or i == 7:
                y[index] = 1
            else:
                y[index] = 0
            index += 1

        f1_scored = f1_score(y, y_t, average='weighted')
        falarm = assess.falsealarmrate(y, [0], y_t, 1)
        arand = adjusted_rand_score(y, y_t)
        jacc = jaccard_score(y, y_t)

        rr = [t[1], t[0], t[2], f1_scored, falarm, arand, jacc, identifiedNoisepoints]
        with open('../data/ann/dbscan.dbann.uniform.shuttle.result.csv', 'a') as fd:
            writer = csv.writer(fd)
            writer.writerow(rr)

        # X_car[X_car.shape[1]] = clusterlmat
        # X_car.to_csv('data/cardio.data.kmeansm.result.csv', index=False, header=False)



def shuttle_re_db_ann_kcenter():
    epss = [(4.5, 10, 0.1), (4.8, 10, 0.1), (5, 10, 0.1), (5.3, 10, 0.1), (5.5, 10, 0.1), (5.8, 10, 0.1), (6, 10, 0.1),
            (6.8, 10, 0.1), (7, 10, 0.1), (9, 10, 0.1), (10, 10, 0.1), (28, 10, 0.1), (28.5, 10, 0.1)]

    # epss = [(6.8, 10, 0.1)]
    for t in epss:
        data = pd.read_csv('../data/shuttle-unsupervised-trn.csv', header=None)
        # datafiltered = data[data[9] != 4]
        # datafiltered = pd.DataFrame(datafiltered)
        # print(datafiltered)
        y = data[9].to_numpy()
        print(y)
        # datafiltered[9].hist()
        # plt.title("shuttle data set histogram")
        # plt.show()
        # clusterlmat = kmeansm(X_car, 2, 176, 10, 0.05, 21)
        clusterlmat = dbann.dbscanann(data, 9, eps=t[0], minpts=t[1], factor=t[2],
                                      initialization=dbann.Initialization.KCENTRE,
                                      plot=False)
        # print(clusterlmat[0][13])
        y_t = clusterlmat[0][10].to_numpy()
        identifiedNoisepoints = np.count_nonzero(y_t == -1)
        print("count of noise points", identifiedNoisepoints)
        # print(y_t)

        a = y != 4
        y = y[y != 4]
        y_t = y_t[a]

        index = 0
        for i in y_t.copy():
            if i == -1:
                y_t[index] = 1
            else:
                y_t[index] = 0
            index += 1
            # print(y_t)


        index = 0
        for i in y.copy():
            if i == 2 or i == 3 or i == 5 or i == 6 or i == 7:
                y[index] = 1
            else:
                y[index] = 0
            index += 1
        print("Actual number of outlier: ", np.count_nonzero(y == 1))
        f1_scored = f1_score(y, y_t, average='weighted')
        falarm = assess.falsealarmrate(y, [0], y_t, 1)
        arand = adjusted_rand_score(y, y_t)
        jacc = jaccard_score(y, y_t)

        rr = [t[1], t[0], t[2], f1_scored, falarm, arand, jacc, identifiedNoisepoints]
        with open('../data/ann/dbscan.dbann.kcenter.shuttle.result.csv', 'a') as fd:
            writer = csv.writer(fd)
            writer.writerow(rr)

        # X_car[X_car.shape[1]] = clusterlmat
        # X_car.to_csv('data/cardio.data.kmeansm.result.csv', index=False, header=False)



def shuttle_re_kmeans():



    data = pd.read_csv('../data/shuttle-unsupervised-trn.csv', header=None)
    # datafiltered = data[data[9] != 4]
    # datafiltered = pd.DataFrame(datafiltered)
    # print(datafiltered)
    y = data[9].to_numpy()
    print(y)
    # datafiltered[9].hist()
    # plt.title("shuttle data set histogram")
    # plt.show()
    # clusterlmat = kmeansm(X_car, 2, 176, 10, 0.05, 21)
    clusterlmat = km.kmeansm(data, 1, 2644, 50, 0.01, 9)
    # print(clusterlmat[0][13])
    y_t = clusterlmat
    identifiedNoisepoints = np.count_nonzero(y_t == -1)
    print("count of noise points", identifiedNoisepoints)
    # print(y_t)

    a = y != 4
    y = y[y != 4]
    y_t = y_t[a]

    index = 0
    for i in y_t.copy():
        if i == -1:
            y_t[index] = 1
        else:
            y_t[index] = 0
        index += 1
        # print(y_t)


    index = 0
    for i in y.copy():
        if i == 2 or i == 3 or i == 5 or i == 6 or i == 7:
            y[index] = 1
        else:
            y[index] = 0
        index += 1
    print("Actual number of outlier: ", np.count_nonzero(y == 1))
    f1_scored = f1_score(y, y_t, average='weighted')
    falarm = assess.falsealarmrate(y, [0], y_t, 1)
    arand = adjusted_rand_score(y, y_t)
    jacc = jaccard_score(y, y_t)

    rr = [1, 2644, 50, f1_scored, falarm, arand, jacc, identifiedNoisepoints]
    with open('data/ann/kmeans.shuttle.result.csv', 'a') as fd:
        writer = csv.writer(fd)
        writer.writerow(rr)

    # X_car[X_car.shape[1]] = clusterlmat
    # X_car.to_csv('data/cardio.data.kmeansm.result.csv', index=False, header=False)


if __name__ == '__main__':
    logging.basicConfig(level="INFO")
    # shuttle_re_db_ann()
    # shuttle_re_db_ann_uniform()
    # shuttle_re_db_ann_kcenter()
    shuttle_re_kmeans()