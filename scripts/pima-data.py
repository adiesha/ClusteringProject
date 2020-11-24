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
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import csv as csv
import kmeansm as km
import modified.dbscanmpp as dbscanppm

def pima_re():
    epss = [(45, 0.2, 0.9), (35, 0.35, 2), (25, 0.2, 0.0005)]
    for t in epss:

        data = pd.read_csv('../data/pimaindiansdiabetes.csv', header=None)
        print(data.head(12))
        y = data[8].to_numpy()
        print(y)
        # clusterlmat = kmeansm(X_car, 2, 176, 10, 0.05, 21)
        clusterlmat = dbscan.dbscanp(data, 8, eps=t[2], minpts=t[0], factor=t[1],
                                     initialization=dbscan.Initialization.UNIFORM, plot=True)
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


def pima_pre_norm_re():
    epss = [(45, 0.02, 0.9), (35, 0.0001, 2), (25, 0.2, 0.00005)]
    for t in epss:

        data = pd.read_csv('../data/pimaindiansdiabetes.csv', header=None)
        print(data.head(12))
        y = data[8].to_numpy()
        print(y)
        scaler = MinMaxScaler()
        arr_scaled = scaler.fit_transform(data)
        data2 = pd.DataFrame(arr_scaled)
        # pca = PCA(n_components=7)
        # principalcomponents = pca.fit(data2.iloc[, 0:8])
        # principledf = pd.DataFrame(principalcomponents)

        # clusterlmat = kmeansm(X_car, 2, 176, 10, 0.05, 21)
        clusterlmat = dbscan.dbscanp(data2, 8, eps=t[2], minpts=t[0], factor=t[1],
                                     initialization=dbscan.Initialization.NONE, plot=False)
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


def pima_pre_norm_and_pca_re():
    epss = [(270, 0.5, 0.5), (270, 0.5, 0.4), (270, 0.5, 0.35), (270, 0.5, 0.3), (270, 0.5, 0.2), (270, 0.5, 0.1)]
    epss2 = [(280, 0.5, 0.5), (280, 0.5, 0.4), (280, 0.5, 0.35), (280, 0.5, 0.3), (280, 0.5, 0.2), (280, 0.5, 0.1)]
    epss3 = [(290, 0.5, 0.5), (290, 0.5, 0.4), (290, 0.5, 0.35), (290, 0.5, 0.3), (290, 0.5, 0.2), (290, 0.5, 0.1)]
    for t in epss3:

        data = pd.read_csv('../data/pimaindiansdiabetes.csv', header=None)
        # print(data.head(12))
        y = data[8].to_numpy()
        print(y)
        scaler = MinMaxScaler()
        arr_scaled = scaler.fit_transform(data)
        data2 = pd.DataFrame(arr_scaled)
        pca = PCA(n_components=7)
        principalcomponents = pca.fit_transform(data2.iloc[:, 0:8])
        principledf = pd.DataFrame(data=principalcomponents)
        # print(principledf.head(12))

        # clusterlmat = kmeansm(X_car, 2, 176, 10, 0.05, 21)
        clusterlmat = dbscan.dbscanp(principledf, 7, eps=t[2], minpts=t[0], factor=t[1],
                                     initialization=dbscan.Initialization.KCENTRE, plot=False)
        # print(clusterlmat[0][13])
        y_t = clusterlmat[0][7].to_numpy()
        # print(y_t)

        index = 0
        for i in y_t.copy():
            if i == -1:
                y_t[index] = 1
            else:
                y_t[index] = 0
            index += 1
        print("cluster labels:", y_t)

        print("eps: ", t[2])
        f_sc = f1_score(y, y_t, average='weighted')
        fa = assess.falsealarmrate(y, [0], y_t, 1)
        ard = adjusted_rand_score(y, y_t)
        js = jaccard_score(y, y_t)

        print(f_sc)
        print(fa)
        print(ard)
        print(js)
        print(t[2])

        rr = [t[0], t[2], t[1], f_sc, fa, ard, js, dbscan.Initialization.KCENTRE]
        with open('../data/pima_pca/dbscan.pima.pca.result.csv', 'a') as fd:
            writer = csv.writer(fd)
            writer.writerow(rr)
        # X_car[X_car.shape[1]] = clusterlmat
        # X_car.to_csv('data/cardio.data.kmeansm.result.csv', index=False, header=False)


def pima_pre_norm_and_pca_kmeans_re():
    epss = [(1, 250, 50, 0.0001), (1, 268, 50, 0.0001), (1, 275, 50, 0.0001)]
    epss2 = [(2, 250, 50, 0.0001), (2, 268, 50, 0.0001), (2, 275, 50, 0.0001)]
    epss3 = [(3, 250, 50, 0.0001), (3, 268, 50, 0.0001), (3, 275, 50, 0.0001)]
    for t in epss3:

        data = pd.read_csv('../data/pimaindiansdiabetes.csv', header=None)
        # print(data.head(12))
        y = data[8].to_numpy()
        print(y)
        scaler = MinMaxScaler()
        arr_scaled = scaler.fit_transform(data)
        data2 = pd.DataFrame(arr_scaled)
        pca = PCA(n_components=7)
        principalcomponents = pca.fit_transform(data2.iloc[:, 0:8])
        principledf = pd.DataFrame(data=principalcomponents)
        # print(principledf.head(12))

        # clusterlmat = kmeansm(X_car, 2, 176, 10, 0.05, 21)
        clusterlmat = km.kmeansm(principledf, t[0], t[1], t[2], t[3], 7)
        # print(clusterlmat[0][13])
        y_t = clusterlmat
        # print(y_t)

        index = 0
        for i in y_t.copy():
            if i == -1:
                y_t[index] = 1
            else:
                y_t[index] = 0
            index += 1
        print("cluster labels:", y_t)

        print("eps: ", t[2])
        f_sc = f1_score(y, y_t, average='weighted')
        fa = assess.falsealarmrate(y, [0], y_t, 1)
        ard = adjusted_rand_score(y, y_t)
        js = jaccard_score(y, y_t)

        print(f_sc)
        print(fa)
        print(ard)
        print(js)
        print(t[2])

        rr = [t[0], t[2], t[1], f_sc, fa, ard, js, "KMEANS--"]
        with open('../data/pima_pca/kmeans.pima.pca.result.csv', 'a') as fd:
            writer = csv.writer(fd)
            writer.writerow(rr)
        # X_car[X_car.shape[1]] = clusterlmat
        # X_car.to_csv('data/cardio.data.kmeansm.result.csv', index=False, header=False)


def pima_pre_norm_and_pca_re_dbm():
    epss = [(260, 0.5, 0.5), (260, 0.5, 0.4), (260, 0.5, 0.35), (260, 0.5, 0.3), (260, 0.5, 0.2), (260, 0.5, 0.1)]
    epss2 = [(268, 0.5, 0.5), (268, 0.5, 0.4), (268, 0.5, 0.35), (268, 0.5, 0.3), (268, 0.5, 0.2), (268, 0.5, 0.1)]
    epss3 = [(280, 0.5, 0.5), (280, 0.5, 0.4), (280, 0.5, 0.35), (280, 0.5, 0.3), (280, 0.5, 0.2), (280, 0.5, 0.1)]
    for t in epss:

        data = pd.read_csv('../data/pimaindiansdiabetes.csv', header=None)
        # print(data.head(12))
        y = data[8].to_numpy()
        print(y)
        scaler = MinMaxScaler()
        arr_scaled = scaler.fit_transform(data)
        data2 = pd.DataFrame(arr_scaled)
        pca = PCA(n_components=7)
        principalcomponents = pca.fit_transform(data2.iloc[:, 0:8])
        principledf = pd.DataFrame(data=principalcomponents)
        # print(principledf.head(12))

        # clusterlmat = kmeansm(X_car, 2, 176, 10, 0.05, 21)
        clusterlmat = dbscanppm.dbscanmp(principledf, 7, eps=t[2], minpts=t[0], factor=t[1], threshold=0.06,
                                     initialization=dbscan.Initialization.NONE, plot=False)
        # print(clusterlmat[0][13])
        y_t = clusterlmat[0][7].to_numpy()
        # print(y_t)

        index = 0
        for i in y_t.copy():
            if i == -1:
                y_t[index] = 1
            else:
                y_t[index] = 0
            index += 1
        print("cluster labels:", y_t)

        print("eps: ", t[2])
        f_sc = f1_score(y, y_t, average='weighted')
        fa = assess.falsealarmrate(y, [0], y_t, 1)
        ard = adjusted_rand_score(y, y_t)
        js = jaccard_score(y, y_t)

        print(f_sc)
        print(fa)
        print(ard)
        print(js)
        print(t[2])

        rr = [t[0], t[2], t[1], f_sc, fa, ard, js, dbscan.Initialization.NONE]
        with open('../data/pima_pca/dbscanm.pima.pca.result.csv', 'a') as fd:
            writer = csv.writer(fd)
            writer.writerow(rr)
        # X_car[X_car.shape[1]] = clusterlmat
        # X_car.to_csv('data/cardio.data.kmeansm.result.csv', index=False, header=False)



if __name__ == '__main__':
    # pima_re(
    logging.basicConfig(level="INFO")
    # pima_pre_norm_and_pca_kmeans_re()
    pima_pre_norm_and_pca_re_dbm()