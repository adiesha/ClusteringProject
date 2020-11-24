import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from enum import Enum
import logging
import random
import networkx as nx
import matplotlib.pyplot as plt
import time
import scipy.io


def dbscanmp(data, k, eps, minpts, factor, initialization=None, threshold=0.1, plot=False, plotPath='data/result.png',
             norm=None,
             leafsize=10):
    start_time = time.time()

    querycount = 0
    # -1: Noise 0:Undefined >0 : cluster number
    c = 0
    # k is the value that shows us which first k number of columns contain the attribute data
    # Create a new column Add undefined label
    labelcolumn = len(data.columns)
    data[labelcolumn] = 0

    # Create the KDTree for the algorithm
    neighbourhoodtree = KDTree(data.iloc[:, 0:k].values, leafsize=leafsize)

    index_array = list(np.arange(0, data.shape[0], 1))
    sample = index_array
    if initialization == Initialization.NONE or factor == 1:
        sample = index_array
    elif initialization == Initialization.UNIFORM:
        sample = random.sample(index_array, int(factor * data.shape[0]))
        sample.sort()
    elif initialization == Initialization.KCENTRE:
        sample = kgreedyinitialization(data, k, int(factor * data.shape[0]), norm)
    core_points = []
    G = nx.Graph()
    G.add_nodes_from(index_array)
    for i in sample:

        neighbourhood = neighbourhoodtree.query_ball_point(data.iloc[i, 0:k], r=eps)
        querycount += 1
        neighbourhood.remove(i)
        # print(len(neighbourhood))
        if len(neighbourhood) >= minpts:
            core_points.append(i)

        logging.info(neighbourhood)
        # remove the i from neighbourhood list to create the seedset

        seedset = neighbourhood
        j = 0
        if len(neighbourhood) >= minpts:
            while j < len(seedset):
                q = seedset[j]
                G.add_edge(i, q)
                j = j + 1

    connected_components = nx.connected_components(G)
    # logging.info("Number of clusters %d", connected_components)
    thresholdnodecount = threshold * data.shape[0]
    for component in connected_components:
        size = len(component)

        if size > 1 and size > thresholdnodecount:
            c += 1
        for node in component:
            if size == 1 or size <= thresholdnodecount:
                logging.info("noise point found, Index: %d threshold expression: %s", node,
                             str(size <= thresholdnodecount))
                data._set_value(node, labelcolumn, -1)
            if size > 1 and size > thresholdnodecount:
                data._set_value(node, labelcolumn, c)

    logging.info("Query Count: %d", querycount)
    endtime = time.time()
    print("--- %s seconds ---" % (endtime - start_time))
    if plot:
        nx.draw(G)
        plt.savefig(plotPath)
        plt.show()

    return data, endtime - start_time, querycount


def kgreedyinitialization(data, k, m, norm=None):
    n = data.shape[0]
    distance = np.full(shape=(n), fill_value=np.inf, dtype=float)
    S = set()
    slicedData = data.iloc[:, 0:k]
    # slicedData[slicedData.shape[1]] = slicedData.index
    # indexCol = slicedData.shape[1] - 1
    data_numpy_a = slicedData.to_numpy()
    for p in range(0, m):
        index_max = np.argmax(distance)
        S.add(index_max)
        baseTuple = np.array(slicedData.iloc[index_max, 0:k])
        logging.info("p: %d", p)
        # slicedData[distCol] = slicedData.apply(lambda row: returnMin(baseTuple, np.array(row[0:k]), distance, row[indexCol]), axis=1)
        # np.apply_along_axis(returnMin, 1, data_numpy_a, baseTuple, distance, k)
        distance = vec_returnMin(data_numpy_a, baseTuple, distance, k)
    return S


def returnMin(row, basetuple, dista, k):
    tuple = np.array(row[0:k])
    index = row[-1]
    tempdistance = np.linalg.norm(basetuple - tuple, ord=None)
    temp = min(dista[index], tempdistance)
    dista[index] = temp
    return temp


def vec_returnMin(row, baseTuple, dista, k):
    tuple = np.array(row[:, 0:k])
    difftuple = tuple - baseTuple
    squaretuple = np.square(difftuple)
    distatuple = np.sqrt(squaretuple)
    distatuplerow = np.sum(squaretuple, axis=1)
    difftuple = np.fmin(distatuplerow, dista)
    return difftuple


def main():
    ## Pima Data Set
    datapima = pd.read_csv("data/pimaindiansdiabetes.csv")
    col = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    datapima.columns = col
    X = datapima.iloc[:, :-1]
    y = datapima.iloc[:, -1]
    resultpima = dbscanmp(X, 4, 0.485, 6, 0.5, initialization=Initialization.KCENTRE, plot=True)
    resultpima[0].to_csv('data/pima.data.dbscan.result.csv', index=False, header=False)
    print("Time: ", resultpima[1])
    print("query count", resultpima[2])
    aapima = kgreedyinitialization(X, 4, 10)
    print(aapima)

    ## Cardio Data Set
    mat = scipy.io.loadmat('data/cardio.mat')
    X_car = mat['X']
    y_car = mat['y']
    X_car = pd.DataFrame(X_car)
    y_car = pd.DataFrame(y_car)
    resultmat = dbscanp(X_car, 4, 0.485, 6, 0.5, initialization=Initialization.KCENTRE, plot=True)
    resultmat[0].to_csv('data/cardio.data.dbscan.result.csv', index=False, header=False)
    print("Time: ", resultmat[1])
    print("query count", resultmat[2])
    aamat = kgreedyinitialization(X_car, 4, 10)
    print(aamat)

    ## Wine Data Set
    win = scipy.io.loadmat('data/wine.mat')
    print(win)
    X_win = win['X']
    y_win = win['y']
    X_win = pd.DataFrame(X_win)
    y_win = pd.DataFrame(y_win)
    resultwin = dbscanp(X_win, 4, 0.485, 6, 0.5, initialization=Initialization.KCENTRE, plot=True)
    resultwin[0].to_csv('data/wine.data.dbscan.result.csv', index=False, header=False)
    print("Time: ", resultwin[1])
    print("query count", resultwin[2])
    aawin = kgreedyinitialization(X_win, 4, 10)
    print(aawin)

    ## Glass Data Set
    glass = scipy.io.loadmat('data/glass.mat')
    print(glass)
    X_gla = glass['X']
    y_gla = glass['y']
    X_gla = pd.DataFrame(X_gla)
    y_gla = pd.DataFrame(y_gla)
    resultgla = dbscanp(X_gla, 4, 0.485, 6, 0.5, initialization=Initialization.KCENTRE, plot=True)
    resultgla[0].to_csv('data/glass.data.dbscan.result.csv', index=False, header=False)
    print("Time: ", resultgla[1])
    print("query count", resultgla[2])
    aagla = kgreedyinitialization(X_gla, 4, 10)
    print(aagla)

    ## Breast Cancer Data sets
    cancer = scipy.io.loadmat('data/breastw.mat')
    print(cancer)
    X_can = cancer['X']
    y_can = cancer['y']
    X_can = pd.DataFrame(X_can)
    y_can = pd.DataFrame(y_can)
    resultcan = dbscanp(X_can, 4, 0.485, 6, 0.5, initialization=Initialization.KCENTRE, plot=True)
    resultcan[0].to_csv('data/cancer.data.dbscan.result.csv', index=False, header=False)
    print("Time: ", resultcan[1])
    print("query count", resultcan[2])
    aacan = kgreedyinitialization(X_gla, 4, 10)
    print(aacan)


class Initialization(Enum):
    NONE = 1
    UNIFORM = 2
    KCENTRE = 3


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    logging.info("Start")
    main()
