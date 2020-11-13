import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import assessments as metrics
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import jaccard_score
import scipy.io
def main():
    
    ## Pima Data set
    datapima = pd.read_csv("data/pimaindiansdiabetes.csv")
    col = ['preg','plas','pres','skin','test','mass','pedi','age','class']

    datapima.columns = col
    X = datapima.iloc[:,:-1]
    y = datapima.iloc[:,-1]
    y.hist()
    plt.title(" Pima Indians diabetes Data Class Distribution")
    plt.show()
    clusteringpima = DBSCAN(eps=0.9, min_samples=6).fit(X)
    print(clusteringpima.labels_)
    print(type(clusteringpima.labels_))
    # print(type(clustering.labels_))
    for index, row in datapima.iterrows():
        print(str(row[8]) + ":" + str(clusteringpima.labels_[index]))

        print('false alarm rate: ', metrics.falsealarmrate(y, [1] , clusteringpima.labels_, -1))
    pima= clusteringpima.labels_
    pima = pima.copy()
    preds = [1 if i==-1 else 0 for i in pima]
    print(jaccard_score(y, preds))
    print(preds)
    
    ## Cardio Data Set
    # mat = scipy.io.loadmat('data/cardio.mat')
    # X_car = mat['X']
    # y_car=mat['y']
    # y_car= pd.DataFrame(y_car)
    # print(list(y_car[0]))
    # y_car.hist()
    # plt.title("Cardiotocography Data Class Distribution")
    # plt.show()
    # X_car = pd.DataFrame(X_car)
    # clusteringcar = DBSCAN(eps=1.5, min_samples=10).fit(X_car)
    # print(clusteringcar.labels_)
    # print('false alarm rate: ', metrics.falsealarmrate(y_car[0], [1], clusteringcar.labels_, -1))
    # car= clusteringcar.labels_
    # car = car.copy()
    # predcar = [1 if i==-1 else 0 for i in car]
    # print(jaccard_score(y_car[0], predcar))
    # print(predcar)
    
    ## Wine Data Set
    # win = scipy.io.loadmat('data/wine.mat')
    # print(win)
    # X_win = win['X']
    # y_win=win['y']
    # print(y_win)
    # X_win = pd.DataFrame(X_win)
    # y_win= pd.DataFrame(y_win)
    # y_win.hist()
    # plt.title("Wine Data Class Distribution")
    # plt.show()
    # clusteringwin = DBSCAN(eps=1.5, min_samples=5).fit(X_win)
    # print(clusteringwin.labels_)
    # # print('false alarm rate: ', metrics.falsealarmrate(data.iloc[:, 36].values, 'o', clustering.labels_, -1))
    # win= clusteringwin.labels_
    # win = win.copy()
    # predwin = [ 1 if i==-1 else 0 for i in win]
    # print(predwin)
    
    ## Glass Data Set
    glass = scipy.io.loadmat('data/glass.mat')
    print(glass)
    X_gla = glass['X']
    y_gla=glass['y']
    X_gla = pd.DataFrame(X_gla)
    y_gla= pd.DataFrame(y_gla)
    y_gla.hist()
    plt.title("Glass Data Class Distribution")
    plt.show()
    clusteringgla = DBSCAN(eps=0.9, min_samples=10, n_jobs=-1).fit(X_gla)
    print(clusteringgla.labels_)
    print('false alarm rate: ', metrics.falsealarmrate(y_gla[0], [1], clusteringgla.labels_, -1))
    gla= clusteringgla.labels_
    gla = gla.copy()
    predgla = [1 if i==-1 else 0 for i in gla.copy()]
    print(jaccard_score(y_gla[0], predgla))
    print(predgla)
   
    ## Breast Cancer Data sets
    # cancer = scipy.io.loadmat('data/breastw.mat')
    # print(cancer)
    # X_can = cancer['X']
    # y_can=cancer['y']
    # X_can= pd.DataFrame(X_can)
    # y_can= pd.DataFrame(y_can)
    # y_can.hist()
    # plt.title("Breast Cancer Wisconsin Original Data Class Distribution")
    # plt.show()
    # clusteringcan = DBSCAN(eps=5, min_samples=6).fit(X_can)
    # print(clusteringcan.labels_)
    # # print('false alarm rate: ', metrics.falsealarmrate(data.iloc[:, 36].values, 'o', clustering.labels_, -1))
    # can= clusteringcan.labels_
    # can = can.copy()
    # predcan = [1 if i==-1 else 0 for i in can.copy()]
    # print(predcan)
if __name__ == '__main__':
    main()
