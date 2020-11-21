import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.cluster import DBSCAN
from sklearn.metrics import jaccard_score

import assessments as metrics


def main():
    mat = loadmat('../data/cardio.mat')
    print(mat)
    X_car = mat['X']

    y_car = mat['y']
    y_car = pd.DataFrame(y_car)
    # print(list(y_car[0]))
    # y_car.hist()
    plt.title("Cardiotocography Data Class Distribution")
    plt.show()
    data = pd.DataFrame(X_car)

    # classes 2, 3, 6, 7 has the least number of frequency. Thus we will use those classes as the outliers.

    clustering = DBSCAN(eps=45, min_samples=15).fit(data)
    print(clustering.labels_)
    print(type(clustering.labels_))
    print(np.array(y_car[0]))
    # print(type(clustering.labels_))

    copy_clusterings = clustering.labels_.copy()
    index = 0
    for i in copy_clusterings:
        if i == -1:
            copy_clusterings[index] = 1
        else:
            copy_clusterings[index] = 0
        index += 1

    print('false alarm rate: ', metrics.falsealarmrate(np.array(y_car[0]), [1], copy_clusterings, 1))
    print(jaccard_score(np.array(y_car[0]), copy_clusterings, average=None))


def test():
    pass

if __name__ == '__main__':
    main()
