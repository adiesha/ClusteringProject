import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import assessments as metrics
import matplotlib.pyplot as plt


def main():
    data = pd.read_csv('data/shuttle-unsupervised-trn.csv', header=None)
    print(data.head())
    print(data.describe())
    y = data[9]
    y.hist()
    plt.show()

    # classes 2, 3, 6, 7 has the least number of frequency. Thus we will use those classes as the outliers.

    clustering = DBSCAN(eps=5, min_samples=6).fit(data.iloc[:, 0:8])
    print(clustering.labels_)
    print(type(clustering.labels_))
    # print(type(clustering.labels_))
    for index, row in data.iterrows():
        print(str(row[9]) + ":" + str(clustering.labels_[index]))

    # print('false alarm rate: ', metrics.falsealarmrate(data.iloc[:, 36].values, 'o', clustering.labels_, -1))


if __name__ == '__main__':
    main()
