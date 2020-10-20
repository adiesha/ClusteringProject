import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


def main():
    data = pd.read_csv('data/satellite-unsupervised-ad.csv', header=None)
    print(data.head())
    clustering = DBSCAN(eps=42, min_samples=7).fit(data.iloc[:, 0:35])
    print()
    print(clustering.labels_)
    print(type(clustering.labels_))
    for index, row in data.iterrows():
        print(row[36] + " " + str(clustering.labels_[index]))
    pass


if __name__ == '__main__':
    main()
