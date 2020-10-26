import pandas as pd
import numpy as np


def falsealarmrate(olabels, oelement, clabels, celement):
    N = np.count_nonzero(olabels == oelement)  # count the negatives in ground truth
    negativeOarray = (olabels == oelement)  # get the binary ground truth negative element array
    correspondingClabelarr = clabels[negativeOarray]  # create the expected negative cluster label array
    fp = len(correspondingClabelarr) - np.count_nonzero(correspondingClabelarr == celement)  # count the false positives
    result = N if N == 0 else fp / N
    return result


def main():
    pass


def test():
    alabels = np.array(['n', 'n', 'n', 'n', 'n', 'o', 'n', 'n', 'n', 'n', 'o', 'n', 'n', 'n', 'n', 'o'])
    clabels = np.array([0, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0])
    print(alabels)
    print(clabels)
    print(np.count_nonzero(alabels == 'o'))

    boolarray = alabels == 'o'
    print(boolarray)
    newarr = clabels[boolarray]
    print(newarr)
    fp = len(newarr) - np.count_nonzero(newarr == -1)
    print(fp)

    falsealarmrate(alabels, 'o', clabels, -1)


if __name__ == '__main__':
    test()
