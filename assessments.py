import numpy as np


def falsealarmrate(olabels, oelement, clabels, celement):
    negativeOarray = np.full(len(olabels), False)
    for oe in oelement:
        negativeOarray = np.bitwise_or((olabels == oe), negativeOarray)
    N = np.count_nonzero(negativeOarray)  # count the negatives in ground truth
    correspondingClabelarr = clabels[negativeOarray]  # create the expected negative cluster label array
    fp = len(correspondingClabelarr) - np.count_nonzero(correspondingClabelarr == celement)  # count the false positives
    result = N if N == 0 else fp / N
    return result


def main():
    pass


def test2():
    alabels = np.array(['n', 'u', 'n', 'n', 'n', 'o', 'n', 'n', 'n', 'n', 'o', 'n', 'n', 'u', 'n', 'o'])
    clabels = np.array([0, -1, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, -1, 0, 0])
    print(falsealarmrate(alabels, ['o', 'u'], clabels, -1))


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

    print(falsealarmrate(alabels, 'o', clabels, -1))


if __name__ == '__main__':
    test()
    test2()
