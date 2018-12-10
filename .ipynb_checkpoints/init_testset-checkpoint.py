# this file randomly generate two set of points of number 10 and 30, each in a unit square.
# this script should be executed only once
import numpy as np

if __name__ == '__main__':
    test10 = np.random.random((10, 2))
    test30 = np.random.random((30, 2))
    with open('test10.txt', 'x') as f:
        for (x, y) in test10:
            print(x, y, file=f)
    with open('test30.txt', 'x') as f:
        for (x, y) in test30:
            print(x, y, file=f)

