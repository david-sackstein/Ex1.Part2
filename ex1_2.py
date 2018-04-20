import os

import numpy
from numpy.core.multiarray import ndarray
from numpy.linalg import svd, norm
from matplotlib.pyplot import *
import scipy.misc as misc


def read_data(lines, row_count):
    headers = {}
    data = None
    row_index = 0
    for line in lines:
        values = line.split(',')
        if not headers:
            for i, value in enumerate(values):
                headers[value] = i
            data = ndarray((row_count, len(headers)))
        else:
            if len(values) == len(headers):
                for i in range(0, len(values)):
                    data[row_index, i] = float(values[i].strip('\"')[:4])
                row_index += 1
            if row_index == row_count:
                break
    return headers, data


def run(file_name):
    try:
        with open(file_name) as file_:
            lines = file_.readlines()
    except:
        print('Failed to open ' + file_name)

    headers, data = read_data(lines, 100)

    for row in range(len(data)):
        print(data[row, headers['lat']])

    decomposed = svd(data)
    u, s, vh = decomposed
    t = u[:, :data.shape[1]]
    ok = np.allclose(data, np.dot(t * s, vh))
    print(ok)


if __name__=='__main__':
    run('kc_house_data.csv')
