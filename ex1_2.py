import os

import numpy
from numpy.core.multiarray import ndarray
from numpy.linalg import svd, norm
from matplotlib.pyplot import *
import scipy.misc as misc


def read_rows(row_count):
    headers = {}
    matrix = None
    row_index = 0
    for line_index, line in enumerate(lines):
        values = line.split(',')
        if not headers:
            for i, value in enumerate(values):
                headers[value] = i
            matrix = ndarray((row_count, len(headers)))
        else:
            if len(values) == len(headers):
                for i in range(0, len(values)):
                    matrix[row_index, i] = float(values[i].strip('\"')[:4])
                row_index += 1
                if row_index == row_count :
                    break
    return headers, matrix


if __name__=='__main__':
    file_name = 'kc_house_data.csv'

    try:
        with open(file_name) as file_:
            lines = file_.readlines()
    except:
        print('Failed to open ' + file_name)

    headers, matrix = read_rows(100)

    for row in range(len(matrix)):
        print(matrix[row, headers['lat']])

    decomposed = svd(matrix)
    u, s, vh = decomposed
    t = u[:, :matrix.shape[1]]
    ok = np.allclose(matrix, np.dot(t * s, vh))
    print(ok)
