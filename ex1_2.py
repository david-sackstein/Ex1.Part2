import os

import numpy
from numpy.core.multiarray import ndarray
from numpy.linalg import svd, norm
from matplotlib.pyplot import *
import scipy.misc as misc

headers = {
    'id': 0,
    'year': 1, 'month': 2, 'day': 3,  # these replace 'date'
    'price': 4,
    'bedrooms': 5,
    'bathrooms': 6,
    'sqft_living': 7,
    'sqft_lot': 8,
    'floors': 9,
    'waterfront': 10,
    'view': 11,
    'condition': 12,
    'grade': 13,
    'sqft_above': 14,
    'sqft_basement': 15,
    'yr_built': 16,
    'yr_renovated': 17,
    'zipcode': 18,
    'lat': 19,
    'long': 20,
    'sqft_living15': 21,
    'sqft_lot15': 22
}

header_count = len(headers)
original_header_count = header_count - 2


def read_data(lines, row_count):
    data = ndarray((row_count, header_count))
    row_index = 0
    for line in lines[1:]:
        values = line.split(',')
        if len(values) != original_header_count:
            continue
        k = 0
        for i in range(len(values)):
            if i == 1:
                year, month, day = _parse_date(values[i])
                data[row_index, 1] = year
                data[row_index, 2] = month
                data[row_index, 3] = day
                k = k + 3
            else:
                data[row_index, k] = float(values[i].strip('\"'))
                k = k + 1
        row_index += 1
        if row_index == row_count:
            break
    return data


def _parse_date(value):
    date_value = value[1:9]
    year = int(date_value[0:4])
    month = int(date_value[4:6])
    day = int(date_value[6:8])
    return year, month, day


def run(file_name):
    try:
        with open(file_name) as file_:
            lines = file_.readlines()
    except:
        print('Failed to open ' + file_name)

    data = read_data(lines, 100)

    for row in range(len(data)):
        print(data[row, headers['lat']])

    decomposed = svd(data)
    u, s, vh = decomposed
    t = u[:, :data.shape[1]]
    ok = np.allclose(data, np.dot(t * s, vh))
    print(ok)


if __name__ == '__main__':
    run('kc_house_data.csv')
