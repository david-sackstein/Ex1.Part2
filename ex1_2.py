import os

import numpy
from numpy.core.multiarray import ndarray
from numpy.linalg import svd, norm, eig
from matplotlib.pyplot import *
import scipy.misc as misc
from scipy import stats

header_names = [
    'year',
    'month',
    'day',
    'bedrooms',
    'bathrooms',
    'sqft_living',
    'sqft_lot',
    'floors',
    'waterfront',
    'view',
    'condition',
    'grade',
    'sqft_above',
    'sqft_basement',
    'yr_built',
    'yr_renovated',
    'zipcode',
    'lat',
    'long',
    'sqft_living15',
    'sqft_lot15'
]

headers = {name: index for (index, name) in enumerate(header_names)}
column_count = len(headers)
original_column_count = 21

ID_INDEX = 0
DATE_INDEX = 1
PRICE_INDEX = 2
MINIMUM_PRICE = 1000

def read_data(lines, row_count):
    data = ndarray((row_count, column_count))
    prices = ndarray((row_count,))
    row_index = 0
    for line in lines:
        values = line.split(',')
        if len(values) != original_column_count:
            continue
        k = 0
        for i in range(original_column_count):
            value = values[i].strip('\"')
            if i == ID_INDEX:  # skip id
                continue
            elif i == PRICE_INDEX:  # price is result not data
                prices[row_index] = float(value)
                continue
            if i == DATE_INDEX:
                year, month, day = _parse_date(value)
                data[row_index, k] = year
                k = k + 1
                data[row_index, k] = month
                k = k + 1
                data[row_index, k] = day
                k = k + 1
            else:
                data[row_index, k] = float(value)
                k = k + 1
        row_index += 1
        if row_index == row_count:
            break
    return data, prices


def _parse_date(value):
    date_value = value[0:8]
    year = int(date_value[0:4])
    month = int(date_value[4:6])
    day = int(date_value[6:8])
    return year, month, day


def inverse_svd(u, s, vh):
    u = u[:, :vh.shape[0]]
    return np.dot(u * s, vh)


def run(file_name):
    with open(file_name) as file_:
        lines = file_.readlines()

    data, prices = read_data(lines[1:], 100)
    predictor = calc_linear_predictor(data, prices)
    new_prices = np.dot(data, predictor)
    rmse = calc_rmse(new_prices, prices)
    print(rmse)

def calc_rmse(lhs, rhs):
    differences = lhs - rhs
    differences_squared = differences ** 2
    mean_of_differences_squared = differences_squared.mean()
    rmse = np.sqrt(mean_of_differences_squared)
    return rmse

def calc_linear_predictor(data, prices):
    u, s, vh = svd(data)
    s_sword = calc_sword(s)
    inverse = inverse_svd(u, s_sword, vh)
    predictor = np.dot(np.transpose(inverse), prices)
    return predictor


def calc_sword(sigma):
    threshold = stats.gmean(sigma) / 100
    rank = 0
    sigma_sword = ndarray(sigma.shape)
    for i in range(len(sigma)):
        if sigma[i] > threshold:
            sigma_sword[i] = 1 / sigma[i]
        else:
            rank = i
            break
    sigma_sword[rank:] = 0
    return sigma_sword


def tests(data):
    decomposed = svd(data)
    u, s, vh = decomposed
    reconstructed = inverse_svd(u, s, vh)
    ok = np.allclose(data, reconstructed)
    print(ok)
    f = np.transpose(data)
    a = np.dot(data, np.transpose(data))
    w, v = eig(a)
    reconstructed = numpy.dot(v * w, np.transpose(v))
    ok = np.allclose(a, reconstructed)
    print(ok)


def print_data_and_prices(data, prices):
    for row in range(len(data)):
        message = ''
        for header_name in header_names:
            message += '{} = {} '.format(header_name, data[row, headers[header_name]])
        print(message, 'prices = ', prices[row])


if __name__ == '__main__':
    data_file_name = 'kc_house_data.csv'
    run(data_file_name)
