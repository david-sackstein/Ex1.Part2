from numpy.core.multiarray import ndarray
from numpy.linalg import svd, norm, eig
from matplotlib.pyplot import *
from scipy import stats, random, math

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
    # 'zipcode',
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
ZIP_CODE_INDEX = 16


def inverse_svd(u, s, vh):
    u = u[:, :vh.shape[1]]
    return np.dot(u * s, vh)


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


def calc_linear_predictor(data, prices):
    u, s, vh = svd(data)
    s_sword = calc_sword(s)
    inverse = inverse_svd(u, s_sword, vh)
    return np.dot(np.transpose(inverse), prices)


def _parse_date(value):
    date_value = value[0:8]
    year = int(date_value[0:4])
    month = int(date_value[4:6])
    day = int(date_value[6:8])
    return year, month, day


def get_valid_rows(lines):
    for line in lines:
        values = line.split(',')
        if len(values) != original_column_count:
            continue
        if not values[DATE_INDEX]:
            continue
        if len(values[DATE_INDEX]) < 8:
            continue
        yield [v.strip('\"') for v in values]


def read_data(lines):

    valid_rows = list(get_valid_rows(lines))
    row_count = len(valid_rows)

    data = ndarray((row_count, column_count))
    prices = ndarray((row_count,))

    for row_index, row in enumerate(valid_rows):
        k = 0
        for i in range(original_column_count):
            value = row[i]
            if i == ID_INDEX or i == ZIP_CODE_INDEX:  # skip
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
    return data, prices


def calc_rmse(lhs, rhs):
    differences = np.asarray([float(d) for d in lhs - rhs if not math.isnan(d)])
    differences_squared = differences ** 2
    mean_of_differences_squared = differences_squared.mean()
    rmse = np.sqrt(mean_of_differences_squared)
    return rmse


def run(file_name):
    with open(file_name) as file_:
        lines = file_.readlines()

    data, prices = read_data(lines[1:])
    indexes = set(range(len(data)))
    
    for train_percent in range(10, 95, 10):

        train_size = int(train_percent * data.shape[0] / 100)
        # TODO make this a random selection of indexes
        train_indexes = range(0, train_size)
        train_data = [v for i, v in enumerate(data) if i in train_indexes]
        train_prices = [v for i, v in enumerate(prices) if i in train_indexes]

        predictor = calc_linear_predictor(train_data, train_prices)

        test_indexes = sorted(indexes - set(train_indexes))
        test_data = [v for i, v in enumerate(data) if i in test_indexes]
        test_prices = [v for i, v in enumerate(prices) if i in test_indexes]

        new_prices = np.dot(test_data, predictor)
        rmse = calc_rmse(new_prices, test_prices)

        print(rmse)


if __name__ == '__main__':
    data_file_name = 'kc_house_data.csv'
    run(data_file_name)


# ======================================================

def tests(data):
    decomposed = svd(data)
    u, s, vh = decomposed
    reconstructed = inverse_svd(u, s, vh)
    ok = np.allclose(data, reconstructed)
    print(ok)


def print_data_and_prices(data, prices):
    for row in range(len(data)):
        message = ''
        for header_name in header_names:
            message += '{} = {} '.format(header_name, data[row, headers[header_name]])
        print(message, 'prices = ', prices[row])
