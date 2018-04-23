from numpy.core.multiarray import ndarray
from numpy.linalg import svd, norm, eig
from matplotlib.pyplot import *
from numpy.random.mtrand import randint
from scipy import stats, random, math

zip_codes_values = [
    98001, 98002, 98003, 98004, 98005, 98006, 98007, 98008, 98010, 98011,
    98014, 98019, 98022, 98023, 98024, 98027, 98028, 98029, 98030, 98031,
    98032, 98033, 98034, 98038, 98039, 98040, 98042, 98045, 98052, 98053,
    98055, 98056, 98058, 98059, 98065, 98070, 98072, 98074, 98075, 98077,
    98092, 98102, 98103, 98105, 98106, 98107, 98108, 98109, 98112, 98115,
    98116, 98117, 98118, 98119, 98122, 98125, 98126, 98133, 98136, 98144,
    98146, 98148, 98155, 98166, 98168, 98177, 98178, 98188, 98198, 98199
]
zip_codes = {value: index for (index, value) in enumerate(zip_codes_values)}

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
    'lat',
    'long',
    'sqft_living15',
    'sqft_lot15'
]

zip_code_header_names = ['zip_{}'.format(z) for z in zip_codes_values]
header_names = header_names[:16] + zip_code_header_names + header_names[16:]
headers = {name: index for (index, name) in enumerate(header_names)}

column_count = len(headers)
original_column_count = 21

ID_INDEX = 0
DATE_INDEX = 1
PRICE_INDEX = 2
ZIP_CODE_INDEX = 16
YEAR_RENOVATED_INDEX = 15


def inverse_svd(u, s, vh):
    return np.dot(u * s, vh)


def calc_sword(sigma):
    threshold = stats.gmean(sigma) / 100
    rank = len(sigma)
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
    u, s, vh = svd(data, False)
    s_sword = calc_sword(s)
    inverse = inverse_svd(u, s_sword, vh)
    return np.dot(np.transpose(inverse), prices)


def _parse_date(value):
    date_value = value[0:8]
    year = int(date_value[0:4])
    month = int(date_value[4:6])
    day = int(date_value[6:8])
    return year, month, day


def is_float(value):
    try:
        return not math.isnan(float(value))
    except ValueError:
        return False


def is_valid(values):
    if len(values) != original_column_count:
        return False
    if not values[DATE_INDEX]:
        return False
    if len(values[DATE_INDEX]) < 8:
        return False
    price = values[PRICE_INDEX]
    if not is_float(price):
        return False
    if float(price) < 1:
        return False
    return True


def get_valid_rows(lines):
    for line in lines:
        values = line.split(',')
        if is_valid(values):
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
            if i == ID_INDEX or i == YEAR_RENOVATED_INDEX:  # skip
                continue
            elif i == PRICE_INDEX:  # price is result not data
                prices[row_index] = float(value)
                continue
            elif i == DATE_INDEX:
                year, month, day = _parse_date(value)
                data[row_index, k] = year
                k = k + 1
                data[row_index, k] = month
                k = k + 1
                data[row_index, k] = day
                k = k + 1
            elif i == ZIP_CODE_INDEX:
                prev_k = k
                for j in range(len(zip_codes_values)):
                    data[row_index, k] = 0
                    k = k + 1
                data[row_index, prev_k + zip_codes[int(value)]] = 1
            else:
                data[row_index, k] = float(value)
                k = k + 1
    return data, prices


def calc_rmse(lhs, rhs):
    differences = lhs - rhs
    differences_squared = differences ** 2
    mean_of_differences_squared = differences_squared.mean()
    rmse = np.sqrt(mean_of_differences_squared)
    return rmse


def run(file_name):
    with open(file_name) as file_:
        lines = file_.readlines()

    data, prices = read_data(lines[1:])

    indexes = [0]*len(data)
    for i in range(len(data)):
        indexes[i] = i
    # indexes = set(range(len(data)))

    percents, test_errors, train_errors = get_percents_and_errors(data, indexes, prices)

    test_error_line, = plot(percents, test_errors, linestyle='--', label='Test Error')
    train_error_line, = plot(percents, train_errors, linestyle='--', label='Train Error')
    legend(handles=[test_error_line, train_error_line])

    ylabel('RMSE')
    xlabel('[%]')
    title('Test Error and Train Error [RMSE] as Functions of Training Set Size in %')
    savefig('ErrorPlots.png')
    clf()


def get_percents_and_errors(data, indexes, prices):
    percents = []
    test_errors = []
    train_errors = []

    data_row_count = data.shape[0]

    for train_percent in range(1, 100, 1):
        train_size = int(train_percent * data_row_count / 100)

        train_indexes, test_indexes = split_data_randomly(data_row_count, train_size)

        train_data, train_prices = get_data_and_prices(data, prices, train_indexes)
        test_data, test_prices = get_data_and_prices(data, prices, test_indexes)

        predictor = calc_linear_predictor(train_data, train_prices)

        percents.append(train_percent)
        # test error
        test_error = _calculate_error_on_data(predictor, test_data, test_prices)
        test_errors.append(test_error)

        # train error
        train_error = _calculate_error_on_data(predictor, train_data, train_prices)
        train_errors.append(train_error)

        print(train_percent)

    return percents, test_errors, train_errors

def _calculate_error_on_data(predictor, data, prices):
    predicted_prices = np.dot(data, predictor)
    rmse = calc_rmse(predicted_prices, prices)
    return rmse


def split_data_randomly(total_size, train_size):
    # create the test_indexes has having all the indexes
    # and the train_indexes as being empty.
    # We will move train_size indexes from test_indexes to train_indexes

    test_indexes = [i for i in range(total_size)]
    train_indexes = []

    for i in range(train_size):
        # we need to remove one of the indexes in test_indexes.
        # but not that test_indexes may contain gaps because we have removed some of its original values.
        # So for instance, if test_indexes now contains [0, 55, 2345, 20000]
        # and we need to remove one more we select a position between 0 and 3 - let's say 1
        # and then we remove 55 which is in place 1 (and move it to train_indexes

        select_position_of_index_to_move = randint(0, len(test_indexes))
        index_to_move = test_indexes[select_position_of_index_to_move]
        test_indexes.remove(index_to_move)
        train_indexes.append(index_to_move)

    return train_indexes, test_indexes


def get_data_and_prices(data, prices, indexes):
    data = [v for i, v in enumerate(data) if i in indexes]
    prices = [v for i, v in enumerate(prices) if i in indexes]
    return data, prices


def test_predictor():
    data = [[1, 1, 0], [2, 4, 6], [1, 2, 3]]
    prices = [1, 5, 3]
    predictor = calc_linear_predictor(data, prices)
    print(predictor)
    # predictor = [1, 1, 1]
    new_prices = np.dot(data, predictor)
    print(new_prices)
    rmse = calc_rmse(new_prices, prices)
    print(rmse)


if __name__ == '__main__':

    # test_predictor()

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
