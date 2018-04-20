def read_rows():
    headers = {}
    rows = []
    for line_index, line in enumerate(lines):
        values = line.split(',')
        if not headers:
            for i, value in enumerate(values):
                headers[value] = i
        else:
            if len(values) == len(headers):
                rows.append(values)
    return headers, rows


if __name__=='__main__':
    file_name = 'kc_house_data.csv'

    try:
        with open(file_name) as file_:
            lines = file_.readlines()
    except:
        print('Failed to open ' + file_name)

    headers, rows = read_rows()

    for row in rows:
        print(row[headers['lat']])