import pickle


def read_data(filename: str) -> [(list, int)]:
    with open(filename, 'r') as inputFile:
        lines = inputFile.readlines()

    data = []
    for raw_row in [line.strip().split(',') for line in lines]:
        row = []
        for item in raw_row:
            if item.__contains__('.'):
                row.append(float(item))
            elif item.__eq__('?'):
                row.append(None)
            else:
                row.append(int(item))
        attr = row.pop(-1)
        data.append((row, attr))

    return data


def pickle_data(data, filename="datadump.pickle"):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)
    return


def unpickle_data(filename="datadump.pickle"):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data
