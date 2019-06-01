import pickle


def read_data(filename: str):
    with open(filename, 'r') as inputFile:
        lines = inputFile.readlines()

    data = [line.strip().split(',') for line in lines]
    data = [(row[0:-1], row[-1]) for row in data]

    return data


def pickle_data(data, filename="datadump.pickle"):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)
    return


def unpickle_data(filename="datadump.pickle"):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data
