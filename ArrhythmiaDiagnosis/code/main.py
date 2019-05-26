import ArrhythmiaDiagnosis.code.dataIO as io
from ArrhythmiaDiagnosis.code.ai_logic import c45

def main():
    filename = "../data/arrhythmia_teach.data"
    data: [(list, int)] = io.read_data(filename)

    tree = c45(data)

    # linear = "linear"
    # nominal = "nominal"
    # valuetypes = ["n"] * 280
    # valuetypes[0] = linear
    # valuetypes[1] = nominal
    # for i in range(2, 21):
    #     valuetypes[i] = linear
    # for i in range(21, 27):
    #     valuetypes[i] = nominal
    # for i in range(27, 279):
    #     valuetypes[i] = linear
    # valuetypes[279] = "classification"

    # results = ["n"] * 16
    # results[0] = "normal"
    # for i in range(1, 15):
    #     results[i] = "arrythmia class " + i.__str__()
    # results[15] = "unclassified"
    return


if __name__ == '__main__':
    main()
