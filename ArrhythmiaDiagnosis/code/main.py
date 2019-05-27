from ArrhythmiaDiagnosis.code.dataIO import read_data
from ArrhythmiaDiagnosis.code.ai_logic import decisionTree, test, predict


def main():
    data = read_data('../data/arrhythmia.data')
    tree = decisionTree(data)

    # testowanie
    # print(test(data[0:50], tree))
    # print(test(data[50:100], tree))
    # print(test(data[100:150], tree))
    # print(test(data[150:200], tree))
    # print(test(data[200:250], tree))
    # print(test(data[250:300], tree))
    # print(test(data[250:300], tree))
    # print(test(data[300:350], tree))
    # print(test(data[350:400], tree))
    # print(test(data[400:], tree))
    # print(predict(tree, data[0][0]))
    # print(data[0][1])
    print(test(data, tree))
    print(tree.depth())


if __name__ == '__main__':
    main()
