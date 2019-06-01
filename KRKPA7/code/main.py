from KRKPA7.code.dataIO import read_data
from KRKPA7.code.ai_logic import decisionTree, test, predict


def main():
    data = read_data('../data/arrhythmia.data')
    tree = decisionTree(data)
    tree.print()
    print(tree.depth())
    print(test(data, tree))


if __name__ == '__main__':
    main()
