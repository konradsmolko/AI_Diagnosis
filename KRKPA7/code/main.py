from random import Random

from KRKPA7.code.dataIO import read_data
from KRKPA7.code.ai_logic import decisionTree, test, predict


def main():
    data = read_data('../data/kr-vs-kp.data')
    x = Random(160698)
    x.shuffle(data, x.random)
    tree = decisionTree(data[:2000])
    # tree.print()
    print(tree.depth())
    print(test(data[2000:], tree))


if __name__ == '__main__':
    main()
