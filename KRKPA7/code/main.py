from random import Random

from KRKPA7.code.dataIO import read_data
from KRKPA7.code.ai_logic import decisionTree, test, predict


def main():
    data = read_data('../data/kr-vs-kp.data')
    labels = ['bkblk', 'bknwy', 'bkon8', 'bkona', 'bkspr', 'bkxbq', 'bkxcr', 'bkxwp', 'blxwp', 'bxqsq', 'cntxt',
              'dsopp', 'dwipd', 'hdchk', 'katri', 'mulch', 'qxmsq', 'r2ar8', 'reskd', 'reskr', 'rimmx', 'rkxwp',
              'rxmsq', 'simpl', 'skach', 'skewr', 'skrxp', 'spcop', 'stlmt', 'thrsk', 'wkcti', 'wkna8', 'wknck',
              'wkovl', 'wkpos', 'wtoeg', 'CLASS']
    rand = Random(160698)
    results = []
    tries = 1000
    for _ in range(tries):
        rand.shuffle(data, rand.random)
        split_point = rand.randint(2000, 2500)
        tree = decisionTree(data[:split_point])
        while tree.depth() >= 20 or tree.depth() < 6:
            rand.shuffle(data, rand.random)
            split_point = rand.randint(2000, 2500)
            tree = decisionTree(data[:split_point])
        results.append(test(data[split_point:], tree))
        # tree.print()
        # print("Splitpoint:", split_point, "Depth:", tree.depth(), "Accuracy:", results[-1])

    print("Average accuracy over", tries, "tries:", sum(results)/tries)


if __name__ == '__main__':
    main()
