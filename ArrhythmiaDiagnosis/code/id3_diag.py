# Python code  for implementing Decision Tree (ID3 Algorithm for classification)
# This code can predict the types of disease causes by combination of particular symptom


import math


class Tree:
    def __init__(self, parent=None):
        self.parent = parent
        self.children = []
        self.label = None
        self.classCounts = None
        self.splitFeatureValue = None
        self.splitFeature = None

    def depth(self):
        depths = [0]
        child: Tree
        for child in self.children:
            depths.append(child.depth)
        return max(depths) + 1


def dataToDistribution(data):
    allLabels = [label for (point, label) in data]

    numEntries = len(allLabels)

    possibleLabels = set(allLabels)
    dist = []
    for aLabel in possibleLabels:
        dist.append(float(allLabels.count(aLabel)) / numEntries)

    return dist


def entropy(dist):
    return -sum([p * math.log(p, 2) for p in dist])


def splitData(data, featureIndex):
    attrValues = [point[featureIndex] for (point, label) in data]

    for aValue in set(attrValues):
        dataSubset = [(point, label) for (point, label) in data
                      if point[featureIndex] == aValue]

        yield dataSubset


def gain(data, featureIndex):
    entropyGain = entropy(dataToDistribution(data))

    for dataSubset in splitData(data, featureIndex):
        entropyGain -= entropy(dataToDistribution(dataSubset))

    return entropyGain


def homogeneous(data):
    return len(set([label for (point, label) in data])) <= 1


def majorityVote(data, node):
    labels = [label for (pt, label) in data]
    choice = max(set(labels), key=labels.count)
    node.label = choice
    node.classCounts = dict([(label, labels.count(label)) for label in set(labels)])

    return node


def buildDecisionTree(data, root, remainingFeatures):
    if homogeneous(data):
        root.label = data[0][1]
        root.classCounts = {root.label: len(data)}
        return root

    if len(remainingFeatures) == 0:
        return majorityVote(data, root)

    # find the index of the best feature to split on
    bestFeature = max(remainingFeatures, key=lambda index: gain(data, index))

    if gain(data, bestFeature) == 0:
        return majorityVote(data, root)

    root.splitFeature = bestFeature

    # add child nodes and process recursively
    for dataSubset in splitData(data, bestFeature):
        aChild = Tree(parent=root)
        aChild.splitFeatureValue = dataSubset[0][0][bestFeature]
        root.children.append(aChild)

        buildDecisionTree(dataSubset, aChild, remainingFeatures - {bestFeature})

    return root


def decisionTree(data):
    return buildDecisionTree(data, Tree(), set(range(len(data[0][0]))))


def dictionarySum(*dicts):
    sumDict = {}

    for aDict in dicts:
        for key in aDict:
            if key in sumDict:
                sumDict[key] += aDict[key]
            else:
                sumDict[key] = aDict[key]

    return sumDict


def predictRecursive(tree, point):
    if not tree.children:
        return tree.classCounts
    elif point[tree.splitFeature] == '?':
        dicts = [predictRecursive(child, point) for child in tree.children]
        return dictionarySum(*dicts)
    else:
        matchingChildren = [child for child in tree.children
                            if child.splitFeatureValue == point[tree.splitFeature]]

        return predictRecursive(matchingChildren[0], point)


def predict(tree, point):
    counts = predictRecursive(tree, point)

    if len(counts.keys()) == 1:
        return max(counts.keys())
    else:
        return max(counts.keys(), key=lambda k: counts[k])


def testClassification(data, tree):
    actual_labels = [label for _, label in data]
    predicted_labels = [predict(tree, row) for row, _ in data]

    correct_labels = [(1 if a == b else 0) for a, b in zip(actual_labels, predicted_labels)]
    return float(sum(correct_labels)) / len(actual_labels)


def main():
    with open('../data/arrhythmia.data', 'r') as inputFile:
        lines = inputFile.readlines()

    data = [line.strip().split(',') for line in lines]
    data = [(row[0:-1], row[-1]) for row in data]
    tree = decisionTree(data)

    # testowanie
    print(testClassification(data, tree))
    print(tree.depth())


if __name__ == '__main__':
    main()
