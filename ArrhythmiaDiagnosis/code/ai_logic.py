import math
from ArrhythmiaDiagnosis.code.tree import Tree


def dataToDistribution(data) -> [float]:
    """ Turn a dataset which has n possible classification labels into a
        probability distribution with n entries. """
    all_attrs = [attr for _, attr in data]

    num_entries = len(all_attrs)

    possibleLabels = set(all_attrs)
    dist = []
    for aLabel in possibleLabels:
        dist.append(float(all_attrs.count(aLabel)) / num_entries)

    return dist


def entropy(dist) -> float:
    return -sum([p * math.log(p, 2) for p in dist])


def majorityVote(data, node: Tree) -> Tree:
    attrs = [attr for _, attr in data]
    choice = max(set(attrs), key=attrs.count)
    node.label = choice
    node.classCounts = dict([(label, attrs.count(label)) for label in set(attrs)])

    return node


def homogenous(data):
    return len(set([row[-1] for row in data])) <= 1

# TODO: c45 tree testing

# TODO: c45 tree building
def gain(data, index):
    entropyGain = entropy(dataToDistribution(data))

    for dataSubset in splitData(data, featureIndex):
        entropyGain -= entropy(dataToDistribution(dataSubset))

    return entropyGain


def c45(data, root, remaining_features) -> Tree:
    if homogenous(data):
        return root
    if len(remaining_features) == 0:
        return majorityVote(data, root)

    # find the index of the best feature to split on
    best_feature = max(remaining_features, key=lambda index: gain(data, index))

# TODO: c45 tree building
from ArrhythmiaDiagnosis.code.tree import Tree


def c45(data, attributes):
    if pure(data) or (other stopping criteria):
        return
    criteria = []
    for attr in attributes:
        criteria.append(compute_criteria(attr))
    crt = select_best_criteria(criteria)
    tree = DecisionNode(crt)
    sub_data = induced subdatasets from data based on crt
    for sub in sub_data:
        node = c45(sub,attributes - crt)
        tree.add node
    return tree

def entropy(dist) -> float:
    return -sum([p * math.log(p, 2) for p in dist])

# def gain(data,)


def buildDecisionTree(data, root, remainingClassifications):
    pass


def homogeneous(data):
    return len(set([row[-1] for row in data])) <= 1


def decisionTree(data: [(list, int)]):
    return buildDecisionTree(data, Tree(), set(range(len(data[0][0]))))

# TODO: c45 tree testing
