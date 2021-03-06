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
            depths.append(child.depth())

        return max(depths) + 1

    def dep(self):
        if self.parent is None:
            return 1
        else:
            return self.parent.depth() + 1

    def print(self):
        tabs = " " * (27 - self.dep())
        print(tabs, self.label, self.classCounts, self.splitFeature, self.splitFeatureValue)
        for child in self.children:
            child.print()
