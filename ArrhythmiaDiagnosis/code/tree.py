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
