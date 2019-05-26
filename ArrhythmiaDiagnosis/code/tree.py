class Tree:
    def __init__(self, parent=None):
        self.parent = parent
        self.children = []
        self.label = None
        self.classCounts = None
        self.splitFeatureValue = None
        self.splitFeature = None
