class Patient:  # Do I even need this?
    def __init__(self, data: list):
        self.classification: int = data.pop(-1)
        self.data: list = data
