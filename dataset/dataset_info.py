class DatasetInfo:
    """
    Class used to gather associated information about a data record.
    """
    def __init__(self, images_AVIRIS, images_WV3, mag1c, labels):
        self.images_AVIRIS = images_AVIRIS
        self.images_WV3 = images_WV3
        self.mag1c = mag1c
        self.labels = labels
