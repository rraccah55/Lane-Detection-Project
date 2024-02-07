import numpy as np

class Parameters:
    def __init__(self, type, upper_bound, lower_bound, vertices, lower_threshold, hough_parameters):
        self.type = type
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.vertices = np.array(vertices, np.int32).reshape((-1, 1, 2))
        self.lower_threshold = lower_threshold
        self.hough_parameters = hough_parameters