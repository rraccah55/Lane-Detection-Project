import numpy as np

class Parameters:
    def __init__(self, algo_type, upper_bound, lower_bound, vertices, lower_threshold, hough_parameters):
        self.algo_type = algo_type
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.lower_threshold = lower_threshold
        self.hough_parameters = hough_parameters
        
        if vertices is not None:
            self.vertices = np.array(vertices, np.int32).reshape((-1, 1, 2))