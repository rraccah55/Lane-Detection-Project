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

class HoughLinesParameters:
    def __init__(self, rho=1, theta=np.pi/180, threshold=10, min_line_length=40, max_line_gap=100):
        self.type = type
        self.rho = rho
        self.theta = theta
        self.threshold = threshold
        self.max_line_gap = max_line_gap
        self.min_line_length = min_line_length