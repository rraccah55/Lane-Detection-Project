import numpy as np

class Parameters:
    def __init__(self, upper_bound, lower_bound, vertices, lower_threshold, hough_parameters, gamma_correction=None, crosswalk_parameters=None):
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.lower_threshold = lower_threshold
        self.hough_parameters = hough_parameters
        self.vertices = np.array(vertices, np.int32).reshape((-1, 1, 2))
        self.gamma_correction = gamma_correction
        self.crosswalk_parameters = crosswalk_parameters

class HoughLinesParameters:
    def __init__(self, rho=1, theta=np.pi/180, threshold=10, min_line_length=40, max_line_gap=100):
        self.type = type
        self.rho = rho
        self.theta = theta
        self.threshold = threshold
        self.max_line_gap = max_line_gap
        self.min_line_length = min_line_length

class CrosswalkParameters:
    def __init__(self, vertices):
        self.vertices = np.array(vertices, np.int32).reshape((-1, 1, 2))