import numpy as np

class HoughLinesParameters:
    def __init__(self, rho=1, theta=np.pi/180, threshold=10, min_line_length=40, max_line_gap=100):
        self.type = type
        self.rho = rho
        self.theta = theta
        self.threshold = threshold
        self.max_line_gap = max_line_gap
        self.min_line_length = min_line_length