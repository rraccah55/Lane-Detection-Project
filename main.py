from HoughLinesParameters import HoughLinesParameters
from Parameters import Parameters
from Algorithm import Algorithm

hough_parameters_day = HoughLinesParameters()
hough_parameters_night = HoughLinesParameters(min_line_length = 10)

parameters_day = Parameters("day", 700, 1080, [[450, 1050], [900, 620], [1250, 650], [1700, 1050]], 150, hough_parameters_day)
parameters_night = Parameters("night", 410, 550, [[380, 380], [180, 530], [600, 380], [750, 530]], 115, hough_parameters_night)

if __name__ == "__main__":
    algorithm = Algorithm()
    
    algorithm.run('day.mp4', 'day_out.avi', parameters_day)
    algorithm.run('night.mp4', 'night_out.avi', parameters_night)