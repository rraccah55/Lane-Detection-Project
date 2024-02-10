from Parameters import Parameters, HoughLinesParameters, CrosswalkParameters
from Algorithm import Algorithm

hough_parameters_day = HoughLinesParameters()
hough_parameters_night = HoughLinesParameters(min_line_length = 10)

crosswalk_parameters = CrosswalkParameters([[600, 800], [1700, 800], [1700, 1000], [600, 1000]])

parameters_day = Parameters(700, 1080, [[450, 1050], [900, 620], [1250, 650], [1700, 1050]], 150, hough_parameters_day)
parameters_night = Parameters(410, 550, [[380, 380], [180, 530], [600, 380], [750, 530]], 115, hough_parameters_night, gamma_correction=0.92)
parameters_crosswalk = Parameters(410, 550, [[450, 1050], [900, 620], [1250, 650], [1700, 1050]], 180, hough_parameters_day, crosswalk_parameters=crosswalk_parameters)

if __name__ == "__main__":    
    #Algorithm(parameters_day).run('day.mp4', 'day_out.avi')
    #Algorithm(parameters_night).run('night.mp4', 'night_out.avi')
    Algorithm(parameters_crosswalk).run('crosswalk.mp4', 'crosswalk_out.avi')