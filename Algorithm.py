import numpy as np
from matplotlib import pyplot as plt
import cv2

# hyperparameters
FIGSIZE = (20, 20)
FRAMES_UNTIL_TURNING = 10
NUM_OF_FRAME_FOR_LANE_CHANGE = 60
LINE_SLOPE = (0.5, 2)
SHOW = False  # for debug

class Algorithm:
    def __init__(self, parameters):
        self.parameters = parameters
        
        if not self.is_crosswalk():
            self.counter_legal_lane_change = 0
            self.is_turning = False
            self.turning_direction = None
            self.turning_counter = 0
            self.turning = False
    
    def add_text_overlay(self, frame, text, font_size=1.0):
        # Choose font and position
        font = cv2.FONT_HERSHEY_SIMPLEX
        position = (200, 600)

        # Add text to the frame
        cv2.putText(frame, text, position, font, font_size, (255, 0, 0), 6, cv2.LINE_AA)
        return frame

    def resize_line(self, line):
        x1, y1, x2, y2 = line
        slope = self.calculate_slope(x1, x2, y1, y2)
        intercept = y1 - slope * x1
        y1 = self.parameters.lower_bound
        y2 = self.parameters.upper_bound
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return x1, y1, x2, y2

    def mask_frame(self, img, vertices):
        mask = np.zeros_like(img)
        mask = cv2.fillPoly(mask, [vertices], (255, 255, 255), 0)
        masked_img = cv2.bitwise_and(img, mask)
        return masked_img

    def draw_lines(self, img, lines, color=(0, 255, 0), thickness=6):
        for line in lines:
            x1, y1, x2, y2 = self.resize_line(line)
            img = cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        return img

    def calculate_slope(self, x1, x2, y1, y2):
        return 0 if (x1 == x2) else (y1 - y2) / (x1 - x2)

    def divide_lines(self, lines, slope_threshold=LINE_SLOPE):
        right_lines = []
        left_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                slope = self.calculate_slope(x1, x2, y1, y2)
                if slope_threshold[0] < abs(slope) < slope_threshold[1]:
                    if slope > 0:
                        right_lines.append(line[0])
                    else:
                        left_lines.append(line[0])

        return np.array(left_lines), np.array(right_lines)

    def preprocess_frame(self, frame):
        temp = frame.copy()

        # brg to gray
        temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)

        # Apply gamma correction
        if self.is_night():
            gamma = 0.92
            temp = np.uint8(cv2.pow(temp / 255.0, gamma) * 255)
            
            self.show_image(temp, title="gamma", cmap='gray')

        # reducing noise
        temp = cv2.GaussianBlur(temp, (7, 7), 0)
        self.show_image(temp, title="gaussian", cmap='gray')

        # applying threshold to emphasize white
        _, temp = cv2.threshold(temp, self.parameters.lower_threshold, 255, cv2.THRESH_BINARY)
        self.show_image(temp, title="threshold", cmap='gray')

        if not self.is_crosswalk():
            # applying canny to get edges
            temp = cv2.Canny(temp, 5, 100)
            self.show_image(temp, title="Canny", cmap='gray')

        # mask the frame to keep only the relevant edges
        temp = self.mask_frame(temp, self.parameters.vertices)
        
        self.show_image(temp, title="preprocess", cmap='gray')
        
        return temp


    def get_line_and_detect_change(self, left_lines, right_lines):
        best_lines = []

        if left_lines.size == 0:
            return best_lines, True, "Changing to left lane"
        elif right_lines.size == 0:
            return best_lines, True, "Changing to right lane"
        else:
            right_line = np.mean(right_lines, axis=0)
            left_line = np.mean(left_lines, axis=0)
            best_lines.append(right_line)
            best_lines.append(left_line)
            return best_lines, False, ""


    def check_if_turning(self, change, direction, is_turning, turning_direction, counter_legal_lane_change):
        turning = False
        turning_counter = 0
        
        if change:
            if not is_turning:
                turning_direction = direction
                counter_legal_lane_change += 1
                is_turning = True
            else:
                if counter_legal_lane_change < FRAMES_UNTIL_TURNING:
                    if not turning_direction == direction:
                        counter_legal_lane_change = 0
                        turning_direction = direction
                    else:
                        counter_legal_lane_change += 1
                else:
                    turning = True
                    counter_legal_lane_change = 0
                    turning_counter = NUM_OF_FRAME_FOR_LANE_CHANGE
        else:
            is_turning = False
            counter_legal_lane_change = 0

        return turning, is_turning, turning_direction, counter_legal_lane_change, turning_counter

    def output_result(self, out_path, frame_size, frames, fps):
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, frame_size)

        for frame in frames:
            out.write(frame)

        out.release()

    def show_image(self, img, title, cmap=None, show=False):
        if not show and not SHOW:
            return
        
        plt.figure(figsize=FIGSIZE)
        plt.imshow(img, cmap=cmap)
        plt.title(title)
        plt.show()
        
    def is_day(self):
        return self.parameters.algo_type == 'day'
    
    def is_night(self):
        return self.parameters.algo_type == 'night'
    
    def is_crosswalk(self):
        return self.parameters.algo_type == 'crosswalk'

    def detec_lane(self, frame):
        res = frame.copy()
        image = frame.copy()
        lines = []

        if not self.turning:
            image = self.preprocess_frame(image)

            hough_parameters = self.parameters.hough_parameters
            lines = cv2.HoughLinesP(image, rho=hough_parameters.rho, theta=hough_parameters.theta, threshold=hough_parameters.threshold, 
                                    minLineLength=hough_parameters.min_line_length, maxLineGap=hough_parameters.max_line_gap)

            left_lines, right_lines = self.divide_lines(lines, slope_threshold=LINE_SLOPE)
            lines, change, direction = self.get_line_and_detect_change(left_lines, right_lines)

            self.turning, self.is_turning, self.turning_direction, self.counter_legal_lane_change, self.turning_counter = self.check_if_turning(
                change, direction, self.is_turning, self.turning_direction, self.counter_legal_lane_change)

        if self.turning:
            res = self.add_text_overlay(res, self.turning_direction, 4)
            self.turning_counter -= 1
            if self.turning_counter == 0:
                self.turning = False
        else:
            res = self.draw_lines(res, lines)
        
        return res
    
    def detect_crosswalk(self, frame):
        res = frame.copy()
        image = frame.copy()
        image = self.preprocess_frame(image)
        
        contours, _hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        rectangleList = []
        
        for contour in contours:
            if cv2.contourArea(contour) < 1000:
                continue
            
            (x,y,w,h) = cv2.boundingRect(contour)
            
            rectangleList.append([(x, y), (x+w, y+h)])
            # cv2.rectangle(res, (x,y), (x+w,y+h), (0,255,0), 2)
            #cv2.drawContours(temp, contour, -1, (0, 225, 0), 3)
        
        if len(rectangleList) > 4:
            for pointA, pointB in rectangleList:
                cv2.rectangle(res, pointA, pointB, (0,255,0), 2)
                
        # (x,y,w,h) = cv2.groupRectangles(rectangleList, 1, eps = 0.2)
        # cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        # self.show_image(frame, title="Contours", show=True)
        return res
    
    def run(self, video_path, out_path):
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("Could not load video file")

        # Get the video data
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        wanted_fps = fps

        if SHOW:
            print(f"{frame_count=}, {fps=}, {frame_width=}, {frame_height=}")
            
        frames = []

        for frame_num in range(0, frame_count, fps // wanted_fps):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()

            if not ret:
                print("Could not load the frame")
                break
                
            if not self.is_crosswalk():
                res = self.detec_lane(frame)
            else:
                res = self.detect_crosswalk(frame)
            
            self.show_image(res, title=frame_num)

            frames.append(res)

        self.output_result(out_path, (frame_width, frame_height), frames, wanted_fps)

        # Release the video capture object
        cap.release()