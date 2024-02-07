import numpy as np
from matplotlib import pyplot as plt
import cv2

# hyperparameters
FIGSIZE = (20, 20)
FRAMES_UNTIL_TURNING = 10
WANTED_FPS = 30
NUM_OF_FRAME_FOR_LANE_CHANGE = 60
LINE_SLOPE = (0.5, 2)
SHOW = False  # for debug

class Algorithm:
    def add_text_overlay(self, frame, text, font_size=1.0):
        # Choose font and position
        font = cv2.FONT_HERSHEY_SIMPLEX
        position = (200, 600)

        # Add text to the frame
        cv2.putText(frame, text, position, font, font_size, (255, 0, 0), 6, cv2.LINE_AA)
        return frame

    def resize_line(self, line, parameters):
        x1, y1, x2, y2 = line
        slope = self.calculate_slope(x1, x2, y1, y2)
        intercept = y1 - slope * x1
        y1 = parameters.lower_bound
        y2 = parameters.upper_bound
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return x1, y1, x2, y2

    def mask_frame(self, img, vertices):
        mask = np.zeros_like(img)
        mask = cv2.fillPoly(mask, [vertices], (255, 255, 255), 0)
        masked_img = cv2.bitwise_and(img, mask)
        return masked_img

    def draw_lines(self, img, lines, parameters, color=(0, 255, 0), thickness=6):
        for line in lines:
            x1, y1, x2, y2 = self.resize_line(line, parameters)
            img = cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        return img

    def calculate_slope(self, x1, x2, y1, y2):
        return 0 if (x1 == x2) else (y1 - y2) / (x1 - x2)

    def divide_lines(self, lines, slope_threshold=(0.5, 2)):
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


    def preprocess_frame(self, frame, parameters):
        temp = frame.copy()

        # brg to gray
        temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)

        # Apply gamma correction
        if parameters.type == "night":
            gamma = 0.92
            temp = np.uint8(cv2.pow(temp / 255.0, gamma) * 255)

        self.show_image(temp, title="gamma", cmap='gray')

        # reducing noise
        temp = cv2.GaussianBlur(temp, (7, 7), 0)
        self.show_image(temp, title="gaussian", cmap='gray')

        # applying threshold to emphasize white
        _, temp = cv2.threshold(temp, parameters.lower_threshold, 255,
                                cv2.THRESH_BINARY)
        self.show_image(temp, title="threshold", cmap='gray')

        # applying canny to get edges
        temp = cv2.Canny(temp, 5, 100)
        self.show_image(temp, title="Canny", cmap='gray')

        # mask the frame to keep only the relevant edges
        return self.mask_frame(temp, parameters.vertices)


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

    def output_result(self, out_path, frame_size, frames):
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'DIVX'), WANTED_FPS, frame_size)

        for frame in frames:
            out.write(frame)

        out.release()

    def show_image(self, img, title, cmap=None):
        if not SHOW:
            return
        
        plt.figure(figsize=FIGSIZE)
        plt.imshow(img, cmap=cmap)
        plt.title(title)
        plt.show()

    def run(self, video_path, out_path, parameters):
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("Could not load video file")

        # Get the video data
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if SHOW:
            print(f"{frame_count=}, {fps=}, {frame_width=}, {frame_height=}")

        counter_legal_lane_change = 0
        is_turning = False
        turning_direction = None
        turning_counter = 0
        turning = False
        frames = []

        for frame_num in range(0, frame_count, fps // WANTED_FPS):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()

            if not ret:
                print("Could not load the frame")
                break

            res = frame.copy()
            image = frame.copy()
            debug = frame.copy()  # TODO need to remove
            lines = []

            if not turning:
                mask = self.preprocess_frame(image, parameters)
                # self.show_image(temp, title="mask", cmap='gray')

                hough_parameters = parameters.hough_parameters
                lines = cv2.HoughLinesP(mask, rho=hough_parameters.rho, theta=hough_parameters.theta, threshold=hough_parameters.threshold, 
                                        minLineLength=hough_parameters.min_line_length, maxLineGap=hough_parameters.max_line_gap)

                if SHOW:
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        cv2.line(debug, (x1, y1), (x2, y2), (255, 0, 0), 2)

                    self.show_image(debug, title="lines")

                left_lines, right_lines = self.divide_lines(lines, slope_threshold=LINE_SLOPE)
                lines, change, direction = self.get_line_and_detect_change(left_lines, right_lines)

                turning, is_turning, turning_direction, counter_legal_lane_change, turning_counter = self.check_if_turning(
                    change, direction, is_turning, turning_direction, counter_legal_lane_change)

            if turning:
                res = self.add_text_overlay(res, turning_direction, 4)
                turning_counter -= 1
                if turning_counter == 0:
                    turning = False
            else:
                res = self.draw_lines(res, lines, parameters)
            
            self.show_image(res, title=frame_num)

            frames.append(res)

        self.output_result(out_path, (frame_width, frame_height), frames)

        # Release the video capture object
        cap.release()
