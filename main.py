import numpy as np
from matplotlib import pyplot as plt
import cv2

# hyperparameters
FIGSIZE = (20, 20)
UPPER_BOUND = 700
LOWER_BOUND = 1080
FRAMES_UNTIL_TURNING = 10
WANTED_FPS = 30
VERTICES_LEFT = np.array([[450, 1070], [900,620], [1000, 620], [1000, 1070]], np.int32).reshape((-1, 1, 2))
VERTICES_RIGHT = np.array([[1000, 650], [1250, 650],[1700, 1070], [1000, 1070]], np.int32).reshape((-1, 1, 2))
NUM_OF_FRAME_FOR_LANE_CHANGE = 60

show = False
def add_text_overlay(frame, text, font_size=1.0):
    # Choose font and position
    font = cv2.FONT_HERSHEY_SIMPLEX
    position = (200, 600)

    # Add text to the frame
    cv2.putText(frame, text, position, font, font_size, (255, 0, 0), 6, cv2.LINE_AA)
    return frame
def resize_line(line):
    x1, y1, x2, y2 = line
    slope = calculate_slope(x1, x2, y1, y2)
    intercept = y1 - slope * x1
    y1 = LOWER_BOUND
    y2 = UPPER_BOUND
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return x1, y1, x2, y2
def mask_frame(img,vertices):
    mask = np.zeros_like(img)
    mask = cv2.fillPoly(mask, [vertices], (255, 255, 255) , 0)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img
def draw_lines(img, lines, color=(0, 255, 0), thickness=6):
    for line in lines:
        x1, y1, x2, y2 = resize_line(line)
        img = cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    return img
def calculate_slope(x1, x2, y1, y2):
    return 0 if (x1 == x2) else (y1 - y2) / (x1 - x2)
def filter_lines(lines, slope_threshold=(0.5, 2)):
    result = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = calculate_slope(x1, x2, y1, y2)
            if slope_threshold[0] < abs(slope) < slope_threshold[1]:
                result.append(line[0])

    return np.array(result)
def preprocess_frame(frame):
    temp = frame.copy()

    temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)  # brg to gray

    temp = cv2.GaussianBlur(temp, (7, 7), 0)  # reducing noise

    _, temp = cv2.threshold(temp, 150, 255, cv2.THRESH_BINARY)  # applying threshold to emphasize white

    temp = cv2.Canny(temp, 5, 100)  # applying canny to get edges

    return mask_frame(temp,VERTICES_LEFT), mask_frame(temp,VERTICES_RIGHT)

#TODO maybe change the way we detect line change
def get_line_and_detect_change(left_lines, right_lines):
    best_lines = []

    if left_lines.size == 0:
        return left_lines, True, "Changing to left lane"
    elif right_lines.size == 0:
        return best_lines, True, "Changing to right lane"
    else:
        right_line = np.mean(right_lines, axis=0)
        left_line = np.mean(left_lines, axis=0)
        best_lines.append(right_line)
        best_lines.append(left_line)
        return best_lines, False, ""
def check_if_turning(change,direction, is_turning, turning_direction, counter_legal_lane_change):
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

if __name__ == "__main__":
    video_path = 'video.mp4'
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Could not load video file")

    # Set frame rate and calculate the frames to capture
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"{frame_count=}, {fps=}, {frame_width=}, {frame_height=}")

    counter_legal_lane_change = 0
    is_turning = False
    turning_direction = None
    turning_counter = 0
    turning = False

    # Go over the different segments
    frames = []
    for frame_num in range(0, frame_count):
        cap.set(cv2.CAP_PROP_POS_FRAMES,frame_num)
        ret, frame = cap.read()
        if not ret:
            print("Could not load the frame")
            break

        res = frame.copy()
        image = frame.copy()
        lines = []

        if not turning:
            left_mask, right_mask = preprocess_frame(image)

            left_lines = cv2.HoughLinesP(left_mask, rho=1, theta=np.pi / 180, threshold=10, minLineLength=50, maxLineGap=100)
            right_lines = cv2.HoughLinesP(right_mask, rho=1, theta=np.pi / 180, threshold=10, minLineLength=50, maxLineGap=1000)

            left_lines = filter_lines(left_lines, slope_threshold=(0.5, 2))
            right_lines = filter_lines(right_lines, slope_threshold=(0.5, 2))

            res_right = res.copy()
            res_left = res.copy()

            res_right = draw_lines(res_right, right_lines)
            res_left = draw_lines(res_left, left_lines)

            lines, change, direction = get_line_and_detect_change(left_lines, right_lines)

            turning, is_turning, turning_direction, counter_legal_lane_change, turning_counter = check_if_turning(change, direction, is_turning, turning_direction, counter_legal_lane_change)

        if turning:
            res = add_text_overlay(res, turning_direction, 4)
            turning_counter -= 1
            if turning_counter == 0:
                turning = False
        else:
            res = draw_lines(res, lines)

        frames.append(res)
    
    out = cv2.VideoWriter('temp.avi',cv2.VideoWriter_fourcc(*'DIVX'), WANTED_FPS, (frame_width, frame_height))
    
    for frame in frames:
        out.write(frame)
    
    out.release()

    # Release the video capture object
    cap.release()
