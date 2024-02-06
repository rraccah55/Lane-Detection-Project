import numpy as np
from matplotlib import pyplot as plt
import cv2

# hyperparameters
FIGSIZE = (20, 20)
FRAMES_UNTIL_TURNING = 10
WANTED_FPS = 30
NUM_OF_FRAME_FOR_LANE_CHANGE = 60
show = False  # for debug


class Parameters:
    def __init__(self, upper_bound, lower_bound, vertices, is_night, lower_threshold, min_line_length):
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.vertices = np.array(vertices, np.int32).reshape((-1, 1, 2))
        self.is_night = is_night
        self.lower_threshold = lower_threshold
        self.min_line_length = min_line_length


parameters_day = Parameters(700, 1080, [[450, 1050], [900, 620], [1250, 650], [1700, 1050]], False, 150, 40)
parameters_night = Parameters(410, 550, [[380, 380], [180, 530], [600, 380], [750, 530]], True, 115, 10)


def add_text_overlay(frame, text, font_size=1.0):
    # Choose font and position
    font = cv2.FONT_HERSHEY_SIMPLEX
    position = (200, 600)

    # Add text to the frame
    cv2.putText(frame, text, position, font, font_size, (255, 0, 0), 6, cv2.LINE_AA)
    return frame


def resize_line(line, parameters):
    x1, y1, x2, y2 = line
    slope = calculate_slope(x1, x2, y1, y2)
    intercept = y1 - slope * x1
    y1 = parameters.lower_bound
    y2 = parameters.upper_bound
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return x1, y1, x2, y2


def mask_frame(img, vertices):
    mask = np.zeros_like(img)
    mask = cv2.fillPoly(mask, [vertices], (255, 255, 255), 0)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img


def draw_lines(img, lines, parameters, color=(0, 255, 0), thickness=6):
    for line in lines:
        x1, y1, x2, y2 = resize_line(line, parameters)
        img = cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    return img


def calculate_slope(x1, x2, y1, y2):
    return 0 if (x1 == x2) else (y1 - y2) / (x1 - x2)


def filter_lines(lines, slope_threshold=(0.5, 2)):
    right_lines = []
    left_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = calculate_slope(x1, x2, y1, y2)
            if slope_threshold[0] < abs(slope) < slope_threshold[1]:
                if slope > 0:
                    right_lines.append(line[0])
                else:
                    left_lines.append(line[0])

    return np.array(left_lines), np.array(right_lines)


def preprocess_frame(frame, parameters):
    temp = frame.copy()

    temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)  # brg to gray

    # Apply gamma correction
    if parameters.is_night:
        gamma = 0.92
        temp = np.uint8(cv2.pow(temp / 255.0, gamma) * 255)

    if show:
        plt.figure(figsize=FIGSIZE)
        plt.imshow(temp, cmap='gray')
        plt.title("gamma")
        plt.show()

    temp = cv2.GaussianBlur(temp, (7, 7), 0)  # reducing noise

    if show:
        plt.figure(figsize=FIGSIZE)
        plt.imshow(temp, cmap='gray')
        plt.title("gaussian")
        plt.show()

    _, temp = cv2.threshold(temp, parameters.lower_threshold, 255,
                            cv2.THRESH_BINARY)  # applying threshold to emphasize white

    if show:
        plt.figure(figsize=FIGSIZE)
        plt.imshow(temp, cmap='gray')
        plt.title("threshold")
        plt.show()

    temp = cv2.Canny(temp, 5, 100)  # applying canny to get edges

    if show:
        plt.figure(figsize=FIGSIZE)
        plt.imshow(temp, cmap='gray')
        plt.title("Canny")
        plt.show()

    return mask_frame(temp, parameters.vertices)


def get_line_and_detect_change(left_lines, right_lines):
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


def check_if_turning(change, direction, is_turning, turning_direction, counter_legal_lane_change):
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
    video_path = 'day.mp4'
    parameters = parameters_day

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
    frames = []

    # TODO for debug need to remove
    start_time = 16
    end_time = start_time + 10  # seconds

    # Set frame rate and calculate the frames to capture
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    #for frame_num in range(start_frame, end_frame):
    #for frame_num in range(0, 1):
    # TODO for debug need to remove

    for frame_num in range(0, frame_count):

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
            mask = preprocess_frame(image, parameters)

            if show:
                plt.figure(figsize=FIGSIZE)
                plt.imshow(mask, cmap="gray")
                plt.title("mask")
                plt.show()

            lines = cv2.HoughLinesP(mask, rho=1, theta=np.pi / 180, threshold=10, minLineLength=parameters.min_line_length,
                                    maxLineGap=100)

            if show:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(debug, (x1, y1), (x2, y2), (255, 0, 0), 2)

                plt.figure(figsize=FIGSIZE)
                plt.imshow(debug)
                plt.title("lines")
                plt.show()

            left_lines, right_lines = filter_lines(lines, slope_threshold=(0.5, 2))

            lines, change, direction = get_line_and_detect_change(left_lines, right_lines)

            turning, is_turning, turning_direction, counter_legal_lane_change, turning_counter = check_if_turning(
                change, direction, is_turning, turning_direction, counter_legal_lane_change)

        if turning:
            res = add_text_overlay(res, turning_direction, 4)
            turning_counter -= 1
            if turning_counter == 0:
                turning = False
        else:
            res = draw_lines(res, lines, parameters)
        if (show):
            plt.figure(figsize=FIGSIZE)
            plt.imshow(res)
            plt.title(frame_num)
            plt.show()

        frames.append(res)

    out = cv2.VideoWriter('temp.avi', cv2.VideoWriter_fourcc(*'DIVX'), WANTED_FPS, (frame_width, frame_height))

    for frame in frames:
        out.write(frame)

    out.release()

    # Release the video capture object
    cap.release()
