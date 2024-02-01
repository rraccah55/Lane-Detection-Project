import numpy as np
from matplotlib import pyplot as plt
import cv2

# hyperparameters
figsize = (10, 10)

# TODO find right parameter
# mask_point_upper_left = np.array([800,600], np.int32)
# mask_point_upper_right = np.array([1200, 600], np.int32)
# mask_point_lower_left = np.array([800,1920], np.int32)
# mask_point_lower_right = np.array([1200, 1920], np.int32)

vertices = np.array([[900,600], [1300, 600], [1500, 1080], [600, 1080]], np.int32)
vertices = vertices.reshape((-1, 1, 2))

def mask_frame(img, vertices):
    mask = np.zeros_like(img)
    
    mask = cv2.fillPoly(mask, [vertices], (255, 255, 255) , 0)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img


def draw_lines(img, lines, color=(0, 255, 0), thickness=3):
    for line in lines:
        x1, y1, x2, y2 = line[0]
        img = cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    return img


def calculate_slope(x1, x2, y1, y2):
    return 0 if (x1 == x2) else (y1 - y2) / (x1 - x2)


def filter_lines(lines, slope_threshold=(0.5, 2)):
    left_lines = []
    right_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        slope = (y2 - y1) / (x2 - x1 + 1e-6)  # Avoid division by zero
        if slope_threshold[0] < abs(slope) < slope_threshold[1]:
            if slope > 0:
                right_lines.append(line)
            else:
                left_lines.append(line)
    return np.array(right_lines), np.array(left_lines)

def preprocess_frame(frame):
    temp = frame.copy()
    temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)  # brg to gray

    _, temp = cv2.threshold(temp, 150, 255, cv2.THRESH_BINARY)  # applying threshold to emphasize white

    temp = cv2.GaussianBlur(temp, (7, 7), 0)  # reducing noise

    temp = cv2.Canny(temp, 5, 100)  # applying canny to get edges

    temp = mask_frame(temp, vertices)

    return temp


if __name__ == "__main__":
    video_path = 'video.mp4'
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Could not load video file")

    # Define the start and end time for the 20-second segment for initial debug
    wanted_fps = 30

    # Set frame rate and calculate the frames to capture
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"{fps=} {frame_width=} {frame_height=}")

    # Go over the different segments
    frames = []
    for frame_num in range(0, frame_count, (fps // wanted_fps)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            print("Could not load the frame")
            break

        res = frame.copy()
        restTemp = frame.copy()
        image = frame.copy()
        image = preprocess_frame(image)
        # TODO need to find good parameters
        lines = cv2.HoughLinesP(image, rho=1, theta=np.pi / 180, threshold=10, minLineLength=50, maxLineGap=1000)

        right_lines, left_lines = filter_lines(lines, slope_threshold=(0.5, 2))

        best_lines = []
        if len(right_lines) != 0:
            best_lines.append(right_lines.mean(axis=1))
        if len(left_lines) != 0:
            best_lines.append(left_lines.mean(axis=1))

        restTemp = draw_lines(restTemp, lines)
        res = draw_lines(res, best_lines)

        #TODO add function to detect lane change
        frames.append(res)
    
    out = cv2.VideoWriter('temp.avi',cv2.VideoWriter_fourcc(*'DIVX'), wanted_fps, (frame_width, frame_height))
    
    for frame in frames:
        out.write(frame)
    
    out.release()

    # Release the video capture object
    cap.release()
