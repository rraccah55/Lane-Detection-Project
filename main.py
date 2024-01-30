import numpy as np
from matplotlib import pyplot as plt
import cv2

# hyperparameters
figsize = (10, 10)

# TODO find right parameter
mask_point1 = (500,680)
mask_point2 = (900,980)


def mask_frame(img, vertices):
    mask = np.zeros_like(img)
    cv2.rectangle(mask, vertices[0], vertices[1], 255, -1)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img


def draw_lines(img, lines, color=(0, 255, 0), thickness=3):
    for line in lines:
        for x1, y1, x2, y2 in line:
            img = cv2.line(img, (x1, y1), (x2, y2), color, thickness)
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
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # brg to gray

    _, frame = cv2.threshold(frame, 150, 255, cv2.THRESH_BINARY)  # applying threshold to emphasize white

    frame = cv2.GaussianBlur(frame, (7, 7), 0)  # reducing noise

    frame = cv2.Canny(frame, 50, 170)  # applying canny to get edges

    frame = mask_frame(frame, [mask_point1,mask_point2])

    return frame


if __name__ == "__main__":
    video_path = 'input.mp4'
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Could not load video file")

    # Define the start and end time for the 20-second segment for initial debug
    start_time = 0
    end_time = start_time + 1  # seconds

    # Set frame rate and calculate the frames to capture
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"{fps=} {frame_width=} {frame_height=}")

    # Go over the different segments
    frames = []
    for frame_num in range(start_frame, end_frame, int(fps)): # TODO make sure which "jump" is the best one
        ret, frame = cap.read()
        if not ret:
            print("Could not load the frame")
            break

        res = frame.copy()
        image = frame.copy()
        image = preprocess_frame(image)
        # TODO need to find good parameters
        res = cv2.HoughLinesP(image, rho=1, theta=np.pi / 180, threshold=30, minLineLength=50, maxLineGap=10)

        right_lines, left_lines = filter_lines(res, slope_threshold=(0.5, 2))

        best_lines = []
        if len(right_lines) != 0:
            best_lines.append(right_lines.mean(axis=1))
        if len(left_lines) != 0:
            best_lines.append(right_lines.mean(axis=1))

        # TODO there is a chance that because we found the points with the masked image, we need to scale them back to
        #  the original image
        res = draw_lines(res, best_lines)

        #TODO add function to detect lane change
        frames.append(res)
    #TODO need to reconstruct video from frames

    # Release the video capture object
    cap.release()
