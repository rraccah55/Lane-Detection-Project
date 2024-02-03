import numpy as np
from matplotlib import pyplot as plt
import cv2

# hyperparameters
figsize = (20, 20)
show = False
upper_bound = 700
lower_bound = 1080

vertices_left = np.array([[450, 1080], [900,620], [1000, 620], [1000, 1080]], np.int32)
vertices_left = vertices_left.reshape((-1, 1, 2))

vertices_right = np.array([[1000, 620], [1300, 620],[1700, 1080], [1000, 1080]], np.int32)
vertices_right = vertices_right.reshape((-1, 1, 2))

is_turning = False
frames_until_turning = 5
turning_direction = None

def add_text_overlay(frame, text, font_size=1.0):
    # Choose font and position
    font = cv2.FONT_HERSHEY_SIMPLEX
    position = (200, 600)

    # Add text to the frame
    cv2.putText(frame, text, position, font, font_size, (255, 0, 0), 6, cv2.LINE_AA)

def resize_line(line):
    x1, y1, x2, y2 = line
    slope = calculate_slope(x1, x2, y1, y2)
    intercept = y1 - slope * x1
    y1 = lower_bound
    y2 = upper_bound
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
    if show:
        plt.figure(figsize=figsize)
        plt.imshow(temp, cmap="gray")
        plt.title("noise")
        plt.show()
    _, temp = cv2.threshold(temp, 150, 255, cv2.THRESH_BINARY)  # applying threshold to emphasize white
    if show:
        plt.figure(figsize=figsize)
        plt.imshow(temp, cmap="gray")
        plt.title("color")
        plt.show()

    temp = cv2.Canny(temp, 5, 100)  # applying canny to get edges
    if show:
        plt.figure(figsize=figsize)
        plt.imshow(temp, cmap="gray")
        plt.title("canny")
        plt.show()

    return mask_frame(temp,vertices_left), mask_frame(temp,vertices_right)

def get_line_and_detect_change(left_lines, right_lines):
    best_lines = []

    if right_lines.size == 0:
        return best_lines, True, "Changing to right lane"
    elif left_lines.size == 0:
        return left_lines, True, "Changing to left lane"
    else:
        right_line = np.mean(right_lines, axis=0)
        left_line = np.mean(left_lines, axis=0)
        best_lines.append(right_line)
        best_lines.append(left_line)
        return best_lines, False, ""

if __name__ == "__main__":
    video_path = 'video.mp4'
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Could not load video file")

    # # Set frame rate and calculate the frames to capture
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"{frame_count} frames")
    print(f"{fps} fps")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    wanted_fps = 1

    turning_counter = 0
    # Go over the different segments
    frames = []
    # for frame_num in range(0, 1):#frame_count):
    for frame_num in range(0, frame_count, wanted_fps):
        # cap.set(cv2.CAP_PROP_POS_FRAMES,187)#frame_num)
        cap.set(cv2.CAP_PROP_POS_FRAMES,frame_num)
        ret, frame = cap.read()
        # if not ret:
        #     print("Could not load the frame")
        #     break

        res = frame.copy()
        restTemp = frame.copy()
        image = frame.copy()
        left_mask, right_mask = preprocess_frame(image)
        if show:
            plt.figure(figsize=figsize)
            plt.imshow(res)
            plt.title("original")
            plt.show()
            plt.figure(figsize=figsize)
            plt.imshow(left_mask, cmap="gray")
            plt.title("left mask")
            plt.show()

            plt.figure(figsize=figsize)
            plt.imshow(right_mask, cmap="gray")
            plt.title("right mask")
            plt.show()

        left_lines = cv2.HoughLinesP(left_mask, rho=1, theta=np.pi / 180, threshold=10, minLineLength=50, maxLineGap=1000)
        right_lines = cv2.HoughLinesP(right_mask, rho=1, theta=np.pi / 180, threshold=10, minLineLength=50, maxLineGap=1000)

        left_lines = filter_lines(left_lines, slope_threshold=(0.5, 2))
        right_lines = filter_lines(right_lines, slope_threshold=(0.5, 2))

        lines, change, direction = get_line_and_detect_change(left_lines, right_lines)

        if change:
            if not is_turning:
                if turning_counter == 0:
                    turning_direction = direction
                    turning_counter+= 1
                    res = draw_lines(res, lines)
                elif turning_counter < frames_until_turning:
                    if not turning_direction == direction:
                        turning_counter = 0
                        turning_direction = direction
                    else:
                        turning_counter+= 1
                    
                    res = draw_lines(res, lines)
                else:
                    print(f"{frame_num=} {change=} {turning_direction=}")
                    # plt.figure(figsize=figsize)
                    # plt.imshow(res)
                    # plt.show()
                    add_text_overlay(res,turning_direction,5)
                    is_turning = True
                    turning_counter = 0
            else:
                print(f"{frame_num=} {change=} {turning_direction=}")
                # plt.figure(figsize=figsize)
                # plt.imshow(res)
                # plt.show()
                add_text_overlay(res,turning_direction,5)
        else:
            if is_turning:
                if turning_counter < frames_until_turning:
                    turning_counter+= 1
                else:
                    is_turning = False
                    turning_counter = 0
                    res = draw_lines(res, lines)
            else:
                res = draw_lines(res, lines)


        # if change == True and turning_counter < frames_until_turning:
        #     frames_until_turning+=1
        # elif change == True:
        #     print(f"{frame_num=} {change=} {direction=}")
        #     # plt.figure(figsize=figsize)
        #     # plt.imshow(res)
        #     # plt.show()
        #     add_text_overlay(res,direction,5)
        # else:
        #     res = draw_lines(res, lines)
            
        
        if show:
            plt.figure(figsize=figsize)
            plt.imshow(res)
            plt.show()

        frames.append(res)
    
    out = cv2.VideoWriter('temp.avi',cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_width, frame_height))
    
    for frame in frames:
        out.write(frame)
    
    out.release()

    # Release the video capture object
    cap.release()
