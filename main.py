import cv2
import numpy as np
import time


def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated_image


def calculate_rotation_angle(frame1, frame2):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    detector = cv2.SIFT_create()
    kp1, des1 = detector.detectAndCompute(gray1, None)
    kp2, des2 = detector.detectAndCompute(gray2, None)
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    matches = matcher.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    if len(good_matches) > 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        transformation, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
        angle = np.arctan2(transformation[1, 0], transformation[0, 0]) * 180 / np.pi
        return angle
    else:
        print("Not enough matches found")
        return None


def autorotator(image1, image2):
    angle = calculate_rotation_angle(image1, image2)
    return rotate_image(image2, angle)


def cancel_rotation(video_name, output_name):
    video_capture = cv2.VideoCapture(video_name)
    _, image1 = video_capture.read()
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    output_video = cv2.VideoWriter(output_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    while True:
        ret, image2 = video_capture.read()
        if not ret:
            break

        rotated_image = autorotator(image1, image2)
        output_video.write(rotated_image)
    video_capture.release()
    output_video.release()
    cv2.destroyAllWindows()


def stabilize_video(input_video, output_video, smoothing_window=5):
    cap = cv2.VideoCapture(input_video)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

    _, prev_frame = cap.read()

    smoothed_angles = [0] * smoothing_window

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        angle = calculate_rotation_angle(prev_frame, frame)
        smoothed_angles.append(angle)
        smoothed_angles.pop(0)
        smoothed_angle = sum(smoothed_angles) / smoothing_window
        rows, cols = frame.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), smoothed_angle, 1)
        rotated_frame = cv2.warpAffine(frame, rotation_matrix, (cols, rows))
        out.write(rotated_frame)
        prev_frame = frame

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def count_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    return num_frames


input_vid = 'video5429499283948714472.mp4'
output1 = 'output1.mp4'
output2 = 'output_final.mp4'
num_frames = count_frames(input_vid)
start_time = time.time()
cancel_rotation(input_vid, output1)
stabilize_video(output1, output2, 15)
end_time = time.time()
print("Общее количество кадров в видео:", num_frames)
print("Время выполнения функции:", end_time - start_time, "секунд")
print("Количество обрабатываемых кадров в секунду:", num_frames / (end_time - start_time))

