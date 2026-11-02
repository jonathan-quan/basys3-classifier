"""Opens the webcam the way the recording view does, reads ~3 s of frames, prints the mode.
No image is saved."""
import time

import cv2


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    frames, shape, start = 0, None, time.perf_counter()
    while time.perf_counter() - start < 3:
        ok, frame = cap.read()
        if ok:
            frames += 1
            shape = frame.shape
    cap.release()
    print('CAMERA', 'opened' if frames else 'NO FRAMES', 'shape', shape, 'fps', round(frames / 3, 1), flush=True)


if __name__ == '__main__':
    main()
