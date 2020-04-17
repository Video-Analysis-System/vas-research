"""
This script tries to detect occlusion by using the Edge background model 
described in the "Automatic_Control_of_Video_Surveillance_Camera_Sab.pdf" 
paper.

python defocus edges.py --input ../clips/Defocusing.mp4 --scale 0.5 -t 0.5
"""

import numpy as np
import cv2
import argparse
import time


class TrackLastNItems:
    def __init__(self, max_size):
        self.max_size = max_size
        self.queue = []

    def update(self, item):
        self.queue.append(item)
        if len(self.queue) > self.max_size:
            self.queue.pop(0)

    def get_first(self):
        return self.queue[0]


class MovinDetection:
    def __init__(self, max_size):
        self.P = 0
        self.tracker = TrackLastNItems(max_size=max_size)

    def update(self, bg):
        self.tracker.update(np.where(bg > 0, 1, 0))
        if len(self.tracker.queue) == self.tracker.max_size:
            self.P = np.count_nonzero(
                (self.tracker.queue[0] - self.tracker.queue[-1]))
    
    def get_P(self):
        return self.P


class MaxTracking:
    def __init__(self):
        self.max = 0

    def update(self, value):
        if self.max < value:
            self.max = value


class BackgroundModel:
    def __init__(self, M=30, alpha=0.3):
        self.alpha = alpha
        self.M = M
        self.M_recent = list()
        self.Pn = None

    def update(self, new_bg):
        # Assume new_bg is return of edge detection algorithm
        # having 255 for edges and 0 for non-edges
        bin_edges = new_bg.copy()
        bin_edges[np.nonzero(bin_edges)] = 1

        self.M_recent.append(bin_edges)
        if len(self.M_recent) > self.M:
            self.M_recent.pop(0)

        if self.Pn is None:
            self.Pn = np.zeros_like(bin_edges)

        self.B = np.zeros_like(bin_edges)
        for i in self.M_recent:
            self.B += i

        self.Pn = self.alpha * self.Pn + (1 - self.alpha) * self.B
        return self.Pn

    def get_bg_model(self):
        return np.where(self.Pn >= (self.M / 2), 255, 0).astype(np.uint8)


def detect_edge(frame):
    if len(frame.shape) == 3:
        tmp = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        tmp = frame.copy()
    edges = cv2.Canny(tmp, 100, 200)
    return edges


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str,
                        help='Path to a video or a sequence of image.', default='vtest.avi')
    parser.add_argument('-s', '--scale', type=float,
                        help='Scale down frame', default=1.0)
    parser.add_argument('--threshold', type=float,
                        help='Decision threshold', default=0.5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    capture = cv2.VideoCapture(cv2.samples.findFileOrKeep(args.input))

    if not capture.isOpened:
        print('Unable to open: ' + args.input)
        exit(0)

    bg_model = BackgroundModel()
    maxim = MaxTracking()
    adap2 = MaxTracking()
    threshold = args.threshold

    md = MovinDetection(max_size=20)

    while True:
        ret, frame = capture.read()
        if frame is None:
            break

        if args.scale != 1.0:
            frame = cv2.resize(frame, None, fx=args.scale, fy=args.scale)

        current_frame_number = capture.get(cv2.CAP_PROP_POS_FRAMES)

        edges = detect_edge(frame)
        if current_frame_number > 1:
            # num_edges = np.count_nonzero(edges)
            num_bg = np.count_nonzero(bg_model.get_bg_model())
            # print(current_frame_number, num_edges/(num_bg+1))
            maxim.update(num_bg)
            if num_bg / (maxim.max + 1) < threshold and num_bg > 0:
                cv2.putText(frame,
                            "DEFOCUSING DETECTED", (5, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 255), 2)

        if current_frame_number > 1:
            md.update(bg_model.get_bg_model())
            adap2.update(np.count_nonzero(bg_model.get_bg_model()))

        # Debug ratio
        if current_frame_number > 30:
            print("{:.4f}".format(md.get_P() / adap2.max))

        bg_model.update(edges)
        bg = bg_model.get_bg_model()

        cv2.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
        cv2.putText(frame, str(current_frame_number), (15, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        cv2.imshow('Frame', frame)
        cv2.imshow('Edge', edges)
        cv2.imshow('bg', bg)

        keyboard = cv2.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break
        if keyboard == ord('s'):
            time.sleep(10)
