"""
This script tries to detect displacement by TSS block matching 
algorithm described in the Sabotage.pdf paper (Calculate TSS for the 
middle pixel only, which is not robust against moving scene (i.e. 
crowd station))

python displacement_bm.py --input ../clips/Displacement.mp4 --algo MOG2 -s 0.5 -t 4.0 -m 20
"""


from __future__ import print_function
import cv2
import numpy as np
import time
import argparse
import imutils

from blockmatching_util import TSS, AverageMeter


def parse_args():
    parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                                Opencv2. You can process both videos and images.')
    parser.add_argument('--input', type=str,
                        help='Path to a video or a sequence of image.', default='vtest.avi')
    parser.add_argument('--algo', type=str,
                        help='Background subtraction method (KNN, MOG2).', default='MOG2')
    parser.add_argument('-s', '--scale', type=float,
                        help='Scale down frame', default=1.0)
    parser.add_argument('-t', '--threshold', type=float,
                        help='Comparison threshold', default=4.0)
    parser.add_argument('-m', '--max_consecutive', type=int,
                        help='Max consecutive frame to average', default=20)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.algo == 'MOG2':
        backSub = cv2.createBackgroundSubtractorMOG2(
            # history=300,
            # varThreshold=120,
            detectShadows=False
        )
    else:
        backSub = cv2.createBackgroundSubtractorKNN()
    capture = cv2.VideoCapture(cv2.samples.findFileOrKeep(args.input))

    if not capture.isOpened:
        print('Unable to open: ' + args.input)
        exit(0)

    count = 0
    threshold = args.threshold

    avg_displacement = AverageMeter(max_size=args.max_consecutive)

    while True:
        ret, frame = capture.read()
        if frame is None:
            break

        if args.scale != 1.0:
            frame = cv2.resize(frame, None, fx=args.scale, fy=args.scale)

        current_frame_number = capture.get(cv2.CAP_PROP_POS_FRAMES)

        # TO gray
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if current_frame_number > 1:
            bg_prev = backSub.getBackgroundImage()

        fgMask = backSub.apply(frame)

        cv2.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
        cv2.putText(frame, str(current_frame_number), (15, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        if current_frame_number > 1:
            displacement = TSS(
                frame, bg_prev, block_size=16, init_step=4)
            avg_displacement.update(displacement)
            print("Displacement: ", displacement)

        if avg_displacement.average() > threshold:
            cv2.putText(frame,
                        "DISPLACEMENT DETECTED", (5, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 255), 2)

        cv2.imshow('Frame', frame)
        cv2.imshow('FG Mask', fgMask)
        cv2.imshow('BG Image', backSub.getBackgroundImage())

        keyboard = cv2.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break
        if keyboard == ord('s'):
            time.sleep(10)
