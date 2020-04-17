"""
This script tries to detect occlusion by comparing histogram
of current frame and background model using BHATTACHARYYA distance

python occlusion.py --input ../clips/Occlusion.mp4 --algo MOG2 --scale 0.5 --threshold 0.4
"""

from __future__ import print_function
import cv2
import numpy as np
import time
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                              Opencv2. You can process both videos and images.')
    parser.add_argument('--input', type=str,
                        help='Path to a video or a sequence of image.', default='vtest.avi')
    parser.add_argument('--algo', type=str,
                        help='Background subtraction method (KNN, MOG2).', default='MOG2')
    parser.add_argument('--scale', type=float,
                        help='Scale down frame', default=1.0)
    parser.add_argument('--threshold', type=float,
                        help='Histogram comparison threshold', default=0.4)

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

    while True:
        ret, frame = capture.read()
        if frame is None:
            break
        
        if args.scale != 1.0:
            frame = cv2.resize(frame, None, fx=args.scale, fy=args.scale)
        
        # # Test add occlusion
        # if capture.get(cv2.CAP_PROP_POS_FRAMES) >= 150:
        #     cv2.rectangle(frame, (0, 0), (frame.shape[1]//2, frame.shape[0]//2), color=(128, 128, 135), thickness=-1)

        # TO gray
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # fgMask = backSub.apply(frame, learningRate=0.002)
        fgMask = backSub.apply(frame)

        cv2.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
        cv2.putTexHISTCMP_BHATTACHARYYAt(frame, str(capture.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        # if capture.get(cv2.CAP_PROP_POS_FRAMES) == 10:
        #     cv2.imwrite("./no_tamp.jpg", frame)
        #     cv2.imwrite("./bg.jpg", backSub.getBackgroundImage())

        # if capture.get(cv2.CAP_PROP_POS_FRAMES) == 85:
        #     cv2.imwrite("./half_tamp.jpg", frame)
        #     cv2.imwrite("./half_fgmask.jpg", fgMask)
        #     cv2.imwrite("./half_bg.jpg", backSub.getBackgroundImage())

        # if capture.get(cv2.CAP_PROP_POS_FRAMES) == 121:
        #     cv2.imwrite("./full_tamp.jpg", frame)
        #     cv2.imwrite("./full_fgmask.jpg", fgMask)
        #     cv2.imwrite("./full_bg.jpg", backSub.getBackgroundImage())

        current_hist = cv2.calcHist([frame], [0], None, [256], [0, 256])
        bg_hist = cv2.calcHist(
            [backSub.getBackgroundImage()], [0], None, [256], [0, 256])
        distance = cv2.compareHist(current_hist, bg_hist,
                                method=cv2.HISTCMP_BHATTACHARYYA)
        
        # # Debug distance
        # if capture.get(cv2.CAP_PROP_POS_FRAMES) >= 10:
        #     print("distance: ", distance)

        if distance > threshold:
            count += 1

        if count >= 5:
            cv2.putText(frame,
                        "TAMPERING DETECTED", (5, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 255), 2)

        cv2.imshow('Frame', frame)
        cv2.imshow('FG Mask', fgMask)
        cv2.imshow('BG Image', backSub.getBackgroundImage())

        keyboard = cv2.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break
        if keyboard == 's':
            time.sleep(10)
