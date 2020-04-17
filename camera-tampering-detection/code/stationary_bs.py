"""
This script tries to localize stationary object using dual foreground segmentation
"""


from __future__ import print_function
import cv2
import argparse
import imutils
import numpy as np

from utils import get_bounding_boxes, draw_bboxes, adjust_gamma


def parse_args():
    parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                              OpenCV. You can process both videos and images.')
    parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='vtest.avi')
    parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
    parser.add_argument('--history', type=int, help='Duration before foreground absorbed.', default='1')
    parser.add_argument('--shadow', help='Whether detect shadow.', default=False, action='store_true')
    return parser.parse_args()


def apply_dilation(image, kernel_size, kernel_type):
    # Apply dilation to image with the specified kernel type and image
    u_image = image.astype(np.uint8)
    kernel = cv2.getStructuringElement(kernel_type, (kernel_size, kernel_size))
    u_image = cv2.morphologyEx(u_image, cv2.MORPH_DILATE, kernel)
    return u_image


if __name__ == "__main__":
    args = parse_args()

    if args.algo == 'MOG2':
        backSub = cv2.createBackgroundSubtractorMOG2(history=args.history, varThreshold=120, detectShadows=args.shadow)
        backSubLong = cv2.createBackgroundSubtractorMOG2(history=args.history, varThreshold=1000, detectShadows=args.shadow)
    else:
        backSub = cv2.createBackgroundSubtractorKNN(dist2Threshold=400, detectShadows=args.shadow)
        backSubLong = cv2.createBackgroundSubtractorKNN(dist2Threshold=400, detectShadows=args.shadow)

    capture = cv2.VideoCapture(cv2.samples.findFileOrKeep(args.input))
    if not capture.isOpened:
        print('Unable to open: ' + args.input)
        exit(0)
    while True:
        ret, frame = capture.read()
        if frame is None:
            break
        frame = imutils.resize(frame, width=450)
        frame = adjust_gamma(frame, gamma=1.7)
        
        fgMask = backSub.apply(frame, learningRate=0.002)
        fgMaskLong = backSubLong.apply(frame, learningRate=0.0001)

        _, fgMask = cv2.threshold(fgMask, 250, 255, cv2.THRESH_BINARY)
        _, fgMaskLong = cv2.threshold(fgMaskLong, 250, 255, cv2.THRESH_BINARY)

        fgMask = apply_dilation(fgMask, 3, cv2.MORPH_ELLIPSE)
        fgMaskLong = apply_dilation(fgMaskLong, 3, cv2.MORPH_ELLIPSE)

        stationary = np.logical_and(fgMask == 0, fgMaskLong == 255)
        stationary = stationary.astype(np.uint8) * 255

        boxes, _ = get_bounding_boxes(stationary)

        cv2.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
        cv2.putText(frame, str(capture.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
        
        cv2.imshow('Frame', frame)
        cv2.imshow('FG Mask', fgMask)
        cv2.imshow('FG Mask Long', fgMaskLong)
        cv2.imshow('Stationary', stationary)
        cv2.imshow('Drawn', draw_bboxes(frame, boxes))

        keyboard = cv2.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break
