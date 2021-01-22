import numpy as np
import cv2
import argparse
import time
from sklearn.mixture import GaussianMixture as GMM
import time

"""
def detect_edge(frame):
    if len(frame.shape) == 3:
        tmp = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        tmp = frame.copy()
    tmp1 = np.uint8(tmp)
    edges = cv2.Canny(tmp1, 25, 75)
    return edges
"""
def detect_edge(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3, scale=1)
    y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3, scale=1)
    absx = cv2.convertScaleAbs(x)
    absy = cv2.convertScaleAbs(y)
    edge = cv2.addWeighted(absx, 0.5, absy, 0.5, 0)
    return edge

def ECR(frame, prev_frame, width, height, crop=True, dilate_rate = 5):
    safe_div = lambda x,y: 0 if y == 0 else x / y
    if crop:
        startY = int(height * 0.3)
        endY = int(height * 0.8)
        startX = int(width * 0.3)
        endX = int(width * 0.8)
        frame = frame[startY:endY, startX:endX]
        prev_frame = prev_frame[startY:endY, startX:endX]

    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_image2 = np.uint8(gray_image)
    edge = cv2.Canny(gray_image2, 0, 200)
    dilated = cv2.dilate(edge, np.ones((dilate_rate, dilate_rate)))
    inverted = (255 - dilated)
    #gray_image2 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray_image3 = np.uint8(prev_frame)
    edge2 = cv2.Canny(gray_image3, 0, 200)
    dilated2 = cv2.dilate(edge2, np.ones((dilate_rate, dilate_rate)))
    inverted2 = (255 - dilated2)
    log_and1 = (edge2 & inverted)
    log_and2 = (edge & inverted2)
    pixels_sum_new = np.sum(edge)
    pixels_sum_old = np.sum(edge2)
    out_pixels = np.sum(log_and1)
    in_pixels = np.sum(log_and2)
    r = float(max(safe_div(float(in_pixels), float(pixels_sum_new)), safe_div(float(out_pixels), float(pixels_sum_old))))
    return r

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str,
                        help='Path to a video or a sequence of image.',
                        default='/home/vinh/big-drive/hanh/Camera/test/test1.avi')
    parser.add_argument('-s', '--scale', type=float,
                        help='Scale down frame', default=1)
    parser.add_argument('--threshold', type=float,
                        help='Decision threshold', default=0.4)
    parser.add_argument('--k', type=float,
                        help='Decision k', default=0.05)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    cap = cv2.VideoCapture(args.input)
    fgbg = cv2.createBackgroundSubtractorMOG2()
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.set(3, width)
    cap.set(4, height)
    cnt = 0
    ECR1 = 0
    tECR = 0
    AECR = 0
    threshold = args.threshold
    k = args.k
    while True:
        ret, frame = cap.read()
        if frame is None:
            break

        t0 = time.time()

        current_frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
        fgmask = fgbg.apply(frame)
        frame1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        background = np.absolute(frame1 - fgmask)

        ERC1 = ECR(frame, background, width, height)
        print(ERC1)
        if current_frame_number > 1:
            tECR += ECR(frame, background, width, height)
        else:
            tECR = ECR(frame, background, width, height)
        #print('tecr', tECR)
        AECR = tECR / current_frame_number
        #print('number frame', current_frame_number)
        print('aecr', AECR)

        if ECR(frame, background, width, height) > threshold:
            cnt = cnt + 1
        if abs(AECR - ECR(frame, background, width, height)) < k:
            cnt = 0
        if cnt >= 10:
            print('.............')
            cv2.putText(frame,
                       "TAMPERING DETECTED", (5, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1,
                       (0, 255, 255), 2)
        cv2.imshow('Frame', frame)
        cv2.imshow('bg', background)
        #print("time:", time.time()-t0)
        keyboard = cv2.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break



