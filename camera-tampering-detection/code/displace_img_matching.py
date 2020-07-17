import numpy as np
import cv2 as cv
# import matplotlib.pyplot as plt

cap = cv.VideoCapture('/home/vinh/big-drive/hanh/Camera/test/test3.avi')
fgbg = cv.createBackgroundSubtractorMOG2()
img1 = None
img = None
img2 = None
count = 0
while True:
    img = img2
    ret, img2 = cap.read()
    if img2 is None:
        break
    #img2 = cap.get(cv.CAP_PROP_POS_FRAMES)
    #img2 = fgbg.apply(img2)

    if img1 is None:
        img1 = img2
    else:
        img1 = img

    orb = cv.ORB_create()
    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # create BFMatcher object
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1, des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)

    list_kp1 = []
    list_kp2 = []
    x = 0
    y = 0
    for mat in matches[0:10]:
        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        # Get the coordinates
        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt

        # Append to each list
        list_kp1.append((x1, y1))
        list_kp2.append((x2, y2))

        #print('x1', x1)

        cv.line(img2, (round(x2), round(y2)), (round(x2+1), round(y2+1)), (0,0,255), 5)

        x = x + abs(x1 - x2)
        y = y + abs(y1 - y2)
    print('x', x)
    print('y', y)
    if x > 100 or y > 100:
        count += 1
    if count >= 10:
        cv.putText(img2,
                   "TAMPERING DETECTED", (5, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 1,
                   (0, 255, 255), 2)
    cv.imshow('Frame', img2)
    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
