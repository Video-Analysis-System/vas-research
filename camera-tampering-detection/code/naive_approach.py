import sys
import time

import numpy as np
import cv2


cap = cv2.VideoCapture(sys.argv[1])
fgbg = cv2.createBackgroundSubtractorMOG2()
ret, frame = cap.read()
fgmask = fgbg.apply(frame)
kernel = np.ones((5, 5), np.uint8)
while(True):
    ret, frame = cap.read()
    if(frame is None):
        print("End of frame")
        break
    else:
        frame = cv2.resize(frame, None, fx=0.4, fy=0.4)
        a = 0
        bounding_rect = []
        fgmask = fgbg.apply(frame)
        fgmask = cv2.erode(fgmask, kernel, iterations=4)
        fgmask = cv2.dilate(fgmask, kernel, iterations=4)
        
        cv2.imshow("bg", fgbg.getBackgroundImage())

        cv2.putText(frame,
                    str(cap.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
        cv2.imshow('frame', frame)

        contours, _ = cv2.findContours(fgmask,
                                       cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for i in range(0, len(contours)):
            bounding_rect.append(cv2.boundingRect(contours[i]))

        for i in range(0, len(contours)):
            x, y, w, h = bounding_rect[i]
            if w >= 40 or h >= 40:
                a += (w * h)

            if a >= int(frame.shape[0]) * int(frame.shape[1]) / 3:
                cv2.putText(frame,
                            "TAMPERING DETECTED", (5, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 255), 2)

            cv2.imshow('frame', frame)

        cv2.imshow('erode', fgmask)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        time.sleep(10)

    # time.sleep(0.05)

cap.release()
cv2.destroyAllWindows()
