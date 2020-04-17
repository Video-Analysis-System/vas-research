
import cv2
import numpy as np

BBOX_MIN_AREA = 100


def get_bounding_boxes(image):
    # Return Bounding Boxes in the format x,y,w,h where (x,y) is the top left corner
    bbox = []
    contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    tmp = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(tmp, contours, -1, (0, 255, 0), 2)
    cv2.imshow('cnt', tmp)

    for cnt in contours:
        if cv2.contourArea(cnt) > BBOX_MIN_AREA:
            rect = cv2.boundingRect(cnt)
            if rect not in bbox:
                bbox.append(rect)

    return bbox, tmp


def draw_bboxes(image, bboxes):
    drawn = image.copy()
    for rect in bboxes:
        x, y, w, h = rect
        cv2.rectangle(drawn, (x,y), (x+w, y+h), (0, 255, 0), thickness=1)
    return drawn


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)
