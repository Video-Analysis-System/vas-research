"""
Utilities for block matching task
"""

import numpy as np
import cv2


def MSD(b1, b2):
    """
    Helper cost function for the Three Step Search algorithm
    @param b1: First block
    @param b2: Second block

    @return cost (float)
    """

    assert len(b1.shape) == 2
    h, w = b1.shape
    return np.sum((b1 - b2) ** 2) / (h*w)


def TSS(frame_cur, bg_prev, cost_func=MSD, block_size=8, init_step=4):
    """
    Three step search algorithm 
    Compute the displacement distance from center block of current frame
        with 5 blocks from previous background model (image), namely: 
        center, left, top, right, bottom blocks
    Next the center block of previous frame is selected among 5 blocks 
        according to the cost function (MSD), and the init_step is halved
    The process repeats until init_step reaches 1
    The displacement is the Euclidean distance of the first and last center 
        boxes.

    @param frame_cur (np image): current gray scale frame
    @param bg_prev (np image): prev background model (image), which can be
                                acquired using function like 
                                cv2.createBackgroundSubtractorMOG2().getBackgroundImage()
    @param cost_func (function pointer): name of the cost function to be used
    @param block_size (int): half the size of the block (default=16)
    @param init_step (int): init step size for the Three Step Search algorithm, typical value 4, 7 
                            (default=4). Larger motion require higher step_size

    @return displacement of the center in Euclidean space (float)
    """

    h, w = frame_cur.shape[:2]
    center_x, center_y = w//2, h//2
    assert (block_size < h//2 and block_size < w//2), "block size too big"
    frame_block = frame_cur[center_y-block_size:center_y +
                            block_size, center_x-block_size:center_x+block_size]

    track = [(center_x, center_y)]

    while init_step > 1:
        bg_list = []
        center_list = []

        left_x, left_y = center_x - init_step, center_y
        top_x, top_y = center_x, center_y - init_step
        right_x, right_y = center_x + init_step, center_y
        btm_x, btm_y = center_x, center_y + init_step

        bg_list.append(bg_prev[center_y-block_size:center_y +
                               block_size, center_x-block_size:center_x+block_size])
        center_list.append((center_x, center_y))

        bg_list.append(bg_prev[left_y-block_size:left_y +
                               block_size, left_x-block_size:left_x+block_size])
        center_list.append((left_x, left_y))

        bg_list.append(bg_prev[top_y-block_size:top_y +
                               block_size, top_x-block_size:top_x+block_size])
        center_list.append((top_x, top_y))

        bg_list.append(bg_prev[right_y-block_size:right_y +
                               block_size, right_x-block_size:right_x+block_size])
        center_list.append((right_x, right_y))

        bg_list.append(bg_prev[btm_y-block_size:btm_y +
                               block_size, btm_x-block_size:btm_x+block_size])
        center_list.append((btm_x, btm_y))

        min_msd = float('inf')
        for i, value in enumerate(bg_list):
            msd = cost_func(frame_block, value)
            if msd < min_msd:
                min_msd = msd
                center_x, center_y = center_list[i]

        track.append((center_x, center_y))
        init_step = init_step // 2

    # # Debug
    # tmp = frame_cur.copy()
    # cv2.imshow("Displacement",
    #            cv2.arrowedLine(tmp, track[0], track[-1],
    #                            color=(0, 255, 255), thickness=2,
    #                            line_type=cv2.LINE_AA,
    #                            tipLength=0.3))

    return np.linalg.norm(np.array(track[-1]) - np.array(track[0]))


class AverageMeter():
    def __init__(self, max_size=20):
        self.queue = list()
        self.max_size = max_size

    def update(self, value):
        self.queue.append(value)
        if len(self.queue) > self.max_size:
            self.queue.pop(0)

    def average(self):
        if len(self.queue) > 0:
            return float(sum(self.queue)) / len(self.queue)
        else:
            return 0.
