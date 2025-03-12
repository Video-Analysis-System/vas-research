# Introduction:
Ultralytics now uses an object-oriented design so that it can train/infer multiple Yolo versions and other object detection models. It's very convenient for casual users but the code is now more difficult to understand than, say Yolov5's code.

# Package components

- The main training loop is now in the class BaseTrainer in “ultralytics/ultralytics/engine/trainer.py”. Specific trainers for detection, segmentation, ... have their separate classes located in different packages

- The trainer for detection is the class DetectionTrainer (inherited from BaseTrainer) in “ultralytics/ultralytics/models/yolo/detect/train.py”, which mostly does some data loading stuffs

- The model class code now handles, not only forward, but also contains loss variables/members to compute losses

- Look like all the models are now in “ultralytics/ultralytics/nn/tasks.py”. The base class is BaseModel consisting of most scaffolding code, including setting loss, which is now “criterion” member, by a method “init_criterion” 

- The DetectionModel class overrides the method “init_criterion” to offer 2 choices for loss E2EDetectLoss and v8DetectionLoss

- All the loss classes are in the file “ultralytics/ultralytics/utils/loss.py”. See my subsection “Losses”

# Losses
- All the loss classes are in the file “ultralytics/ultralytics/utils/loss.py”. The main detection loss is v8DetectionLoss. 

  - E2EDetectLoss calls v8DetectionLoss. Look like they got this from Yolov10: https://github.com/ultralytics/ultralytics/issues/16613

  - v8DetectionLoss has a very important member called “assigner”. Assigner's role is to assign ground truth bounding boxes to anchors for training. Anchors here mean either anchor points as in anchor-free models or anchor boxes in other models. Ultralytics uses an algorithm called “TAL” (Task alignment learning” from the paper “TOOD: Task-aligned One-stage Object Detection”. The code is based on PPYOLOE's code here:

  - https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py

  - This code, in turn, is based on the original TAL code: 
  - https://github.com/fcjian/TOOD/blob/master/mmdet/core/bbox/assigners/task_aligned_assigner.py
  - As we can see, this is based on mmdetection code. And mmdetection code supports many other assigners
  - It looks like Ultralytics doesn't use T-head in the TOOD paper?
