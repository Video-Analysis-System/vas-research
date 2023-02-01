- [1. Face detection based on Retinaface code](#1-face-detection-based-on-retinaface-code)
  - [Dataset](#dataset)
  - [Get started](#get-started)
  - [Baseline:](#baseline)
# 1. Face detection based on Retinaface code
For our first coding exercise, we'll work on face detection based on Retinaface code. The aim is to experiment with various techniques to improve the accuracy of the baseline model Mobilenet.
## Dataset 
The training set is about 1/3 the original Widerface training set. For validation purpose, we keep all of the Widerface validation set.
The dataset is located in the training server under the directory:
/home/ubuntu/duongld12/Pytorch_Retinaface/data
## Get started
- Clone the repository https://github.com/biubug6/Pytorch_Retinaface to the location of your choice  
- cd "Pytorch_Retinaface/data"
- mkdir widerface
- cd widerface
- Copy the subdirectories "train" and "val" of the dataset (at the location specified in the previous section) to the current "widerface" directory
- You might need to open the file "config.py" to change the parameter "batch_size" according to the memory of your GPU. For Rtx2060 super with 7GB, I have to set batch size to 32
- If you don't want to use a pretrained file, you should set the parameter "pretrain" to False in the "config.py" file
- Let's start the training process:  CUDA_VISIBLE_DEVICES=0 python train.py --network mobile0.25 
- It'll create a directory "weights" and store the trained checkpoints there
- Evaluation: See the instruction in the repo. I had a problem "Out-of-memory" when using GPU for running the evaluation code so in the end, I run the code using CPU

## Baseline:
Without using a pretrained model, running the training command above with the Widerface subset we use, I got the following result:
- Easy Val AP: 0.848556325273459 
- Medium Val AP: 0.8182168942485764 
- Hard Val AP: 0.6428735812942992

So yeah, there are a lot of room for improvement here.

 