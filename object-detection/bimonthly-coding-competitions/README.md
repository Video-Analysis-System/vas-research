- [1. Face detection](#1-face-detection)
  - [Dataset](#dataset)
  - [Get started](#get-started)
  - [Baseline:](#baseline)
  - [Rules](#rules)
# 1. Face detection
For our first coding competition, we'll work on face detection. The aim is to experiment with various techniques to improve the accuracy of the baseline Retinface model (with Mobilenet backbone).
## Dataset 
The training set is about 1/3 the original Widerface training set. For validation purpose, we keep all of the Widerface validation set.
The dataset is located in the training server under the directory:
/home/ubuntu/duongld12/Pytorch_Retinaface/data
## Get started
- Note that this is only a simple baseline to get started. You don't have to use it for your own codebase
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

## Rules
The Retinaface code is only a baseline to get started. You **DON'T HAVE TO USE IT** for your codebase. That is, you're free  to use Yolo models, transformers or whatever fancy stuffs you can come up with. There are, however, a few rules that you'll have to follow:
- No pretrained models, except for models trained on **Imagenet-1k** only
- No additional data
- To limit our scope & at the same time, allow for maximum possible creativity, **the model's parameter should only be less than or equal to 45M**, which is about Resnet101's size.
- To check the model size & MACs, please use one of these 2 tools: [Pytorch-OpCounter](https://github.com/Lyken17/pytorch-OpCounter) and [Flops-Counter](https://github.com/sovrasov/flops-counter.pytorch). Model parameters are not perfect but at least, we'll have some ideas on how fast it might be and how much memory it might take
- The final score = speed_score + 2 * accuracy_score
- The winner takes it all (550k yeah)
- The coding competition will last 2 months from 3rd February to 3rd April. We'll check the results & techniques in our seminar on 6th April
- Let's hope that we'll learn many exciting techniques and, with luck, we might invent new and useful algorithms
  
  
 