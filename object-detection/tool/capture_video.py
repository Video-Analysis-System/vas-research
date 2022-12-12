import argparse
import cv2


def extractImages(pathIn, pathOut,time, imageName):
    count = 0
    vidcap = cv2.VideoCapture(pathIn)
    success,image = vidcap.read()
    success = True
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*time))    # example: 1000 = 1s
        success,image = vidcap.read()
        print ('Read a new frame: ', success)
        cv2.imwrite(pathOut + "/"+imageName+"%d.jpg" % count, image)     # save frame as JPEG file
        count = count + 1

if __name__=="__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--pathIn", help="path to video")
    a.add_argument("--pathOut", help="path folder save images")
    a.add_argument("--time", type =int, help = "time per capture") 
    a.add_argument("--imageName", help = "image name") 
    args = a.parse_args()
    extractImages(args.pathIn, args.pathOut, args.time, args.imageName)

#run code: python3 capture_video.py --pathIn video/video.mp4 --pathOut image --time 1000 --imageName frame