from os import listdir, remove
import argparse

def remove_image(path_labels, path_images):
    labels = listdir(path_labels)
    images = listdir(path_images)
    for image in images:
        if '{}.{}'.format(image.split('.')[0], 'txt') not in labels:
            print('Going to remove %s' % image)
            remove(f'{path_images}/%s' % image)


if __name__=="__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--pathLabels", help="path to labels folder ")
    a.add_argument("--pathImages", help="path to images folder")
    args = a.parse_args()
    remove_image(args.pathLabels, args.pathImages)

#Run code: python3 remove_image.py --pathLabels labels  --pathImages images