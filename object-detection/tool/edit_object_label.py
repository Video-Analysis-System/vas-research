import os
import argparse

def edit_object(folder_path, number):
    files = []
    # Add the path of txt folder
    for i in os.listdir(folder_path):
        if i.endswith('.txt'):
            files.append(folder_path+"/"+i)

    for item in files:
        # define an empty list
        file_data = []

        # open file and read the content in a list
        with open(item, 'r') as myfile:
            for line in myfile:
                # remove linebreak which is the last character of the string
                currentLine = line[:-1]
                data = currentLine.split(" ")
                # add item to the list
                file_data.append(data)
        
        # Edit number class in any line by one
        for i in file_data:
            if i[0].isdigit():
                temp = number
                i[0] = str(int(temp))

        # Write back to the file
        f = open(item, 'w')
        for i in file_data:
            res = ""
            for j in i:
                res += j + " "
            f.write(res)
            f.write("\n")
        f.close()
    print("Done!")

if __name__=="__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--pathLabels", help="path to labels folder ")
    a.add_argument("--number", type=int, help="number of class name")
    args = a.parse_args()
    edit_object(args.pathLabels, args.number)

#Run code: python3 edit_object_label.py --pathLabels labels --number 0
