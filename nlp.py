#text dataset creation from images 

import cv2
import pytesseract

import os

path = 'D:\\Study Material\\project practices\\Complete Project\\Document Classification\\Memo\\'

files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        files.append(os.path.join(r, file))
labelName = "__label__1"
with open("file1.txt", 'w') as writer:
    for f in files:
        img = cv2.imread(f,0)
        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        img = cv2.medianBlur(img, 3)
        img = cv2.bilateralFilter(img,9,75,75)
        text = pytesseract.image_to_string(img).rstrip()
        text = text.split()
        
        new=""
        for t in text:
            new = new +" "+t
        print(new)  
        writer.write(labelName+" "+new+"\n")
        

