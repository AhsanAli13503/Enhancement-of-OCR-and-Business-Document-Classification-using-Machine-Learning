#text dataset creation from images 

import cv2
import pytesseract

import os
line=[]
filepath = 'test.txt'
path = []
label = []
with open(filepath) as fp:
    for cnt, line in enumerate(fp):
        data = line.split()
        path.append( data[0]) 
        label.append("__label__" + data[1])
i=0
with open("dataset3.txt", 'w') as writer:
    for pa in path:
        try:
            img = cv2.imread(pa,0)
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
            writer.write(label[i]+" "+new+"\n")
            i=i+1

        except:
            print("errror")