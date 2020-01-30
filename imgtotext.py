import numpy as np
import cv2
import tensorflow as tf 
from wordsegment import load, segment
from ocrModelLoader import abc,alphabat,number
from charsegment import charsegment

class_names = ['0','1']

def predict_alpa(img):
    for i in range (len(alphabat)):
        pred = alphabat[i].predict(img)
        result=class_names[np.argmax(pred)] 
        if result == "1":
           return chr(97+i)
    return ""

def predict_num(img):
    for i in range (len(number)):
        pred = number[i].predict(img)
        result=class_names[np.argmax(pred)]
        if result == "1":
           return str(i)
    return ""

def img_to_text(imgdata):
    text = ""
    for img in imgdata:
        dim = (16,16)
        test = cv2.resize(img, dim, interpolation =cv2.INTER_AREA)
        
        test = (np.expand_dims(test,0))
        test = test /255.0
        pred = abc.predict(test)
        result=class_names[np.argmax(pred)] 
        if result == "1":
           word = predict_alpa(test)
           
           text = text + word
        else:
           word = predict_num(test)
         
           text = text + word
    return text       


