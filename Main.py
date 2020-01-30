import cv2
from invoice import invoice

typeOfimg="noSuchType"
tableimg =0

def main_fun(img):
    data = []
    data,typeOfimg,tableimg = invoice(img,0)
    return data,typeOfimg,tableimg

