import numpy as np
import cv2 as cv
import numpy

def detectAndCompute(cmpimg,qurimg):
    orb = cv.ORB_create()
    __, des1 = orb.detectAndCompute(cmpimg,None)
    __, des2 = orb.detectAndCompute(qurimg,None)
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1,des2)
    return len(matches)

def templateMatch(queryimg):
    in1 = cv.imread("document\\invoice\\invoice.png",0)
    in2 = cv.imread("document\\invoice\\input3.png",0)
    in3 = cv.imread("document\\invoice\\input5.png",0)
    
    let1 = cv.imread("document\\letter\\letter1.jpg",0)
    let2 = cv.imread("document\\letter\\letter2.jpg",0)
    let3 = cv.imread("document\\letter\\letter3.png",0)

    rece1 = cv.imread("document\\rece\\rece.jpg",0)
    rece2 = cv.imread("document\\rece\\rece2.jpg",0)
    rece3 = cv.imread("document\\rece\\rece3.jpg",0)
    
    
    match =[]
    match.append(detectAndCompute(in1,queryimg))
    match.append(detectAndCompute(in2,queryimg))
    match.append(detectAndCompute(in3,queryimg))

    match.append(detectAndCompute(let1,queryimg))
    match.append(detectAndCompute(let2,queryimg))
    match.append(detectAndCompute(let3,queryimg))

    match.append(detectAndCompute(rece1,queryimg))
    match.append(detectAndCompute(rece2,queryimg))
    match.append(detectAndCompute(rece3,queryimg))
    match = np.array(match)
    a=int(np.argmax(match))
    a = int((a+1)/3)
    if a == 0:
        return 'i'
    elif a == 1:
        return 'l'
    else:
        return 'r'
