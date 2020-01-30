import cv2
def charsegment(roi):
    __,th=cv2.threshold(roi, 0, 255,cv2.THRESH_OTSU|cv2.THRESH_BINARY_INV)
    _, thresh  = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY)
   
    __,contours,hirerchy = cv2.findContours(th,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    sorted_ctrs = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    segment=[]
    for c in sorted_ctrs:
        x, y, w, h = cv2.boundingRect(c)
        
        roi =thresh[y :y + h, x:x + w]
        BLUE = [255,255,255]
        roi= cv2.copyMakeBorder(roi,15,15,15,15,cv2.BORDER_CONSTANT,value=BLUE)
        segment.append(roi)
    return segment
