import cv2
import pytesseract

roi = cv2.imread("1.jpg",0)
text = pytesseract.image_to_string(roi)
file1 = open("myfile.txt","w") 
# \n is placed to indicate EOL (End of Line) 
file1.write(text)  
file1.close() #to change file access modes 
  
