import cv2
import easyocr
import numpy as np
import pytesseract
from PIL import Image
import imutils
import matplotlib.pyplot as plt
#pytesseract.pytesseract.tesseract_cmd = r"C:\Users\21 Tech\AppData\Local\Programs\Python\Python312\Lib\site-packages\pytesseract"
original_image = cv2.imread("car_images/Cars234.png")

# turn image grayscale -> binarize image (boost contrast)
grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
binarized_image = cv2.bilateralFilter(grayscale_image, 11,17,17)
plt.imshow(cv2.cvtColor(binarized_image, cv2.COLOR_BGR2RGB))
plt.show()


edge = cv2.Canny(binarized_image, 170,200)
cnts, new = cv2.findContours(edge.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
image1 = original_image.copy()
cv2.drawContours(image1,cnts,-1,(0,225,0),3)
cnts =sorted(cnts, key=cv2.contourArea, reverse=True)[:30]
NumberPlateCount = None
image2 = image.copy()
cv2.drawContours(image2,cnts,-1,(0,255,0),3)
count = 0
name = 1
for i in cnts:
    perimeter = cv2.arcLength(i, True)
    approx = cv2.approxPolyDP(i,0.02*perimeter,True)
    if(len(approx)==4):
        NumberPlateCount = approx
        x,y,w,h = cv2.boundingRect(i)
        crp_img = image[y:y+h, x:x+w]
        cv2.imwrite(str(name)+ '.png', crp_img)
        name +=1
        break

crp_img_loc = '1.png'

image = cv2.imread(crp_img_loc)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()




reader = easyocr.Reader(['en'])

# Load the image
image_path = '1.png'
result = reader.readtext(image_path)


# this is used to detect the text
for detection in result:
    print(detection[1])  # Index 1 contains the recognized text
