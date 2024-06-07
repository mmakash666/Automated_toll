import cv2
import easyocr
import numpy as np
import pytesseract
from PIL import Image
import imutils
import matplotlib.pyplot as plotter
import os
#pytesseract.pytesseract.tesseract_cmd = r"C:\Users\21 Tech\AppData\Local\Programs\Python\Python312\Lib\site-packages\pytesseract"

# DRY code for showing images via matplotlib
def show_image(image):
    plotter.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plotter.show()

def detect_numberplate(original_image_path, cropped_image_path):
    original_image = cv2.imread(original_image_path)
    grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY) # turn image grayscale
    smoothened_image = cv2.bilateralFilter(grayscale_image, 11,17,17) # smoothen image, except for pronounced edges
    edges_highlighted_image = cv2.Canny(smoothened_image, 170,200) # turn everything in the image black and white apart from the edges

    # using the edges to draw contours on the image
    contours, new = cv2.findContours(edges_highlighted_image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contoured_image = original_image.copy()
    cv2.drawContours(contoured_image,contours,-1,(0,225,0),3)

    # refining the contours
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]
    NumberPlateCount = None
    refined_countoured_image = original_image.copy()
    cv2.drawContours(refined_countoured_image,contours,-1,(0,255,0),3)

    count = 0
    image_name_numbering = 1
    for i in contours:
        perimeter = cv2.arcLength(i, True)
        approx = cv2.approxPolyDP(i,0.02*perimeter,True)
        if(len(approx)==4):
            NumberPlateCount = approx
            x,y,w,h = cv2.boundingRect(i)
            crp_img = original_image[y:y+h, x:x+w]
            cv2.imwrite(str(image_name_numbering)+ '.png', crp_img)
            image_name_numbering +=1
            break

    os.replace("1.png", cropped_image_path)
    cropped_image = cv2.imread(cropped_image_path)

    # showing all image created in each processing step
    debug = True
    if debug:
        show_image(grayscale_image)
        show_image(smoothened_image)
        show_image(edges_highlighted_image)
        show_image(contoured_image)
        show_image(refined_countoured_image)
        show_image(cropped_image)




def read_license_number_OCR(numberplate_image_path):
    reader = easyocr.Reader(['en'], False)
    result = reader.readtext(numberplate_image_path)
    for detection in result:
        print(detection[1])  # Index 1 contains the recognized text




if __name__ == '__main__':
    detect_numberplate("car_images/Cars14.png", 'numberplate_cropped.png')
    read_license_number_OCR('numberplate_cropped.png')