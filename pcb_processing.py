import cv2
import imutils
import os

read_path = "/Users/ameyakunder/pbv3_imagerec/pbv3_compvision/cropped/"
'''write_path = "/Users/ameyakunder/pbv3_imagerec/pbv3_compvision/post_process"

for image_name in os.listdir(read_path):
	inputPath = os.path.join(path, imageName) 
	image = cv2.imread(inputPath, 0)'''

image = cv2.imread(f"{read_path}2010006.JPG", -1)
(b, g, r) = cv2.split(image)
ret1, thresh1 = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)

cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#blur = cv2.GaussianBlur(resize, (3,3), 0)
#cv2.imshow("Blur", blur)



