import cv2
import imutils
import os.path
import numpy as np

ref_point = []

def shape_select(event, x, y, flags, param):
  
  global ref_point

  if event == cv2.EVENT_LBUTTONDOWN:
    ref_point = [(x, y)]

  elif event == cv2.EVENT_LBUTTONUP:
    ref_point.append((x, y))

    cv2.rectangle(image, ref_point[0], ref_point[1], (0, 0, 0), 2)
    cv2.imshow("image", image)

first = input("Enter first file number: ")
first = int(first)
last = input("Enter last file number: ")
last = int(last)

for i in np.arange(first, last):
	#change adress depending upon the location of directory
	read_add = f"/Users/ameyakunder/pbv3_imagerec/drive-download/IMG_{i}.JPG"
	#checking if the file exists
	if os.path.exists(read_add):
		image = cv2.imread(read_add)
		image = imutils.resize(image, width = 900)
		#reads and resizes the image

		clone = image.copy()
		cv2.namedWindow("image")
		cv2.moveWindow("image", 20, 20)
		cv2.setMouseCallback("image", shape_select)

		while True:
			# display the image and wait for a keypress
			cv2.imshow("image", image)
			key = cv2.waitKey(1) & 0xFF

			# if the 'r' key is pressed, reset the cropping region
			if key == ord("r"):
				image = clone.copy()

			# if the 'c' key is pressed, break from the loop
			elif key == ord("c"):
				break

		if len(ref_point) == 2:
			crop_img = clone[ref_point[0][1]:ref_point[1][1], ref_point[0][0]:ref_point[1][0]]
			crop_img = cv2.resize(crop_img, (650, 374))
			cv2.imshow("crop_img", crop_img)
			cv2.waitKey(0)

		serial = input("Enter the serial number: ")
		write_add = f"/Users/ameyakunder/pbv3_imagerec/pbv3_compvision/cropped/{serial}.JPG"
		cv2.imwrite(write_add, crop_img)
		cv2.destroyAllWindows()




