import cv2
import numpy as np
import glob
import os
from PIL import ImageEnhance, Image

read_path = "/Users/ameyakunder/pbv3_imagerec/pbv3_compvision/cropped/*.JPG"
write_path = "/Users/ameyakunder/pbv3_imagerec/pbv3_compvision/post_process/"

for x in glob.glob(read_path):
	
	y = x.split("/", 6)[6]
	
	if os.path.exists(f"{write_path}{y}"):
		continue
	
	else:
		flag = 1 
		
		while flag == 1:
			
			image = Image.open(x)
			enhancer = ImageEnhance.Contrast(image)
			PIL_image = enhancer.enhance(2)

			image = cv2.cvtColor(np.array(PIL_image), cv2.COLOR_RGB2BGR)

			cv2.imshow("Image1", image)
			cv2.waitKey(0)
			cv2.destroyAllWindows()

			(b, g, r) = cv2.split(image)

			thresh_type = input("Input blue/green/red thresh value (b/g/r): ")
			thresh_value = input("Input threshold value: ")
			thresh_value = int(thresh_value)

			if thresh_type == 'b':
				ret, thresh = cv2.threshold(b, thresh_value, 255, cv2.THRESH_BINARY_INV)
			
			elif thresh_type == 'g':
				ret, thresh = cv2.threshold(g, thresh_value, 255, cv2.THRESH_BINARY_INV)
			
			elif thresh_type == 'r':
				ret, thresh = cv2.threshold(r, thresh_value, 255, cv2.THRESH_BINARY)

			cv2.imshow("Threshold", thresh)
			cv2.waitKey(0)
			cv2.destroyAllWindows()

			var = input("If you're satisfied, enter 'q'. Else any other key: ")
			if var == 'q':
				flag = 0
			elif var != 'q':
				continue

		cv2.imwrite(f"{write_path}{y}", thresh)

