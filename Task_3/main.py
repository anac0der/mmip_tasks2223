import numpy as np
import cv2
import sys
from canny import grad_module, nonmax_suppression, canny
from ridges import ridge_detection

path = sys.argv[-2]
output_path = sys.argv[-1]
img = np.array(cv2.imread(path, cv2.IMREAD_GRAYSCALE)).astype("int64")

command = sys.argv[1]

if command == 'grad':
	sigma = float(sys.argv[2])
	result_img = grad_module(img, sigma)[0]

elif command == 'nonmax':
	sigma = float(sys.argv[2])
	result_img = nonmax_suppression(img, sigma)

elif command == 'canny':
	sigma = float(sys.argv[2])
	thr2 =  float(sys.argv[3])
	thr1 =  float(sys.argv[4])
	result_img = canny(img, sigma, thr1, thr2)

elif command == 'vessels':
	s = [2, 3, 4]
	result_img = ridge_detection(img, s)

else:
	raise Exception('Некорректная команда!')

cv2.imwrite(output_path, result_img)