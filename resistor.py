import cv2
import numpy as np
import os
import math
import matplotlib.pyplot as plt



def color_deviation_cols(img, dim, k1, k2):
	std_cols = []
	channels = img.shape[2]
	for i in range(0, img.shape[dim]):
		std_dev = []
		for c in range(0, channels):
			std_dev.append(np.std(img[:,i,c]))
		std_cols.append(k1*std_dev[0] + k2*(std_dev[1] + std_dev[2]))

	return np.array(std_cols)

def color_deviation_rows(img, dim, k1, k2):
	std_cols = []
	channels = img.shape[2]
	for i in range(0, img.shape[dim]):
		std_dev = []
		for c in range(0, channels):
			std_dev.append(np.std(img[i,:,c]))
		std_cols.append(k1*std_dev[0] + k2*(std_dev[1] + std_dev[2]))

	return np.array(std_cols)

def normalize(deviations):
	deviations *= (255.0/deviations.max())
	return deviations

def find_bounds(img, std_cols, std_rows, v_thresh, h_thresh):
	xmin =-1
	ymin = 1
	for x in range(0, img.shape[1]):
		if std_cols[x] > h_thresh:
			if xmin < 0:
				xmin = x
			xmax = x

	ymin = -1
	ymax = 1

	for y in range(0, img.shape[0]):
		if std_rows[y] > v_thresh:
			if ymin < 0:
				ymin = y
			ymax = y 

	return list([xmin, xmax, ymin, ymax])

def get_bounding_box(img):
	img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	ret, thresh = cv2.threshold(img,127,255,0)

	kernel = np.ones((11,11),np.uint8)
	erosion = cv2.erode(thresh,kernel,iterations = 1)

	invert = cv2.bitwise_not(erosion)

	im2, contours, hierarchy = cv2.findContours(invert, 1, 2)
	cnt = contours[0]

	rect = cv2.minAreaRect(cnt)
	centre = rect[0]
	width = rect[1][0]
	height = rect[1][1]
	angle = rect[2]
	if width < height:
	    angle = angle + 90
	    temp = width
	    width = height
	    height = temp
	box = cv2.boxPoints(rect)
	box = np.int0(box)
	
	return box, rect

def get_rotation_matrix(centre, rect):
	angle = rect[2]

	rot_mat = cv2.getRotationMatrix2D(centre, angle, 1.0)

	return rot_mat

def crop_and_rotate(img, rot_mat, rect):
	width = rect[1][0]
	height = rect[1][1]
	print('Rot Mat', rot_mat)
	result = cv2.warpAffine(img, rot_mat, (int(width),int(height)), flags=cv2.INTER_LINEAR)

	#cv2.imshow("res", result)
	#cv2.waitKey()

	rows = img.shape[0]
	cols = img.shape[1]

	delta_width = cols-width
	#print(delta_width)
	delta_height = rows-height
	#print(delta_height)

	#print("Rows: ", int(delta_height/2, rows-int(delta_height/2)))

	#result = result[int(delta_height/2):rows-int(delta_height/2)][int(delta_width/2):cols-int(delta_width/2)]
	cv2.imshow("test", result)
	cv2.waitKey()
	return result

def get_horizontal(img):
	box, rect = get_bounding_box(img)
	cv2.drawContours(img,[box],0,(0,0,255),2)

	angle = rect[2]
	width = rect[1][0]
	height = rect[1][1]

	rows = img.shape[0]
	cols = img.shape[1]
	print("Original Rows, Cols: ", rows, cols)
	print("Rectangle Centre: ", rect[0][1], rect[0][0])

	print("Rect Height, Width: ", height, width)
	print("Angle: ", angle)

	angle_rad = angle*0.0174533

	#new_width = int(math.cos(angle_rad)*height + math.sin(angle_rad)*width)
	#new_height = int(math.sin(angle_rad)*height + math.cos(angle_rad)*width)

	new_width = cols
	new_height = rows
	print("New Height, New Width: ", new_height, new_width)

	y_offset = (new_height-1)/2.0 - rect[0][1]
	x_offset = (new_width-1)/2.0 - rect[0][0]
	print("Offsets: ", x_offset, y_offset)

	M = np.array([[1,0, x_offset],[0, 1, y_offset]])
	img = cv2.warpAffine(img,M,(new_width, new_height))

	#cv2.imshow("translate", img)
	#cv2.waitKey()

	centre = ((new_height)/2.0, (new_width)/2.0)

	rot_mat = get_rotation_matrix(centre, rect)
	return crop_and_rotate(img, rot_mat, rect)


def main():
	#Parameters
	v_threshold = 60
	h_threshold = 120
	k1 = 1
	k2 = 10

	img = cv2.imread(os.path.join(os.getcwd(), 'images', 'ours1.jpg'), 1)

	rgbimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	labimg = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

	median_lab = cv2.medianBlur(labimg, 5)
	gaussian_lab = cv2.GaussianBlur(median_lab, (5,5), 0)

	std_cols = color_deviation_cols(gaussian_lab, 1, k1, k2)
	std_rows = color_deviation_rows(gaussian_lab, 0, k1, k2)

	normalized_cols = normalize(std_cols)
	normalized_rows = normalize(std_rows)

	col_bands = np.tile(normalized_cols, (50, 1))
	row_bands = np.transpose(np.tile(normalized_rows, (50, 1)))

	xmin, xmax, ymin, ymax = find_bounds(labimg, std_cols, std_rows, v_threshold, h_threshold)

	crop_img = img[ymin:ymax, xmin:xmax]

	rotated_img = get_horizontal(crop_img)

	out_img = rotated_img

	fig = plt.figure(figsize=(3,2))

	fig.add_subplot(3,2,1)
	plt.imshow(labimg)

	fig.add_subplot(3,2,2)
	plt.imshow(row_bands, cmap='gray')

	fig.add_subplot(3,2,3)
	plt.imshow(col_bands, cmap='gray')

	fig.add_subplot(3,2,5)
	plt.imshow(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))

	fig.add_subplot(3,2,6)
	plt.imshow(cv2.cvtColor(rotated_img, cv2.COLOR_BGR2RGB))

	plt.show()

	#cv2.imshow("jang", crop_img)

	#cv2.waitKey()

if __name__ == '__main__':
	main()
