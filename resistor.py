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
            std_dev.append(np.std(img[:, i, c]))
        std_cols.append(k1 * std_dev[0] + k2 * (std_dev[1] + std_dev[2]))

    return np.array(std_cols)


def color_deviation_rows(img, dim, k1, k2):
    std_cols = []
    channels = img.shape[2]
    for i in range(0, img.shape[dim]):
        std_dev = []
        for c in range(0, channels):
            std_dev.append(np.std(img[i, :, c]))
        std_cols.append(k1 * std_dev[0] + k2 * (std_dev[1] + std_dev[2]))

    return np.array(std_cols)


def normalize(deviations):
    deviations *= (255.0 / deviations.max())
    return deviations


def find_bounds(img, std_cols, std_rows, v_thresh, h_thresh):
    xmin = -1
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
    ret, thresh = cv2.threshold(img, 127, 255, 0)

    kernel = np.ones((11, 11), np.uint8)
    erosion = cv2.erode(thresh, kernel, iterations=1)

    invert = cv2.bitwise_not(erosion)

    cv2.imshow("invert", invert)
    cv2.waitKey()

    im2, contours, hierarchy = cv2.findContours(invert, 1, 2)

    cnt = contours[0]
    for i in range(1, len(contours)):
    	if len(contours[i]) > len(cnt):
    		cnt = contours[i]


    #print("CONTOURS: ", contours[1])

    rect = cv2.minAreaRect(cnt)
    centre = rect[0]
    width = rect[1][0]
    height = rect[1][1]
    angle = rect[2]

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

    nH = img.shape[0]
    nW = img.shape[1]

    h = img.shape[0]
    w = img.shape[1]

    cos = abs(rot_mat[0, 0])
    sin = abs(rot_mat[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    center = tuple(np.array(img.shape[1::-1]) / 2)

    rot_mat[0, 2] += (nW / 2 - center[0])
    rot_mat[1, 2] += (nH / 2 - center[1])

    result = cv2.warpAffine(img, rot_mat, (nW, nH), flags=cv2.INTER_LINEAR)

    cv2.imshow("ressss", result)
    cv2.waitKey()

    rows = result.shape[0]
    cols = result.shape[1]
    print("Rows, Cols: ", rows, cols)

    delta_width = cols - width
    if delta_width < 0:
        delta_width = 0
    print("Delta_W: ", delta_width)
    delta_height = rows - height
    if delta_height < 0:
        delta_height = 0
    print("Delta_H: ", delta_height)

    print("Rows: ", int(delta_height / 2), rows - int(delta_height / 2))
    print("Cols: ", int(delta_width / 2), cols - int(delta_width / 2))

    result = result[int(delta_height / 2):rows - int(delta_height / 2),
             int(delta_width / 2):cols - int(delta_width / 2)]
    result = result[10:(result.shape[0] - 10), 10:(result.shape[1] - 10)]

    cv2.imshow("snack", result)
    cv2.waitKey()

    res_center = tuple(np.array(result.shape[1::-1]) / 2)
    print("Res Center: ", res_center)

    h = result.shape[0]
    w = result.shape[1]

    if width < height:
        rot = cv2.getRotationMatrix2D(res_center, 90, 1.0)
        cos = abs(rot[0, 0])
        sin = abs(rot[0, 1])

        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        rot[0, 2] += (nW / 2 - res_center[0])
        rot[1, 2] += (nH / 2 - res_center[1])

        result = cv2.warpAffine(result, rot, (nW, nH), flags=cv2.INTER_LINEAR)

        cv2.imshow("yeet", result)
        cv2.waitKey()

    return result


def get_horizontal(img):
    box, rect = get_bounding_box(img)
    cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

    angle = rect[2]
    width = rect[1][0]
    height = rect[1][1]

    rows = img.shape[0]
    cols = img.shape[1]
    print("Original Rows, Cols: ", rows, cols)
    print("Rectangle Centre: ", rect[0][1], rect[0][0])

    print("Rect Height, Width: ", height, width)
    print("Angle: ", angle)
    cv2.imshow("gsdg", img)
    cv2.waitKey()

    new_width = cols
    new_height = rows
    print("New Height, New Width: ", new_height, new_width)

    y_offset = (new_height - 1) / 2.0 - rect[0][1]
    x_offset = (new_width - 1) / 2.0 - rect[0][0]
    print("Offsets: ", x_offset, y_offset)

    M = np.array([[1, 0, x_offset], [0, 1, y_offset]])
    img = cv2.warpAffine(img, M, (new_width, new_height))

    centre = ((new_width - 1) / 2.0, (new_height - 1) / 2.0)
    print("Centre", centre)

    rot_mat = get_rotation_matrix(centre, rect)
    return crop_and_rotate(img, rot_mat, rect)


def process_resistor(img):
    height, width = img.shape[:2]
    print("shape", img.shape[:2])

    chop_pct = int(0.10*height)

    # crop out the reflective part
    crop_im_top = img[0:int(height / 2 - chop_pct), 0:width]
    crop_im_bot = img[int(height / 2 + chop_pct):height, 0:width]
    crop_img = np.concatenate((crop_im_top, crop_im_bot), axis=0)
    #cv2.imshow("crop", crop_img)
    #cv2.waitKey()

    # average the columns
    for i in range(0, crop_img.shape[1]):
        channels = crop_img.shape[2]
        for c in range(0, channels):
            val = cv2.mean(crop_img[:, i, c])
            crop_img[:, i, c] = val[0]

    #cv2.imshow("crop", crop_img)
    #cv2.waitKey()

    # make it thicker for easier viewing
    double_height = np.concatenate((crop_img, crop_img), axis=0)
    d_height, d_width = double_height.shape[:2]

    # make a copy in LAB
    rgb_img = np.copy(crop_img)
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

    # find edges
    print("double_height", double_height.shape)
    #cv2.imshow("iajnsf", double_height)
    #cv2.waitKey()
    gray_img = cv2.cvtColor(double_height, cv2.COLOR_RGB2GRAY)
    print("Img Shape: ", gray_img.shape)
    blur = cv2.blur(gray_img, (5, 5))
    #cv2.imshow("Blur", blur)
    #cv2.waitKey()
    kernel = np.ones((5, 5), np.uint8)
    erode = cv2.morphologyEx(blur, cv2.MORPH_ERODE, kernel)
    #cv2.imshow("Erode", erode)
    #cv2.waitKey()
    edges = cv2.Canny(erode, 1, 57)
    #cv2.imshow("Canny", edges)
    #cv2.waitKey()

    min_line_length = 20
    max_line_gap = 20
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 15, min_line_length, max_line_gap)
    pos_line = []

    # add the rings location to pos_line and display the huff lines
    for x in range(0, len(lines)):
        for x1, y1, x2, y2 in lines[x]:
            pos_line.append(x1)
            cv2.line(double_height, (x1, y1), (x2, y2), (0, 255, 0), 2)

    pos_line.sort()
    rings = []

    # go through every other line, average the value between it and the next, use that as the value for the ring
    cv2.imshow("res line", double_height)
    cv2.waitKey()
    print("Len: ", len(pos_line))
    for i in range(0, len(pos_line), 2):
        x = pos_line[i]
        print("I: ", i)
        x1 = pos_line[i + 1]
        ring_section = rgb_img[2:d_width-2, x+2:x1-2]
        value = [ring_section[:, :, i].mean() for i in range(ring_section.shape[-1])]
        rings.append(value)

    return double_height, rings


def match_colour(rgb_val):
	colours = [('black', [50,50,50]),
	('brown', [70,30,30]),
	('red', [120,50,50]),
	('orange', [255,165,0]),
	('yellow', 	[255,255,0]),
	('green', [40,70,40]),
	('blue', [55,65,95]),
	('purple', [128,0,128]),
	('gray', [128,128,128]),
	('white', [255,255,255])]

	#print("RGB: ", rgb_val)

	closest = 0
	shortest_dist = 442		# max distance possible
	for i in range(0, len(colours)):
		#print(colours[i])
		#print("Test: ", rgb_val[2]**2 - colours[i][1][2]**2)
		distance = math.sqrt(2*(rgb_val[0] - colours[i][1][0])**2 + 
		4*(rgb_val[1] - colours[i][1][1])**2 +
		3*(rgb_val[2] - colours[i][1][2])**2)

		#print("Distance: ", distance)

		if distance < shortest_dist:
			#print("Colour: ", colours[i][0])
			#print("Distance: ", distance)
			shortest_dist = distance
			closest = i


	colour_name = colours[closest][0]
	value = colours[closest][1]
	return colour_name, value


def main():
    # Parameters
    v_threshold = 60
    h_threshold = 120
    k1 = 1
    k2 = 10

    img = cv2.imread(os.path.join(os.getcwd(), 'images', 'r5.jpg'), 1)
    #cv2.imshow("orig", img)
    #cv2.waitKey()

    # res_center = tuple(np.array(img.shape[1::-1])/2)
    # rot = cv2.getRotationMatrix2D(res_center, 90, 1.0)
    #img = cv2.warpAffine(img, rot, (img.shape[0],img.shape[1]), flags = cv2.INTER_LINEAR)
    #img = img[200:600, 100:300]

    rgbimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    labimg = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

    median_lab = cv2.medianBlur(labimg, 5)
    gaussian_lab = cv2.GaussianBlur(median_lab, (5, 5), 0)

    std_cols = color_deviation_cols(gaussian_lab, 1, k1, k2)
    std_rows = color_deviation_rows(gaussian_lab, 0, k1, k2)

    normalized_cols = normalize(std_cols)
    normalized_rows = normalize(std_rows)

    col_bands = np.tile(normalized_cols, (50, 1))
    row_bands = np.transpose(np.tile(normalized_rows, (50, 1)))

    xmin, xmax, ymin, ymax = find_bounds(labimg, std_cols, std_rows, v_threshold, h_threshold)

    crop_img = img[ymin:ymax, xmin:xmax]

    rotated_img = get_horizontal(crop_img)
    #cv2.imshow("orig", rotated_img)
    #cv2.waitKey()


    # Bronwyn was here
    out_img = rotated_img
    resistor_w_lines, rings = process_resistor(out_img)
    print('resistor rings', rings)

    #resistor_w_lines = cv2.cvtColor(resistor_w_lines, cv2.COLOR_RGB2Lab)

    for i in range(0,len(rings)):
    	name, val = match_colour(rings[i])
    	print("Name: ", name)

    fig = plt.figure(figsize=(4, 2))

    fig.add_subplot(4, 2, 1)
    plt.imshow(labimg)

    fig.add_subplot(4, 2, 2)
    plt.imshow(row_bands, cmap='gray')

    fig.add_subplot(4, 2, 3)
    plt.imshow(col_bands, cmap='gray')

    fig.add_subplot(4, 2, 5)
    plt.imshow(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))

    fig.add_subplot(4, 2, 6)
    plt.imshow(cv2.cvtColor(rotated_img, cv2.COLOR_BGR2RGB))

    fig.add_subplot(4, 2, 7)
    plt.imshow(cv2.cvtColor(resistor_w_lines, cv2.COLOR_BGR2RGB))

    plt.show()


if __name__ == '__main__':
    main()
