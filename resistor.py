import cv2
import numpy as np
import math
from matplotlib import pyplot as plt

img = cv2.imread('resistor.jpg', 1)
rgbimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
median = cv2.medianBlur(rgbimg, 5)
gaussian = cv2.GaussianBlur(median, (5, 5), 0)
kernel = np.ones((5,5),np.uint8)
opening = cv2.morphologyEx(gaussian, cv2.MORPH_OPEN, kernel)


grayimg = cv2.cvtColor(opening, cv2.COLOR_RGB2GRAY)
blur = cv2.blur(grayimg, (5, 5))
erode = cv2.morphologyEx(blur, cv2.MORPH_ERODE, kernel)
edges = cv2.Canny(erode, 15, 60)

colorEdge = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
colorEdgeCopy = np.copy(colorEdge)

lines = cv2.HoughLines(edges, 1, np.pi / 180, 80, None, 10, 5)

if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv2.line(colorEdge, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)

cv2.imshow("Source", img)
cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", colorEdge)
#cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", colorEdgeCopy)

cv2.waitKey()

# plt.subplot(121), plt.imshow(erode, cmap='gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(edges, cmap='gray')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
#
# plt.show()