import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# Read image -- BGR
img = cv.imread("../data/images/deer.jpg")

# Create histogram for each color channel
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
    plt.ylim([0,5000])
    plt.title("Histogram RGB")
plt.show()

# Change color space to HSV
img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

# Create histogram for each chanel HSV
channel = ('h','s','v')
for i in range(len(color)):
    histr = cv.calcHist([img_hsv],[i],None,[256],[0,256])
    plt.plot(histr,color = color[i], label=channel[i])
    plt.xlim([0,256])
    plt.ylim([0, 50000])
    plt.title("Histogram HSV")
    plt.legend()
plt.show()

cv.imshow('img',img)
img_hsv_2 = img_hsv.copy()
hue = img_hsv_2[:,:,0]
#img_hsv_2[0] = np.where(hue>80/256,hue,0)
print(hue.shape, img_hsv_2.shape)


img_2 = cv.cvtColor(img_hsv_2, cv.COLOR_HSV2BGR)
cv.imshow('img_hue',hue)
cv.imshow('img_saturation',img_hsv_2[:,:,1])
cv.imshow('img_value',img_hsv_2[:,:,2])
cv.waitKey(0)
cv.destroyAllWindows()

