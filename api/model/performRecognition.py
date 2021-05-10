
import argparse as ap
import joblib
import cv2
import numpy as np
from skimage.feature import hog
import os

#Get the path of the training set
# Argparse Tutorial¶ https://docs.python.org/3/howto/argparse.html
# create file a.py is like 'ls' that is for show on recent directory
parser = ap.ArgumentParser()
# FileType objects¶
parser.add_argument("-c","--classiferPath", help="Path to Classifier File", required="True")
parser.add_argument("-i", "--image", help="Path to Image", required="True")

# class argparse.Namespace
# Simple class used by default by parse_args() to create an object holding attributes and return it.
# This class is deliberately simple,
# just an object subclass with a readable string representation.
# If you prefer to have dict-like view of the attributes,
# you can use the standard Python idiom, vars():

args = vars(parser.parse_args())

# Load the classifier
# joblib.load(filename, mmap_mode=None
# Parameters:	filename: str, pathlib.Path, or file object.
# The file object or path of the file from which to load the object
clf , pp = joblib.load(args["classiferPath"])

# Read the input image
# loads an image from a file.
# The function imread loads an image from the specified file and returns it.
# If the image cannot be read (because of missing file, improper permissions, unsupported or invalid format),
# the function returns an empty matrix ( Mat::data==NULL ).
# Currently, the following file formats are supported:
# Windows bitmaps - *.bmp, *.dib (always supported)
# JPEG files - *.jpeg, *.jpg, *.jpe (see the Note section)
# JPEG 2000 files - *.jp2 (see the Note section)
# Portable Network Graphics - *.png (see the Note section)
# WebP - *.webp (see the Note section)
# Portable image format - *.pbm, *.pgm, *.ppm *.pxm, *.pnm (always supported)
# PFM files - *.pfm (see the Note section)
# Sun rasters - *.sr, *.ras (always supported)
# TIFF files - *.tiff, *.tif (see the Note section)
# OpenEXR Image files - *.exr (see the Note section)
# Radiance HDR - *.hdr, *.pic (always supported)
# Raster and Vector geospatial data supported by GDAL (see the Note section)
# https://docs.opencv.org/master/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56
im = cv2.imread(args["image"])
cv2.namedWindow("TEST", cv2.WINDOW_NORMAL)
cv2.imshow("TEST", im)
cv2.waitKey()
# Convert to grayscale and apply Gaussian filtering
# cvtColor()
# Converts an image from one color space to another.
# cvtColor(img, img, COLOR_BGR2Luv);
# Parameters
# src	input image: 8-bit unsigned, 16-bit unsigned ( CV_16UC... ), or single-precision floating-point
# dst	output image of the same size and depth as src.
# code	color space conversion code (see ColorConversionCodes https://docs.opencv.org/master/d8/d01/group__imgproc__color__conversions.html#ga4e0972be5de079fed4e3a10e24ef5ef0).
# dstCn	number of channels in the destination image; if the parameter is 0, the number of the channels is derived automatically from src and code
# ColorConversionCodes
# https://docs.opencv.org/master/d8/d01/group__imgproc__color__conversions.html#ga4e0972be5de079fed4e3a10e24ef5ef0

im_gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
cv2.namedWindow("Gray", cv2.WINDOW_NORMAL)
cv2.imshow("Gray", im_gray)
cv2.waitKey()

# GaussianBlur()
# cv.GaussianBlur(	src, ksize, sigmaX[, dst[, sigmaY[, borderType]]]	)
# Parameters
# src	    input image; the image can have any number of channels, which are processed independently, but the depth should be CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.
#
# dst	    output image of the same size and type as src.
#
# ksize	    Gaussian kernel size. ksize.width and ksize.height can differ but they both must be positive and odd. Or, they can be zero's and then they are computed from sigma
#
# sigmaX	Gaussian kernel standard deviation in X direction.
#
# sigmaY	Gaussian kernel standard deviation in Y direction;
#           if sigmaY is zero, it is set to be equal to sigmaX,
#           if both sigmas are zeros, they are computed from ksize.width and ksize.height,
#           respectively (see getGaussianKernel(https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#gac05a120c1ae92a6060dd0db190a61afa) for details);
#           to fully control the result regardless of possible future modifications of all this semantics,
#           it is recommended to specify all of ksize, sigmaX, and sigmaY.\
#
# borderType	pixel extrapolation method, see BorderTypes. BORDER_WRAP is not supported.
# https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#gaabe8c836e97159a9193fb0b11ac52cf1
im_gray = cv2.GaussianBlur(im_gray,(5,5), 0)
cv2.namedWindow("GaussianBlur", cv2.WINDOW_NORMAL)
cv2.imshow("GaussianBlur", im_gray)
cv2.waitKey()
# Threshold the image
# ◆ threshold()
# Applies a fixed-level threshold to each array element.
# Parameters
# src	input array (multiple-channel, 8-bit or 32-bit floating point).

# dst	output array of the same size and type and the same number of channels as src.

# thresh	threshold value.

# maxval	maximum value to use with the THRESH_BINARY(https://docs.opencv.org/master/d7/d1b/group__imgproc__misc.html#ggaa9e58d2860d4afa658ef70a9b1115576a147222a96556ebc1d948b372bcd7ac59)
#           and THRESH_BINARY_INV(https://docs.opencv.org/master/d7/d1b/group__imgproc__misc.html#ggaa9e58d2860d4afa658ef70a9b1115576a19120b1a11d8067576cc24f4d2f03754) thresholding types.

# type	    thresholding type (see ThresholdTypes https://docs.opencv.org/master/d7/d1b/group__imgproc__misc.html#gaa9e58d2860d4afa658ef70a9b1115576).
# https://docs.opencv.org/master/d7/d1b/group__imgproc__misc.html#gae8a4a146d1ca78c626a53577199e9c57


ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)
cv2.namedWindow("threshold", cv2.WINDOW_NORMAL)
cv2.imshow("threshold", ret)
cv2.waitKey()

# Find contours in the image
# findContours()
# Finds contours in a binary image.

# The function retrieves contours from the binary image using the algorithm [233](https://docs.opencv.org/master/d0/de3/citelist.html#CITEREF_Suzuki85) .
# The contours are a useful tool for shape analysis and object detection and recognition. See squares.cpp in the OpenCV sample directory

# cv.findContours(image, mode, method[, contours[, hierarchy[, offset]]]	)
# Parameters
# image	      Source, an 8-bit single-channel image. Non-zero pixels are treated as 1's. Zero pixels remain 0's, so the image is treated as binary .
#             You can use compare, inRange, threshold , adaptiveThreshold, Canny, and others to create a binary image out of a grayscale or color one.
#             If mode equals to RETR_CCOMP or RETR_FLOODFILL, the input can also be a 32-bit integer image of labels (CV_32SC1).

# contours	  Detected contours. Each contour is stored as a vector of points (e.g. std::vector<std::vector<cv::Point> >).
# hierarchy	   Optional output vector (e.g. std::vector<cv::Vec4i>), containing information about the image topology. It has as many elements as the number of contours. For each i-th contour contours[i], the elements hierarchy[i][0] , hierarchy[i][1] , hierarchy[i][2] , and hierarchy[i][3] are set to 0-based indices in contours of the next and previous contours at the same hierarchical level, the first child contour and the parent contour, respectively. If for the contour i there are no next, previous, parent, or nested contours, the corresponding elements of hierarchy[i] will be negative.
# mode	       Contour retrieval mode, see RetrievalModes
# method	Contour approximation method, see ContourApproximationModes
# offset	Optional offset by which every contour point is shifted. This is useful if the contours are extracted from the image ROI and then they should be analyzed in the whole image context.
# https://docs.opencv.org/master/d3/dc0/group__imgproc__shape.html#gadf1ad6a0b82947fa1fe3c3d497f260e0

# RETR_EXTERNAL
# Python: cv.RETR_EXTERNA : retrieves only the extreme outer contours. It sets hierarchy[i][2]=hierarchy[i][3]=-1 for all the contours.

# CHAIN_APPROX_SIMPLE
# Python: cv.CHAIN_APPROX_SIMPLE

ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cv2.namedWindow("findContours", cv2.WINDOW_NORMAL)
cv2.imshow("findContours", im_th.copy())
cv2.waitKey()

# Get rectangles contains each contour
# cv.boundingRect(array)
# Calculates the up-right bounding rectangle of a point set or non-zero pixels of gray-scale image.
# The function calculates and returns the minimal up-right bounding rectangle for the specified point set or non-zero pixels of gray-scale image.
# Parameters
# array:	Input gray-scale image or 2D point set, stored in std::vector or Mat(https://docs.opencv.org/master/d3/d63/classcv_1_1Mat.html).

# https://docs.opencv.org/master/d3/dc0/group__imgproc__shape.html#ga103fcbda2f540f3ef1c042d6a9b35ac7
rects = [cv2.boundingRect(ctr) for ctr in ctrs]
print(rects)


# For each rectangular region, calculate HOG features and predict
# the digit using Linear SVM.
for rect in rects:
    # Draw the rectangles
    # Python OpenCV | cv2.rectangle() method Tutorial from GreekforGreek
    # image = cv2.rectangle(image, start_point, end_point, color, thickness)
    # https://www.geeksforgeeks.org/python-opencv-cv2-rectangle-method/
    cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
    #####
    cv2.namedWindow("Rectangle" + str(rect[0]), cv2.WINDOW_NORMAL)
    cv2.imshow("Rectangle" + str(rect[0]), cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3))
    cv2.waitKey()

    # Make the rectangular region around the digit
    leng = int(rect[3] * 1.6)
    pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
    pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
    roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
    # Resize the image
    #OpenCV Resize image using cv2.resize()
    # The syntax of resize function in OpenCV is
    # cv2.resize(src, dsize[, dst[, fx[, fy[, interpolation]]]])
    # https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/

    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA) #มีนัยสำคัญต่อการวิเคราะห์
    # print(roi)

    # Morphological Operations
    # https://docs.opencv.org /3.4/db/df6/tutorial_erosion_dilatation.html
    # Dilates an image by using a specific structuring element.
    # dilate()
    # https://docs.opencv.org/3.4/d4/d86/group__imgproc__filter.html
    roi = cv2.dilate(roi, (3, 3))
    # Calculate the HOG features
    roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualize=False)
    roi_hog_fd = pp.transform(np.array([roi_hog_fd], 'float64'))

    nbr = clf.predict(roi_hog_fd)
    # Python OpenCV | cv2.putText() method Tutorials from GreekforGreek
    # https://www.geeksforgeeks.or /python-opencv-cv2-puttext-method/
    # image = cv2.putText(image, text, org, font, fontScale, color, thickness, cv2.LINE_AA, False)
    cv2.putText(im, str(int(nbr[0])), (rect[0], rect[1]), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)

cv2.namedWindow("Resulting Image with Rectangular ROIs", cv2.WINDOW_NORMAL)
cv2.imshow("Resulting Image with Rectangular ROIs", im)
# Python OpenCV | cv2.imwrite() method Tutorials from GreeksforGreeks
# https://www.geeksforgeeks.org/python-opencv-cv2-imwrite-method/
# Document cv2.imwrite()
# https://docs.opencv.org/3.4/d4/da8/group__imgcodecs.html#gabbc7ef1aa2edfaa87772f1202d67e0ce
cv2.imwrite('result.jpg', cv2.putText(im, str(int(nbr[0])), (rect[0], rect[1]), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3))
cv2.waitKey()

