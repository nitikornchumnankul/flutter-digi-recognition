from sklearn.datasets import fetch_openml
import numpy as np
from skimage.feature import hog
from sklearn import preprocessing

# โหลด Dataset
# Fixes #11317
# I have changed the use of fetch_mldata in examples,
# but not yet in benchmarks under the assumption that those will be replaced with
# fetch_openml soon enough (even if after 0.20 release).
# Waiting on the OpenML fetcher (#11419) to be completed.
# https://github.com/scikit-learn/scikit-learn/pull/11466

# Unfortunately fetch_mldata() has been replaced in the latest version of sklearn as fetch_openml().
# So, instead of using:
# from sklearn.datasets import fetch_mldata
# mnist = fetch_mldata('MNIST original')

# You must use:
# from sklearn.datasets import fetch_openml
# mnist = fetch_openml('mnist_784')
# x = mnist.data
# y = mnist.target
# https://stackoverflow.com/questions/47324921/cant-load-mnist-original-dataset-using-sklearn
# https://www.programcreek.com/python/example/117644/sklearn.datasets.fetch_openml

# mnist_784
# https://www.openml.org/d/554
dataset = fetch_openml('mnist_784')

# Extract the features and labels
#Data Type and How to use numpy.array
#https://numpy.org/doc/stable/reference/generated/numpy.array.html?highlight=array#numpy.array
features = np.array(dataset.data, 'int16')
label = np.array(dataset.target, 'int')

# Extract the hog features
# ประกาศ list_hog_fd ให้สามารถเก็บข้อมูลในรูปแบบ array
list_hog_fd = []
# skimage.feature.hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3), block_norm='L2-Hys', visualize=False, transform_sqrt=False, feature_vector=True, multichannel=None, *, channel_axis=None)
# Extract Histogram of Oriented Gradients (HOG) for a given image. https://learnopencv.com/histogram-of-oriented-gradients/
# Compute a Histogram of Oriented Gradients (HOG) by
# (optional) global image normalization
# 1.computing the gradient image in row and col
# 2.computing gradient histograms
# 3.normalizing across blocks
# 4.flattening into a feature vector
# Parameters
# orientations    : (int, optional) => จํานวนของช่องเก็บการวางแนว
# pixels_per_cell : 2-tuple (int, int), optional => ขนาด (เป็นพิกเซล) ของเซลล์
# cells_per_block : 2-tuple (int, int), optional => จํานวนเซลล์ในแต่ละบล็อก
# visualize        : bool, optional => แสดงรูปออกมา

# https://scikit-image.org/docs/dev/api/skimage.feature.html?highlight=hog#skimage.feature.hog

# การใช้ numpy.reshape ถ้าเป็นในโค้ดด้านล่างจะแทนด้วย  features.reshape(row,col)
#Gives a new shape to an array without changing its data.

for feature in features:
    fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
    list_hog_fd.append(fd) #นำ Data เข้าไปเก็บใน list_hog_fd ในรูปแบบของ array

hog_feature = np.array(list_hog_fd,'float64') #เปลี่ยนเป็น Data Type

# Normalize the features



