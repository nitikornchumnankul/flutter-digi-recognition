from sklearn.datasets import fetch_openml

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
dataset = fetch_openml('mnist_784')

# Extract the features and labels