import os
import gzip
import numpy as np


def load_mnist(path, kind='train'):

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
    labels = one_hot(labels)
    
    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)
    images = np.reshape(images, [-1,28,28,1])        
    return images, labels


def one_hot(labels):
    
    onehot = np.zeros([len(labels),10])
    for (i,val) in enumerate(labels) : 
        onehot[i,val] = 1
    return onehot