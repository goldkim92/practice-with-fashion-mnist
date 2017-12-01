import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from collections import namedtuple
from tqdm import tqdm


class network(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.dataset_dir = args.dataset_dir
        self.fine_size = args.fine_size
        
        OPTIONS = namedtuple('OPTIONS','nf n_pred')
        self.options = OPTIONS._make((args.nf, 10))
        
        self.build_model()
        self.saver = tf.train.Saver()
        
    def build_model(self):
        