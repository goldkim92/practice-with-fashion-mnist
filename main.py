#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 18:33:42 2017

@author: jm
"""

#%% import 
import argparse
import os
import tensorflow as tf
from model import network

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
tf.reset_default_graph()

#%% argument parser
# dest : 별도의 변수 지정
parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_name', dest='dataset_name', default='fashion', help='the name of the dataset')
parser.add_argument('--dataset_dir', dest='dataset_dir', default='./data', help='path of the dataset')
parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test samples are saved here')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--phase', dest='phase', default='train', help='train, test')

parser.add_argument('--image_size', dest='image_size', type=int, default=28, help='height and width of input image')
parser.add_argument('--image_nc', dest='image_nc', type=int, default=1, help='# of input image channels')
parser.add_argument('--label_n', dest='label_n', type=int, default=10, help='# of labels')
parser.add_argument('--nf', dest='nf', type=int, default=12, help='# of filters in first conv layer')

parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--epoch', dest='epoch', type=int, default=10, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=128, help='# of batch_size')

args = parser.parse_args()

#%%
def main(_):
    dataset_dir = os.path.join(args.dataset_dir, args.dataset_name)
    test_dir = os.path.join(args.test_dir, args.dataset_name)
    checkpoint_dir = os.path.join(args.checkpoint_dir, args.dataset_name)    

    # make directory if not exist
    try: os.makedirs(test_dir)
    except: pass
    try: os.makedirs(checkpoint_dir)
    except: pass

    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True
    with tf.Session(config=tfconfig) as sess:
        model = network(sess,args)
        model.train(args) if args.phase == 'train' else model.test(args)
        
if __name__ == '__main__':
    tf.app.run()
    