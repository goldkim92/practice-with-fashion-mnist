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

#%% argument parser
# dest : 별도의 변수 지정
parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_dir', dest='dataset_dir', default='./data/fasion', help='path of the dataset')
parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test samples are saved here')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--phase', dest='phase', default='train', help='train, test')
parser.add_argument('--input_nc', dest='input_nc', type=int, default=3, help='# of input image channels')
parser.add_argument('--output_nc', dest='output_nc', type=int, default=3, help='# of output image channels')
parser.add_argument('--nf', dest='nf', type=int, default=12, help='# of filters in first conv layer')


args = parser.parse_args()

#%%
def main():
    # make directory if not exist
    try: os.makedirs(args.test_dir)
    except: pass
    try: os.makedirs(args.checkpoint_dir)
    except: pass

    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True
    with tf.Session(config=tfconfig) as sess:
        model = network(sess,args)
        model.train(args) if args.phase == 'train' else model.test(args)
        
if __name__ == '__main__':
    tf.app.run()
    