import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from collections import namedtuple
from tqdm import tqdm

import module
import util

class network(object):
    
    def __init__(self, sess, args):
        self.sess = sess
        self.dataset_dir = args.dataset_dir
        self.checkpoint_dir = args.checkpoint_dir
        self.image_size = args.image_size
        self.image_nc = args.image_nc
        self.label_n = args.label_n
        self.lr = args.lr
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        
        OPTIONS = namedtuple('OPTIONS','image_size image_nc, nf n_pred')
        self.options = OPTIONS._make((self.image_size, self.image_nc, args.nf, 10))
        
        self.build_model()
        self.saver = tf.train.Saver()
        
    
    def build_model(self):
        self.input_images = tf.placeholder(tf.float32, 
                                          [None,self.image_size,self.image_size,self.image_nc],
                                          name='input_images')
        self.labels = tf.placeholder(tf.float32, [None,self.label_n], name='labels')
        
        # loss funciton
        self.pred = module.densenet(self.input_images, self.options, reuse=False, name='densenet')
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.pred))
        
        # accuracy
        corr = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.labels, 1))    
        self.accr = tf.reduce_mean(tf.cast(corr, "float"))
   
        # summary
        self.sum_accr = tf.summary.scalar('accuracy',self.accr)
        self.summary = tf.summary.merge([self.sum_accr])
        
        # print trainable variables
        t_vars = tf.trainable_variables()
        self.module_vars = [var for var in t_vars if 'densenet' in var.name]
        for var in t_vars: print(var.name)
    
    def train(self, args):
        self.optim = tf.train.AdamOptimizer(self.lr).minimize(self.loss, var_list=self.module_vars)

        # load train data
        self.load_data()
        
        # summary
        self.summary_loss = tf.summary.scalar('loss',self.loss)
        self.writer = tf.summary.FileWriter('./log/fashion', self.sess.graph)
        
        self.sess.run(tf.global_variables_initializer())        
        
#        start_time = time.time()
        
        total_batch = math.ceil(self.n_train/self.batch_size)
        
        for epoch in range(self.epoch):
            
            idx_permute_list = np.random.permutation(self.n_train)
            cost = 0.
            
            for i in tqdm(range(total_batch)):
                
                # get batch images and labels
                idx_rand  = idx_permute_list[i*self.batch_size:min((i+1)*self.batch_size, self.n_train-1)]
                batch_images = self.X_train[idx_rand, :]
                batch_labels = self.y_train[idx_rand, :] 
                
                # update network
                feeds = {self.input_images: batch_images, self.labels: batch_labels}
                self.sess.run(self.optim, feed_dict=feeds)
                cost += self.sess.run(self.loss, feed_dict=feeds)
                
            avg_cost = cost / total_batch
           
            # DISPLAY & SAVE
            disp_each = 1
            if (epoch+1) % disp_each == 0 or epoch == epoch-1:
                print ("Epoch: %03d/%03d, Cost: %f" % (epoch+1, epoch, avg_cost))
                feeds = {self.input_images: batch_images, self.labels: batch_labels}
                train_acc = self.sess.run(self.accr, feed_dict=feeds)
                print (" TRAIN ACCURACY: %.3f" % (train_acc))
                feeds = {self.input_images: self.X_test, self.labels: self.y_test}
#                test_acc, summary = self.sess.run([self.accr, self.summary], feed_dict=feeds)
#                test_acc = self.sess.run(self.accr, feed_dict=feeds)
#                print (" TEST ACCURACY: %.3f" % (test_acc))
#                self.writer.add_summary(summary, epoch+1)
                
                self.save(self.checkpoint_dir, epoch+1)
        
        
    def load_data(self):
        self.X_train, self.y_train = util.load_mnist(self.dataset_dir, kind='train')
        self.n_train = np.size(self.X_train, 0)
        
        self.X_test, self.y_test = util.load_mnist(self.dataset_dir, kind='t10k')
        self.n_test = np.size(self.X_test, 0)

    def save(self, checkpoint_dir,step):
        model_name = "densenet.model"
#        model_dir = "%s_%s" % (self.dataset_dir, self.image_size)
        checkpoint_dir = os.path.join(checkpoint_dir, self.dataset_dir)

        try: os.makedirs(checkpoint_dir)
        except: pass

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)
        