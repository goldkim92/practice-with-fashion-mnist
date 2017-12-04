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
        self.dataset_name = args.dataset_name
        self.dataset_dir = os.path.join(args.dataset_dir, args.dataset_name)
        self.test_dir = os.path.join(args.test_dir, args.dataset_name)
        self.checkpoint_dir = os.path.join(args.checkpoint_dir, args.dataset_name)
        self.image_size = args.image_size
        self.image_nc = args.image_nc
        self.label_n = args.label_n
        self.lr = args.lr
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.continue_train = args.continue_train
        
        OPTIONS = namedtuple('OPTIONS',['image_size', 'image_nc', 'nf', 'n_pred'])
        self.options = OPTIONS(self.image_size, self.image_nc, args.nf, 10)
        
        self.build_model()
        self.summary()
    
    def build_model(self):
        # placeholder
        self.input_images = tf.placeholder(tf.float32, 
                                          [None,self.image_size,self.image_size,self.image_nc],
                                          name='input_images')
        self.labels = tf.placeholder(tf.float32, [None,self.label_n], name='labels')
        
        # loss funciton
        self.pred = module.densenet(self.input_images, self.options, reuse=False, name='densenet')
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.pred))
        
        # accuracy
        corr = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.labels, 1))    
        self.accr_count = tf.reduce_sum(tf.cast(corr, "float"))
        
        #placeholder for summary
        self.total_loss = tf.placeholder(tf.float32)
        self.accr = tf.placeholder(tf.float32)
          
        # print trainable variables
        t_vars = tf.trainable_variables()
        self.module_vars = [var for var in t_vars if 'densenet' in var.name]
        for var in t_vars: print(var.name)
    
    
    def summary(self):
        self.saver = tf.train.Saver()
        
        # summary writer
        self.writer_cost = tf.summary.FileWriter(os.path.join('.','log',self.dataset_name,'cost'), self.sess.graph)
        self.writer_train_accr = tf.summary.FileWriter(os.path.join('.','log',self.dataset_name,'train_accr'),self.sess.graph)
        self.writer_test_accr = tf.summary.FileWriter(os.path.join('.','log',self.dataset_name,'test_accr'),self.sess.graph)        
        
        # summary session
        self.sum_cost = tf.summary.scalar('cost function', self.total_loss)
        self.sum_accr = tf.summary.scalar('accuaracy', self.accr)
#        self.summary = tf.summary.merge([self.sum_train_accr, self.sum_test_accr])
#        self.summary = tf.summary.merge_all()
        
        
    def train(self, args):
        # load train data
        self.load_data()
        
        self.optim = tf.train.AdamOptimizer(self.lr).minimize(self.loss, var_list=self.module_vars)
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
            if (epoch+1) % disp_each == 0 or epoch == self.epoch-1:
                # cost and accuracy
                print ("Epoch: %03d/%03d, Cost: %f" % (epoch+1, self.epoch, avg_cost))
                train_accr = self.accuracy('train')
                test_accr = self.accuracy('test')
                
                # summary
                summary = self.sess.run(self.sum_cost, feed_dict={self.total_loss:avg_cost})
                self.writer_cost.add_summary(summary, epoch+1)
                
                summary = self.sess.run(self.sum_accr, feed_dict={self.accr:train_accr})
                self.writer_train_accr.add_summary(summary, epoch+1)
                
                summary = self.sess.run(self.sum_accr, feed_dict={self.accr:test_accr})
                self.writer_test_accr.add_summary(summary, epoch+1)
                
                # checkpoint
#                self.checkpoint_save(epoch+1)

    
    def accuracy(self, phase='test', batch_size=100):
        # train or test or validate
        Dataset = namedtuple('Dataset',['X_', 'y_', 'n_'])
        if phase=='train':
            dataset = Dataset(self.X_train, self.y_train, self.n_train)
        elif phase=='test':
            dataset = Dataset(self.X_test, self.y_test, self.n_test)

        # accuracy
        accr = 0.    
        for i in range(0, int(dataset.n_ / batch_size)):
            feeds = {
                    self.input_images: dataset.X_[i*batch_size : (i+1)*batch_size],
                    self.labels: dataset.y_[i*batch_size : (i+1)*batch_size]
                    }
            accr += self.sess.run(self.accr_count, feed_dict=feeds)
        
        accr = accr / dataset.n_
        print(" %s ACCURACY: %.3f" % (phase.upper(), accr))    
        
        return accr
    
    
    def load_data(self):
        self.X_train, self.y_train = util.load_mnist(self.dataset_dir, kind='train')
        self.n_train = np.size(self.X_train, 0)
        
        self.X_test, self.y_test = util.load_mnist(self.dataset_dir, kind='t10k')
        self.n_test = np.size(self.X_test, 0)


    def checkpoint_save(self, step):
        model_name = "densenet.model"
        
        self.saver.save(self.sess,
                        os.path.join(self.checkpoint_dir, model_name),
                        global_step=step)
    
    
    def checkpoint_load(self):
        print(" [*] Reading checkpoint...")
        
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(self.checkpoint_dir, ckpt_name))
            return True
        else:
            return False
        