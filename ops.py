import tensorflow as tf
import tensorflow.contrib.slim as slim

def conv2d(input_, output_dim, ks=3,s=1,padding='SAME',name='conv2d'):
    with tf.variable_scope(name):
        return slim.conv2d(input_, output_dim, ks, s, padding=padding,
                           weights_initializer=tf.contrib.layers.xavier_initializer())
        
def maxpool2d(input_, ks=2, s=2, padding='VALID',name='maxpool2d'):
    with tf.variable_scope(name):
        return slim.max_pool2d(input_, ks, s, padding=padding)
    
def fully_connected(input_, num_outputs, name='fc'):
    with tf.variable_scope(name):
        return slim.fully_connected(input_, num_outputs,
                                    weights_initializer=tf.contrib.layers.xavier_initializer())