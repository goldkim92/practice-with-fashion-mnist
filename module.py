import tensorflow as tf
from ops import conv2d, maxpool2d, fully_connected
  
 
def densenet(image, options, reuse=False,name='densenet'):
    
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
    
    h_conv1 = conv2d(image, options.nf, name='conv1')
    h_db1 = denseblock(h_conv1, options, name='db1')    
    h_maxpool1 = maxpool2d(h_db1, name='pool1')
    h_db2 = denseblock(h_maxpool1, options, name='db2')
    
    h_flat = tf.reshape(h_db2, [-1,tf.size(h_db2)])
    h_fc1 = fully_connected(h_flat, options.nf * options.nf, name='fc1')
    h_fc2 = fully_connected(h_fc1, options.n_pred, name='fc2')
    
    return h_fc2
  
    
def denseblock(input_, options, reuse=False, name='denseblock'):
    
    with tf.variable_scope(name):
        h_conv1 = conv2d(input_, options.nf, name='h_conv1')
        h_conv2 = conv2d(tf.concat((input_,h_conv1),axis=3), options.nf, name='h_conv2')
        h_conv3 = conv2d(tf.concat((input_,h_conv1,h_conv2),axis=3), options.nf, name='h_conv3')
        h_conv4 = conv2d(tf.concat((input_,h_conv1,h_conv2,h_conv3),axis=3), options.nf, name='h_conv4')
        h_conv5 = conv2d(tf.concat((input_,h_conv1,h_conv2,h_conv3,h_conv4),axis=3), options.nf, name='h_conv5')
        
        return h_conv5
     
    
    