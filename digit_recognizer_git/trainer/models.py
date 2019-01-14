# coding:utf-8
import tensorflow as tf

#############################
## CNN model               ##
#############################

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev = 0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape = shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

def add_one_cnn_layer(x, kernel_h, kernel_w, channel_num, output_node, pooling):
  w_conv = weight_variable([kernel_h, kernel_w, channel_num, output_node])
  b_conv = bias_variable([output_node])
  h_conv = tf.nn.relu(conv2d(x, w_conv) + b_conv)
  if not pooling:
    return h_conv
  return max_pool_2x2(h_conv)

def create_cnn_model(x, layer_nodes_list, keep_prob):
  '''
  Create a multiple layer cnn model for images of size 28 * 28. It is used for
  trainning on minst dataset.

  Args: 
    x: input images tensor.
    layer_nodes_list: a list of integers, representing the number of output
    nodes of each layer.
    keep_prob: the dropout rate of the second last full connection layer.
  Returns:
    A tensorflow graph representing the cnn model.
  '''
  x_images = tf.reshape(x, [-1, 28, 28, 1])
  internal = x_images

  # Convolution layers.
  for layer_idx, num_nodes in enumerate(layer_nodes_list):
    input_channels = 1
    if layer_idx > 0:
      input_channels = layer_nodes_list[layer_idx - 1]
    internal = add_one_cnn_layer(internal, 5, 5, input_channels, num_nodes, layer_idx < 2)

  # First full-connection layer
  w_fc1 = weight_variable([7 * 7 * layer_nodes_list[-1], 1024])
  b_fc1 = bias_variable([1024])
  h_pool1_flat = tf.reshape(internal, [-1, 7 * 7 * layer_nodes_list[-1]])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, w_fc1) + b_fc1)
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Second full-connection layer
  W_fc2 = weight_variable([1024, 10])
  b_fc2 = bias_variable([10])
  y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
  return y_conv
