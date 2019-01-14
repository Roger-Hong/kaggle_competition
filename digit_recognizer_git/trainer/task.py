import time
import csv
import os.path
import numpy as np
import tensorflow as tf
import pandas as pd
from pandas.compat import StringIO
from tensorflow.python.lib.io import file_io
from absl import flags

import models

TEST_LABELS_NAME = 'cnn_32_64_64_64.csv'
TRAIN_IMAGES_COUNT = 42000
TEST_IMAGES_COUNT = 28000

TRAIN_CHUNK_SIZE = 1000
TEST_CHUNK_SIZE = 1000

FLAGS = flags.FLAGS

flags.DEFINE_string('dataset_path', '', 'path to dataset')
flags.DEFINE_string('output_path', '', 'path to output directory')
flags.DEFINE_string('layer_nodes', '32_64', 'number of nodes in each layer joined by "_". E.g. "32_64"')
flags.DEFINE_integer('train_rounds', 200, 'times to use train dataset to train', lower_bound=0)
flags.DEFINE_bool('using_gpu', False, 'Whether to use GPU for training')

def read_file_iteratively(filename):
	'''
	Read a file line by line from directory DATASET_PATH.

	Args:
		filename: the name of the csv file in DATASET_PATH.

	Returns:
		An iterator to read the csv file line by line.
	'''
	file_path = os.path.join(FLAGS.dataset_path, filename)
	file_stream = file_io.FileIO(file_path, mode="r")
	return pd.read_csv(StringIO(file_stream.read()), sep=',', iterator=True)

def split_images_and_labels(train_data):
	'''
	Convert train_data with shape (:, 785) into image (:, 784) and label (:, 1).
	The format of each row in train_data is [label, image_pixels].
	'''
	num_row = len(train_data)
	images = np.asarray(train_data[:, 1:], dtype=np.int)
	labels = np.zeros([num_row, 10])
	for row_idx in range(num_row):
		labels[row_idx][int(train_data[row_idx][0])] = 1
	return images, labels

def main(_):
	if FLAGS.dataset_path == '':
		raise Exception('dataset directory path should not be empty')
        if FLAGS.output_path == '':
                raise Exception('output directory path should not be empty')
	layer_nodes = map(lambda str: int(str), FLAGS.layer_nodes.split('_'))
	print '@@@@  Model layers: %s' % layer_nodes
	x = tf.placeholder(tf.float32, [None, 784])
	y_ = tf.placeholder(tf.float32, [None, 10])
	keep_prob = tf.placeholder(tf.float32)
	y_conv = models.create_cnn_model(x, layer_nodes, keep_prob)

	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = y_, logits = y_conv))
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)	

	correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))	

	sess = tf.InteractiveSession()
	sess.run(tf.global_variables_initializer())

	print "@@@@  Start training!!"
	train_start_time = time.time()
	# Train 200 steps
	for round_idx in range(FLAGS.train_rounds):
		round_start_time = time.time()
		train_iterator = read_file_iteratively('train.csv')
		for chunk_idx in range(TRAIN_IMAGES_COUNT / TRAIN_CHUNK_SIZE):
			chunk_data = train_iterator.get_chunk(TRAIN_CHUNK_SIZE).values
			images, labels = split_images_and_labels(chunk_data)
			train_step.run(feed_dict={x: images, y_: labels, keep_prob: 0.5})
		# Test the model accuracy with the first 1000 images in train dataset.
		test_iterator = read_file_iteratively('train.csv')
		chunk_data = test_iterator.get_chunk(TRAIN_CHUNK_SIZE).values
		images, labels = split_images_and_labels(chunk_data)
		train_accuracy = accuracy.eval(feed_dict = {x: images, y_: labels, keep_prob: 1.0})
		print "@@@@  Round %d,  took %g, train_accuracy %g" % (round_idx, time.time() - round_start_time, train_accuracy)
	print "Successfully trained!!! Took %g" % (time.time() - train_start_time)
	print "=================================================="

	print "@@@@  Start evaluation!!"
	output_file_name = 'cnn_%s.csv' % FLAGS.layer_nodes
	if FLAGS.using_gpu:
		output_file_name = 'cnn_%s_GPU.csv' % FLAGS.layer_nodes
	test_start_time = time.time()
	with file_io.FileIO(os.path.join(FLAGS.output_path, output_file_name), "w") as csvfile:
		writer = csv.writer(csvfile, delimiter=',')
		writer.writerow(['ImageId', 'Label'])

	test_iterator = read_file_iteratively('test.csv')
	for chunk_idx in range (TEST_IMAGES_COUNT / TEST_CHUNK_SIZE):
		test_images = test_iterator.get_chunk(TEST_CHUNK_SIZE).values
		test_labels = y_conv.eval(feed_dict = {x: test_images, keep_prob: 1.0})

		with file_io.FileIO(os.path.join(FLAGS.output_path, output_file_name), "a") as csvfile:
			writer = csv.writer(csvfile, delimiter=',')
			for idx, labels in enumerate(test_labels, start=1):
				writer.writerow([chunk_idx * TEST_CHUNK_SIZE + idx, labels.argmax()])
	print "Successfully evaluation!!! Took %g" % (time.time() - test_start_time)
	

if __name__ == '__main__':
	tf.app.run()
