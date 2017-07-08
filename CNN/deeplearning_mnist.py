import numpy as np
import pandas as pd
import scipy.io
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf



#########################  settings  #########################
# number of classes
NUM_CLASSES= 10
LEARNING_RATE = 1e-4
TRAINING_ITERATIONS = 20000
image_size = 784
image_width = 28
image_height = 28
labels_count = NUM_CLASSES
DROPOUT = 0.5
BATCH_SIZE = 50
# set to 0 to train on all available data
VALIDATION_SIZE = 2000


### loading data ###

def load_data_mnist():
    data = scipy.io.loadmat("hw01_data/mnist/train.mat")
    train = data['trainX']
    test = scipy.io.loadmat("hw01_data/mnist/test.mat")
    test = test['testX']
    # shuffle training datasets
    np.random.seed(500)
    position = np.random.permutation(train.shape[0]).tolist()
    train = train[position]
    trainx = train[:, :train.shape[1] - 1]
    trainy = train[:, train.shape[1] - 1]
    # convert from [0:255] => [0.0:1.0]
    trainx = np.multiply(trainx, 1.0 / 255.0)
    test = np.multiply(test,1.0/255.00)
    return trainx,trainy,test

def one_hot(labels_train):
    y_train=np.zeros((labels_train.shape[0], NUM_CLASSES))
    for i in range(labels_train.shape[0]):
        y_train[i][labels_train[i]-1]=1
    #row of Y_train: lable k converted to a vector with value 1 on index k position and 0 elsewhere'''
    return y_train

image,labels,test=load_data_mnist()
print image.shape,labels.shape, test.shape
y = one_hot(labels)

### spliting into validatiaon and training
X_train, X_val, y_train, y_val = train_test_split(image,y,test_size=VALIDATION_SIZE, random_state=10)

# weight initialization
# weights should be initialised with a small amount of noise for symmetry breaking, and to prevent 0 gradients.
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# bias initialization,
# initialise RELU bias with a slightly positive initial bias to avoid "dead neurones"
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# convolution with zero padding
# strides = [1, stride, stride, 1].
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# pooling
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# input & output of NN
# images
x = tf.placeholder('float', shape=[None, image_size])
# labels
y_ = tf.placeholder('float', shape=[None, labels_count])


######################### architecture of nn #########################
"""
x (images) -> convolve: 32 filters of 5 by 5 filters w_conv1 + bias b_conv1 ->
ReLU -> hidden units: h_conv1 -> max pooling of 2 by 2: h_pool1 ->
convolve: 64 filters of 5 by 5 filters w_conv2 + bias b_conv2 ->
ReLU -> hidden units: h_conv2 -> max pooling of 2 by: h_pool2 ->
flatten -> fully connected by w_fc1 +b_fc1 of 1024 units-> ReLU
dropout -> fully connected by w_fc2 +b_fc2 of 10 units -> Softmax -> output
"""

#########################  first convolutional layer #########################
#  5 by 5 shape
#  stride of 1
#  32 filters, hence 32 biases
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
"""
The first layer is a convolution, followed by max pooling. 
The convolution computes 32 features for each 5x5 patch. 
Its weight tensor has a shape of [5, 5, 1, 32]. 
The first two dimensions are the patch size 
the next is the number of input channels (1 means that images are grayscale) (depth = 1)
and the last is the number of output channels.
There is also a bias vector with a component for each output channel.
"""

# reshape the image vector to an image : (n,784) => (n,28,28,1)
image = tf.reshape(x, [-1,image_width,image_height,1])
print (image.get_shape()) # =>(n,28,28,1)
h_conv1 = tf.nn.relu(conv2d(image, W_conv1) + b_conv1)
print (h_conv1.get_shape()) # => (n, 28, 28, 32)
h_pool1 = max_pool_2x2(h_conv1)
print (h_pool1.get_shape()) # => (n, 14, 14, 32)

"""
# Prepare for visualization
# display 32 fetures in 4 by 8 grid
layer1 = tf.reshape(h_conv1, (-1, image_height, image_width, 4 ,8))
# reorder so the channels are in the first dimension, x and y follow.
layer1 = tf.transpose(layer1, (0, 3, 1, 4,2))
layer1 = tf.reshape(layer1, (-1, image_height*4, image_width*8))
"""

######################### second convolutional layer #########################

"""
The second layer has 64 features for each 5x5 patch. 
Its weight tensor has a shape of [5, 5, 32, 64]. 
The first two dimensions are the patch size
the next is the number of input channels 
(32 channels correspond to 32 featured that we got from previous convolutional layer) (depth = 32 from first conv)
and the last is the number of output channels
There is also a bias vector with a component for each output channel.
"""

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
print (h_conv2.get_shape()) # => (n, 14,14, 64)
h_pool2 = max_pool_2x2(h_conv2)
print (h_pool2.get_shape()) # => (n, 7, 7, 64)

"""
# Prepare for visualization
# display 64 fetures in 4 by 16 grid
layer2 = tf.reshape(h_conv2, (-1, 14, 14, 4 ,16))
# reorder so the channels are in the first dimension, x and y follow.
layer2 = tf.transpose(layer2, (0, 3, 1, 4,2))
layer2 = tf.reshape(layer2, (-1, 14*4, 14*16))
"""


#########################  fully connected layer #########################

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

# (n, 7, 7, 64) => (n, 3136)
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
print (h_fc1.get_shape()) # => (n, 1024)


#########################  output layer #########################

# apply dropout before the output layer to prevent overfitting

# dropout
keep_prob = tf.placeholder('float')
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, labels_count])
b_fc2 = bias_variable([labels_count])

y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# lost
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

# optimization
train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)

# evaluation
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
predict = tf.argmax(y,1)

print (y.get_shape()) # => (n, 10)

epochs_completed = 0
index_in_epoch = 0
num_examples = X_train.shape[0]


# serve data by batches
def next_batch(batch_size,train_images,train_labels):
    global index_in_epoch
    global epochs_completed

    start = index_in_epoch
    index_in_epoch += batch_size

    # when all training data have been already used, it is reorder randomly
    if index_in_epoch > num_examples:
        # finished epoch
        epochs_completed += 1
        # shuffle the data
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        train_images = train_images[perm]
        train_labels = train_labels[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return train_images[start:end], train_labels[start:end]

# start TensorFlow session
init = tf.initialize_all_variables()
sess = tf.InteractiveSession()

sess.run(init)

# visualisation variables
train_accuracies = []
validation_accuracies = []
x_range = []

display_step = 1

for i in range(TRAINING_ITERATIONS):

    # get new batch
    batch_xs, batch_ys = next_batch(BATCH_SIZE,X_train,y_train)

    # check progress on every 1st,2nd,...,10th,20th,...,100th... step
    if i % display_step == 0 or (i + 1) == TRAINING_ITERATIONS:

        train_accuracy = accuracy.eval(feed_dict={x: batch_xs,
                                                  y_: batch_ys,
                                                  keep_prob: 1.0})
        if (VALIDATION_SIZE):
            validation_accuracy = accuracy.eval(feed_dict={x: X_val[0:BATCH_SIZE],
                                                           y_: y_val[0:BATCH_SIZE],
                                                           keep_prob: 1.0})
            print('training_accuracy / validation_accuracy => %.2f / %.2f for step %d' % (
            train_accuracy, validation_accuracy, i))

            validation_accuracies.append(validation_accuracy)

        else:
            print('training_accuracy => %.4f for step %d' % (train_accuracy, i))
        train_accuracies.append(train_accuracy)
        x_range.append(i)

        # increase display_step
        if i % (display_step * 10) == 0 and i:
            display_step *= 10
    # train on batch
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: DROPOUT})

# check final accuracy on validation set
if (VALIDATION_SIZE):
    validation_accuracy = accuracy.eval(feed_dict={x: X_val,
                                                   y_: y_val,
                                                   keep_prob: 1.0})
    print('validation_accuracy => %.4f' % validation_accuracy)
    plt.plot(x_range, train_accuracies, '-b', label='Training')
    plt.plot(x_range, validation_accuracies, '-g', label='Validation')
    plt.legend(loc='lower right', frameon=False)
    plt.ylim(ymax=1.1, ymin=0.6)
    plt.ylabel('accuracy')
    plt.xlabel('step')
    plt.show()