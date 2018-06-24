
import tensorflow as tf
import input_data
import sys
import os
import itertools
import re
import time
from random import randint

train_images_file = ""
train_labels_file = ""
test_images_file = ""
test_labels_file = ""

# Parameters
learning_rate = 0.001
training_iters = 200000
batch_size = 128
display_step = 10

model_path = os.environ["RESULT_DIR"]+"/model"

# This helps distinguish instances when the training job is restarted.
instance_id = randint(0,9999)

def main(argv):

    if len(argv) < 12:
        sys.exit("Not enough arguments provided.")

    global train_images_file, train_labels_file, test_images_file, test_labels_file, learning_rate, training_iters

    i = 1
    while i <= 12:
        arg = str(argv[i])
        if arg == "--trainImagesFile":
            train_images_file = str(argv[i+1])
        elif arg == "--trainLabelsFile":
            train_labels_file = str(argv[i+1])
        elif arg == "--testImagesFile":
            test_images_file = str(argv[i+1])
        elif arg == "--testLabelsFile":
            test_labels_file = str(argv[i+1])
        elif arg == "--learningRate":
            learning_rate = float(argv[i+1])
        elif arg =="--trainingIters":
            training_iters = int(argv[i+1])
        i += 2

if __name__ == "__main__":
    main(sys.argv)


x_train_1st = trainImagesFile
y_train_1st = trainLabelsFile

x_test = testImagesFile
y_test = testLabelsFile

enc = OneHotEncoder()

enc.fit(y_train_1st)
y_train_1st=enc.transform(train_images_file)
y_train_1st=pd.SparseDataFrame([pd.SparseSeries(y_train_1st[i].toarray().ravel()) for i in np.arange(y_train_1st.shape[0])])

enc.fit(y_test)
y_test=enc.transform(test_images_file)
y_test=pd.SparseDataFrame([pd.SparseSeries(y_test[i].toarray().ravel()) for i in np.arange(y_test.shape[0])])

x_train = x_train_1st.as_matrix()
y_train = y_train_1st.as_matrix()

x_test = x_test.as_matrix()
y_test = y_test.as_matrix()

def next_batch(num, data, labels):
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]
    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b) # the equation

y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

batch_size = 100
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# train the model mini batch with 100 elements, for 1K times
for i in range(1000):
    
    batch_x, batch_y = next_batch(batch_size, x_vals, y_vals)
    sess.run(train_step, feed_dict={x: batch_x, y_: batch_y})


correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
start_time = time.time()
test_accu = sess.run(accuracy, feed_dict={x: x_test, y_: y_test})
end_time = time.time()



print('Accuracy: {}\n'.format(test_accu))
print('Inference time: {}\n'.format(end_time - start_time))


