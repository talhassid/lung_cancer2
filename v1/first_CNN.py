#In[1]#################################################################################################################################################

import cv2
import dicom  # for reading dicom files
import math
import os  # for doing directory operations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # for some simple data analysis (right now, just to load in the labels data and quickly reference it)
import tensorflow as tf

IMG_PX_SIZE = 50 #to make the slices in same size.
SLICE_COUNT = 20  #numbers of slices in each chunk.

def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1,1,1,1,1], padding='SAME')

def maxpool3d(x):
    #                        size of window         movement of window as you slide about
    return tf.nn.max_pool3d(x, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME')

def convolutional_neural_network(x):
    #                # 3 x 3 x 3 patches, 1 channel, 32 features to compute.
    weights = {'W_conv1':tf.Variable(tf.random_normal([3,3,3,1,32])),
               #       3 x 3 x 3 patches, 32 channels, 64 features to compute.
               'W_conv2':tf.Variable(tf.random_normal([3,3,3,32,64])),
               #                                  64 features
               'W_fc':tf.Variable(tf.random_normal([54080,1024])),
               'out':tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
               'b_conv2':tf.Variable(tf.random_normal([64])),
               'b_fc':tf.Variable(tf.random_normal([1024])),
               'out':tf.Variable(tf.random_normal([n_classes]))}

    #                            image X      image Y        image Z
    x = tf.reshape(x, shape=[-1, IMG_PX_SIZE, IMG_PX_SIZE, SLICE_COUNT, 1])

    conv1 = tf.nn.relu(conv3d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool3d(conv1)


    conv2 = tf.nn.relu(conv3d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool3d(conv2)

    fc = tf.reshape(conv2,[-1, 54080])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out'])+biases['out']

    return output

def train_neural_network(x):
    much_data = np.load('muchdata-50-50-20.npy')
    train_data = much_data[:-100] #2 for sampleimages and 100 for stage1
    validation_data = much_data[-100:]

    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)

    hm_epochs = 30
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        successful_runs = 0
        total_runs = 0

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for data in train_data:
                total_runs += 1
                try:
                    X = data[0]
                    Y = data[1]
                    _, c = sess.run([optimizer, cost], feed_dict={x: X, y: Y})
                    epoch_loss += c
                    successful_runs += 1
                except Exception as e:
                    pass

            print('Epoch', epoch+1, 'completed out of',hm_epochs,'loss:',epoch_loss)

            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

            print('Accuracy:',accuracy.eval({x:[i[0] for i in validation_data], y:[i[1] for i in validation_data]}))

        print('Done. Finishing accuracy:')
        print('Accuracy:',accuracy.eval({x:[i[0] for i in validation_data], y:[i[1] for i in validation_data]}))

        print('fitment percent:',successful_runs/total_runs)

# data_dir = '/home/talhassid/PycharmProjects/input/sample_images/'
data_dir = '/VISL2_net/talandhaim/stage1/'
patients = os.listdir(data_dir)
labels_df = pd.read_csv('/home/talhassid/PycharmProjects//input/stage1_labels.csv', index_col=0)

def chunks(l, n):
#creates l sized chunks from list n. seperating list to lists.
    n=int(n)
    for i in range(0, len(l), n):
        yield l[i:i + n]

def mean(l):
#mean of a list
    return sum(l) / len(l)

def process_data(patient,labels_df,img_px_size=50,hm_slices=20):
    label = labels_df.get_value(patient, 'cancer') #the value for the cancer column
    path = data_dir + patient
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2])) # sorting the dicom by x image position

    new_slices = []

    slices = [cv2.resize(np.array(each_slice.pixel_array),(IMG_PX_SIZE,IMG_PX_SIZE)) for each_slice in slices]

    chunk_number = math.ceil(len(slices) / SLICE_COUNT) #number of chunks

    for slice_chunk in chunks(slices, chunk_number):
        slice_chunk = list(map(mean, zip(*slice_chunk)))
        new_slices.append(slice_chunk)

    if len(new_slices) == SLICE_COUNT-1:
        new_slices.append(new_slices[-1])

    if len(new_slices) == SLICE_COUNT-2:
        new_slices.append(new_slices[-1])
        new_slices.append(new_slices[-1])

    if len(new_slices) == SLICE_COUNT+2:
        new_val = list(map(mean, zip(*[new_slices[SLICE_COUNT-1],new_slices[SLICE_COUNT],])))
        del new_slices[SLICE_COUNT]
        new_slices[SLICE_COUNT-1] = new_val

    if len(new_slices) == SLICE_COUNT+1:
        new_val = list(map(mean, zip(*[new_slices[SLICE_COUNT-1],new_slices[SLICE_COUNT],])))
        del new_slices[SLICE_COUNT]
        new_slices[SLICE_COUNT-1] = new_val

    #left column nocancer,right column cancer
    if label == 1: label=np.array([0,1])
    elif label == 0: label=np.array([1,0])

    return np.array(new_slices), label

much_data = []

#just to know where we are, each 100 patient we will print out
for num, patient in enumerate(patients):
    if num%10==0:
        print(num)
    try:
        img_data,label = process_data(patient,labels_df,img_px_size=IMG_PX_SIZE, hm_slices=SLICE_COUNT)
        #print(img_data.shape,label)
        much_data.append([img_data,label])
    except KeyError as e:
        print('This is unlabeled data!')

np.save('muchdata-{}-{}-{}.npy'.format(IMG_PX_SIZE,IMG_PX_SIZE,SLICE_COUNT), much_data)

n_classes = 2
keep_rate = 0.8

x = tf.placeholder('float')
y = tf.placeholder('float')

train_neural_network(x)



