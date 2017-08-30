import os
import dicom
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.ndimage

"""
    DEFINES
"""
IMG_PX_SIZE = 150
HM_SLICES = 20
IMG_SIZE_PX = 50
SLICE_COUNT = 20
MIN_BOUND = -1000.0
MAX_BOUND = 400
"""
   math functions
"""
#n-sized chunks from list l
def chunks( l,n ):
    count=0
    for i in range(0, len(l), n):
        if(count < HM_SLICES):
            yield l[i:i + n]
            count=count+1

def mean(l):
    return sum(l) / len(l)

"""
   network functions
"""




def process_data(paitent_path) : #cat
    slices = load_scan(paitent_path)
    image_3D_arr = get_pixels_hu(slices) #see what the output it. np.array of slices
    image_3D_arr,new_spacing = resample(image_3D_arr,slices, [1,1,1])
    return image_3D_arr,new_spacing

def conv3d(x, W):
    """
    Conv3D implements a form of cross-correlation.
    """
    return tf.nn.conv3d(x, W, strides=[1,1,1,1,1], padding='SAME')

def maxpool3d(x):
    """
    Performs 3D max pooling on the input.
    """
    #                        size of window         movement of window as you slide about
    return tf.nn.max_pool3d(x, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME')
#the network
def convolutional_neural_network(x,n_classes,keep_rate):
    #                # 5 x 5 x 5 patches, 1 channel, 32 features to compute.
    weights = {'W_conv1':tf.Variable(tf.random_normal([3,3,3,1,32])),
               #       5 x 5 x 5 patches, 32 channels, 64 features to compute.
               'W_conv2':tf.Variable(tf.random_normal([3,3,3,32,64])),
               #                                  64 features
               'W_fc':tf.Variable(tf.random_normal([54080,1024])),
               'out':tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
               'b_conv2':tf.Variable(tf.random_normal([64])),
               'b_fc':tf.Variable(tf.random_normal([1024])),
               'out':tf.Variable(tf.random_normal([n_classes]))}

    #                            image X      image Y        image Z
    x = tf.reshape(x, shape=[-1, IMG_SIZE_PX, IMG_SIZE_PX, SLICE_COUNT, 1])

    conv1 = tf.nn.relu(conv3d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool3d(conv1)

    # relu: Computes rectified linear: max(features, 0).
    conv2 = tf.nn.relu(conv3d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool3d(conv2)
    # Given tensor, this operation returns a tensor that has the same values as tensor with shape shape.
    fc = tf.reshape(conv2,[-1, 54080])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])
    # dropout: Computes dropout.
    # With probability keep_prob, outputs the input element scaled up by 1 / keep_prob, otherwise outputs 0.
    # The scaling is so that the expected sum is unchanged.
    fc = tf.nn.dropout(fc, keep_rate)
    # matmul: Multiplies matrix
    output = tf.matmul(fc, weights['out'])+biases['out']

    return output

def train_neural_network(y,x,n_classes,keep_rate,train_data,validation_data):
    prediction = convolutional_neural_network(x,n_classes,keep_rate)
    # reduce_mean: Computes the mean of elements across dimensions of a tensor.
    # softmax_cross_entropy_with_logits: Computes softmax cross entropy between logits and labels.
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    # minimize : Add operations to minimize loss by updating var_list.
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)

    hm_epochs = 10
    # session :A class for running TensorFlow operations.
    # A Session object encapsulates the environment in which Operation objects are executed, and Tensor objects are evaluated.
    with tf.Session() as sess:
        # run: Evaluate the tensor
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
                    print(str(e))
                    pass


            print('Epoch', epoch+1, 'completed out of',hm_epochs,'loss:',epoch_loss)

            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

            print('Accuracy:',accuracy.eval({x:[i[0] for i in validation_data], y:[i[1] for i in validation_data]}))

        print('Done. Finishing accuracy:')
        print('Accuracy:',accuracy.eval({x:[i[0] for i in validation_data], y:[i[1] for i in validation_data]}))

        print('fitment percent:',successful_runs/total_runs)


"""
   preprocessing data functions
"""


# Load the scans in given folder path and returns a list of all patient's slices, plus resizing it.
def resizing_and_loading(path_patient ,img_px_size=50, hm_slices=20): #blond

    slices = [dicom.read_file(path_patient + '/' + s) for s in os.listdir(path_patient)]
    #We're sorting by the actual image position in the scan
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    for each_slice in slices:
        each_slice.pixel_array = cv2.resize(np.array(each_slice.pixel_array),(img_px_size,img_px_size))

    new_slices = []
    chunk_sizes = math.floor(len(slices) / HM_SLICES)
    for slice_chunk in chunks(slices, chunk_sizes):
        slice_chunk = list(map(mean, zip(*slice_chunk)))
        new_slices.append(slice_chunk)

    if len(new_slices) == hm_slices-1:
        new_slices.append(new_slices[-1])

    if len(new_slices) == hm_slices-2:
        new_slices.append(new_slices[-1])
        new_slices.append(new_slices[-1])

    if len(new_slices) == hm_slices+2:
        new_val = list(map(mean, zip(*[new_slices[hm_slices-1],new_slices[hm_slices],])))
        del new_slices[hm_slices]
        new_slices[hm_slices-1] = new_val

    if len(new_slices) == hm_slices+1:
        new_val = list(map(mean, zip(*[new_slices[hm_slices-1],new_slices[hm_slices],])))
        del new_slices[hm_slices]
        new_slices[hm_slices-1] = new_val

    for each_slice in slices:
        each_slice.pixel_array = new_slices[each_slice]


    #we're resizing our images from 512x512 to 150x150
    # slices = [cv2.resize(np.array(each_slice.pixel_array),(img_px_size,img_px_size)) for each_slice in slices]
    # chunk_sizes = math.floor(len(slices) / HM_SLICES)
    # for slice_chunk in chunks(slices, chunk_sizes):
    #     slice_chunk = list(map(mean, zip(*slice_chunk)))
    #     new_slices.append(slice_chunk)
    #
    # if len(new_slices) == hm_slices-1:
    #     new_slices.append(new_slices[-1])
    #
    # if len(new_slices) == hm_slices-2:
    #     new_slices.append(new_slices[-1])
    #     new_slices.append(new_slices[-1])
    #
    # if len(new_slices) == hm_slices+2:
    #     new_val = list(map(mean, zip(*[new_slices[hm_slices-1],new_slices[hm_slices],])))
    #     del new_slices[hm_slices]
    #     new_slices[hm_slices-1] = new_val
    #
    # if len(new_slices) == hm_slices+1:
    #     new_val = list(map(mean, zip(*[new_slices[hm_slices-1],new_slices[hm_slices],])))
    #     del new_slices[hm_slices]
    #     new_slices[hm_slices-1] = new_val
    # return np.array(new_slices)

    return slices

def load_scan(path):

    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness


    return slices

#Changes the units of the pixel's to HU.
def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):

        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)
        # image[slice_number] = normalize(image[slice_number])

    return np.array(image, dtype=np.int16)

#defines a new unified spacing between pixels
def resample(image, scan, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    return image, new_spacing

#Normalize the images' pixles values
def normalize(image):
    image = (image - MIN_BOUND)/(MAX_BOUND - MIN_BOUND)
    image[image>1]=1
    image[image<0]=0
    return image

