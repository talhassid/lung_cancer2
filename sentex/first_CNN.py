import os
from sentex.commands import process_data, train_neural_network

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import os  # for doing directory operations
import numpy as np
import pandas as pd  # for some simple data analysis (right now, just to load in the labels data and quickly reference it)
import tensorflow as tf

IMG_PX_SIZE = 50 #to make the slices in same size.
SLICE_COUNT = 20  #numbers of slices in each chunk.

data_dir = '/home/talhassid/PycharmProjects/input/sample_images/'
# data_dir = '/VISL2_net/talandhaim/stage1/'
patients = os.listdir(data_dir)
labels_df = pd.read_csv('/home/talhassid/PycharmProjects//input/stage1_labels.csv', index_col=0)

# much_data = []
#
# #just to know where we are, each 100 patient we will print out
# for num, patient in enumerate(patients):
#     if num%10==0:
#         print(num)
#     try:
#         img_data,label = process_data(patient,labels_df,data_dir,img_px_size=IMG_PX_SIZE, hm_slices=SLICE_COUNT)
#         #print(img_data.shape,label)
#         much_data.append([img_data,label])
#     except KeyError as e:
#         print('This is unlabeled data!')
#
# np.save('muchdata-{}-{}-{}.npy'.format(IMG_PX_SIZE,IMG_PX_SIZE,SLICE_COUNT), much_data)

n_classes = 2
keep_rate = 0.8

x = tf.placeholder('float')
y = tf.placeholder('float')

trained_network = train_neural_network(x,y,10,2)
processed_data = np.load('muchdata-50-50-20.npy')
predict_input_fn = tf.estimator.inputs.numpy_input_fn(x={"X":processed_data},num_epochs=1,shuffle=False)

predictions = list(trained_network.predict(input_fn=predict_input_fn))

print ("finish")


