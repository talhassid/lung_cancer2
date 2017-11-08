import os
from sentex.commands import process_data, train_neural_network, load_process_data, test

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import os  # for doing directory operations
import numpy as np
import pandas as pd  # for some simple data analysis (right now, just to load in the labels data and quickly reference it)


IMG_PX_SIZE = 50 #to make the slices in same size.
SLICE_COUNT = 20  #numbers of slices in each chunk.
EPOCHS_COUNT = 10
VALIDATION_COUNT = 2 #hm patients will be in validation

data_dir = '/home/talhassid/PycharmProjects/input/sample_images/'
# data_dir = '/VISL2_net/talandhaim/stage1/'
patients = os.listdir(data_dir)
labels_df = pd.read_csv('/home/talhassid/PycharmProjects//input/stage1_labels.csv', index_col=0)
# load_process_data(patients,labels_df,data_dir)
prediction = train_neural_network(epochs_count=EPOCHS_COUNT,validation_count=VALIDATION_COUNT)
processed_data = np.load('muchdata-50-50-20.npy')
# predict_input_fn = tf.estimator.inputs.numpy_input_fn(x={"X":processed_data},num_epochs=1,shuffle=False)
test(prediction)

print ("finish")


