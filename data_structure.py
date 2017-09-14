import os
import pandas as pd
import numpy as np
import tensorflow as tf
from commands import process_data
from commands import train_neural_network


"""
    DEFINES
"""
IMG_PX_SIZE = 150
HM_SLICES = 20
IMG_SIZE_PX = 50
SLICE_COUNT = 20


#In[1]#################################################################################################################################################
"""
    importing the labels data
"""
labels = pd.read_csv('/home/talhassid/PycharmProjects/input/stage1_labels.csv', index_col=0)
#In[2]#################################################################################################################################################
""" 
    building the data structure of patients and their images
"""
counter = 0
much_data = []
patients_dict={}
#data_dir = '/home/talhassid/PycharmProjects/input/sample_images/'
data_dir = '/VISL2_net/talandhaim/stage1_partly/'
patients = os.listdir(data_dir)
#for patient in patients:
#    try:
#        counter = counter + 1
#        print("patient number:", counter)
#        patient_id = patient
#        full_path_patient_dir = data_dir + patient
#        patient_dict = {}
#        patients_dict.update({patient_id:patient_dict})
#        patient_images_list=[]
#        patients_dict[patient_id].update({"path":full_path_patient_dir})
#        patients_dict[patient_id].update({"images":patient_images_list})
#        label = labels.get_value(patient_id, 'cancer')
#        patients_dict[patient_id].update({"label":label})
#        patient_images = os.listdir(full_path_patient_dir)
#        for image in patient_images:
#            full_path_image = full_path_patient_dir + image
#            patients_dict[patient_id]["images"].append(full_path_image)
#         """
#             downsample the data , make the depth uniform
#         """

#        img_data,label = process_data(data_dir,patient,labels,img_px_size=IMG_SIZE_PX, hm_slices=SLICE_COUNT)
#        much_data.append([img_data,label])
#        patients_dict[patient_id].update({"data":img_data})
#
#    except KeyError as e:
#        print('This is unlabeled data!')
#
#np.save('muchdata-{}-{}-{}.npy'.format(IMG_SIZE_PX,IMG_SIZE_PX,SLICE_COUNT), much_data)


#In[3]#################################################################################################################################################
n_classes = 2
batch_size = 10

#will consist a tensor of floating point numbers.
x = tf.placeholder('float')
# the target output classes will consist a tensor
y = tf.placeholder('float')

keep_rate = 0.8

much_data = np.load('muchdata-50-50-20.npy')

train_data = much_data[:-2]
validation_data = much_data[-2:]
numofpatients = len(patients)

train_neural_network(y,x,n_classes,keep_rate,train_data,validation_data, numofpatients)


print("END")

