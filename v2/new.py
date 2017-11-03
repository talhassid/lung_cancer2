import os

import tensorflow as tf

from v2.network import train_neural_network
from v2.preprocessing import load_scan, get_pixels_hu, resample

# Some constants
PIXEL_MEAN = 0.25
MIN_BOUND = -1000.0
MAX_BOUND = 400.0

IMG_PX_SIZE = 150
IMG_SIZE_PX = 50
SLICE_COUNT = 20

patients_dict = {}

INPUT_FOLDER = '../input/sample_images/'
patients = os.listdir(INPUT_FOLDER)
patients.sort()
patients_images_list = []
i = 0
for patient_id in patients:
    patient_images = load_scan(INPUT_FOLDER + patient_id)
    patient_pixels = get_pixels_hu(patient_images)
    pix_resampled, spacing = resample(patient_pixels, patient_images, [1,1,1])
    patients_images_list.append(pix_resampled)
    patient_dict = {}
    patients_dict.update({patient_id:patient_dict})
    patients_dict[patient_id].update({"path":INPUT_FOLDER + patient_id})
    patients_dict[patient_id].update({"images":pix_resampled})
    i = i + 1
    print ("patient", i)

train_data = patients_images_list[:-2]
validation_data = patients_images_list[-2:]
# will consist a tensor of floating point numbers.
input_tensor = tf.placeholder('float')
# the target output classes will consist a tensor.
target_tensor = tf.placeholder('float')
train_neural_network(train_data,validation_data,input_tensor,target_tensor)
print ("finish")
