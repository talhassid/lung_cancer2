import  tensorflow as tf
import numpy as np

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

w = tf.Variable(tf.zeros([1, 1], dtype=tf.float32))
b = tf.Variable(tf.ones([1, 1], dtype=tf.float32))
y_hat = tf.add(b, tf.matmul(x, w))

















def train(output_layer):
    saver = tf.train.import_meta_graph('flowers-model.meta')



    sess = tf.Session()
    saver = tf.train.Saver()
    saver.save(sess, '/home/talhassid/PycharmProjects/lung_cancer/sentex/our_model')
    return output_layer

def test(output_layer,saver,sess):
    saver.restore(sess, tf.train.latest_checkpoint('./'))
    #The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
    x_batch = images.reshape(1, image_size,image_size,num_channels)
    graph = tf.get_default_graph()
    y_pred = graph.get_tensor_by_name("y_pred:0")

    ## Let's feed the images to the input placeholders
    x= graph.get_tensor_by_name("x:0")
    y_true = graph.get_tensor_by_name("y_true:0")
    y_test_images = np.zeros((1, 2))

    feed_dict_testing = {x: x_batch, y_true: y_test_images}
    result=sess.run(y_pred, feed_dict=feed_dict_testing)





    sess = tf.Session()
    saver = tf.train.import_meta_graph('/home/talhassid/PycharmProjects/lung_cancer/sentex/our_model')
    saver.restore(sess,tf.train.latest_checkpoint('./'))
    much_data = np.load('muchdata-50-50-20.npy')
    feed_dict = much_data[0][0] #A dictionary to pass numeric values to computational graph
    pred = sess.run([output_layer],feed_dict=feed_dict)
    print (pred)
