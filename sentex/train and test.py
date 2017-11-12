import tensorflow as tf
def test(x,y):
    sess=tf.Session()
    #First let's load meta graph and restore weights
    saver = tf.train.import_meta_graph('/home/talhassid/PycharmProjects/lung_cancer/sentex/model.ckpt.meta')
    saver.restore(sess,tf.train.latest_checkpoint('./'))

    graph = tf.get_default_graph()

    #Now, access the op that you want to run.
    op_to_restore = graph.get_tensor_by_name("op_to_restore:0")

    # Now, let's access and create placeholders variables and
    # create feed-dict to feed new data
    much_data = np.load('muchdata-50-50-20.npy')
    test_data = much_data[-1]
    X = test_data[0]
    Y = test_data[1]
    feed_dict = {x:X,y:Y}
    print ("prediction[no_cancer , cancer]:", sess.run(op_to_restore,feed_dict=feed_dict))
    prediction=tf.argmax(op_to_restore,1)
    print (prediction.eval(feed_dict=feed_dict))





    for index in range(0,19):
        test_data = much_data[index]
        X = test_data[0]
        Y = test_data[1]
        feed_dict = {x:X,y:Y}
        prediction=tf.nn.softmax(op_to_restore)
        print ("\np_id:",much_data[index][2], "prediction[no_cancer , cancer]:", sess.run(prediction,feed_dict=feed_dict))
        print ("p_id:" ,much_data[index][2], "prediction[no_cancer , cancer]:",prediction.eval(feed_dict=feed_dict))
