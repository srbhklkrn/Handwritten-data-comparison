import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import gzip
import os
import math
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
def extract_data(filename,num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        #data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
        data = data.reshape(num_images, IMAGE_SIZE*IMAGE_SIZE)
        minValue=np.min(np.min(data, axis=1), axis=0)
        maxValue=np.max(np.max(data, axis=1), axis=0)
        diff=maxValue-minValue
        data=data/diff
        return data
def extract_labels(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        return labels
def gradient(prediction,actual,inputData):
    gradient=np.matmul(np.transpose(inputData),actual-prediction)
    return gradient
def probability_Labels(labels):
    M=labels.size
    result=np.zeros(shape=(M,CLASS_SIZE))
    for i in range(0,M):
        val=labels[i]
        result[i,val]=1
    return result
def probability_Labels_USPS(labels):
    M,N=labels.shape
    result=np.zeros(shape=(M,CLASS_SIZE))
    for i in range(0,M):
        val=int(labels[i][0])
        result[i,val]=1
    return result
#training data
def softmax(data,weightsVector,size):
    a=np.exp(np.matmul(data,weightsVector))
    sum=np.sum(a,axis=1).reshape(size,1)
    result=np.divide(a,sum)
    return result
def trainLogisticRegression(data):
    eta=0.3
    lambdaVal=0.01
    M,N=data.shape
    # accuracy is greater than 0.9 then stop 785*10
    weights_vector=np.random.rand(IMAGE_FEATURE_SIZE,CLASS_SIZE)
    weights_vector=np.insert(weights_vector,0,0.01,axis=0)
    predicted_output=np.zeros(shape=(M,CLASS_SIZE))
    for i in range(0,ITERATIONS):
        predicted_output=softmax(data,weights_vector,M)
        gradient_res=gradient(predicted_output,train_labels_prob,data)
        #gradient descent
        weights_vector_copy=weights_vector
        weights_vector_copy[0,:]=0
        weights_vector=weights_vector+((eta/M)*(gradient_res+lambdaVal*weights_vector_copy))
    return weights_vector,predicted_output
def classifyOutput(labels):
    M,N=labels.shape
    result=np.zeros(shape=(M,1))
    result=np.argmax(labels, axis=1)
    r_len=int(len(result))
    result=result.reshape(r_len,1)
    return result
def accuracy(actual,predicted,size):
    boolarr=np.equal(actual,predicted)
    count=np.count_nonzero(boolarr == True)
    accuracy=(count*100)/size
    return accuracy
def logistic_regression():
    global weights_vector
    global predicted_output
    weights_vector,predicted_output=trainLogisticRegression(train_data_insert_dim)
    train_data_predicted_classify=classifyOutput(predicted_output)
    t_len=int(len(train_data_predicted_classify))
    accuracyTrain=accuracy(train_labels,train_data_predicted_classify,t_len)
    #testing trained model on test data
    test_len=int(len(test_data_insert_dim))
    test_predict=softmax(test_data_insert_dim,weights_vector,test_len)
    test_predict_classify=classifyOutput(test_predict)
    accuracyTest=accuracy(test_labels,test_predict_classify,test_len)
    print("TEST ACCURACY::",accuracyTest)
    #testing on USPS data
    u_len=int(len(ImageX_insert_dim))
    usps_predict=softmax(ImageX_insert_dim,weights_vector,u_len)
    usps_predict_classify=classifyOutput(usps_predict)
    accuracyUsps=accuracy(ImageY,usps_predict_classify,u_len)
    print("USPS ACCURACY::",accuracyUsps)
    #60000 * 10
    # Test data accuracy
def softmaxNN(s):
    a=np.exp(s)
    M,N=s.shape
    sum=np.sum(a,axis=1).reshape(M,1)
    result=np.divide(a,sum)
    return result
def tanh_prime(x):
    return  1 - np.tanh(x)**2
def snn_predict(x,V,W):
    A = np.matmul(x,V)
    Z = np.tanh(A)
    #layer 2
    B = np.matmul(Z,W)
    prediction = softmaxNN(B)
    return prediction
def snn_train(x,t,tClass,V,W,lambdaT,eta):
    # layer 1
    xSize=int(len(x))
    tSize=int(len(tClass))
    A = np.matmul(x,V)
    Z = np.tanh(A)
    #layer 2
    B = np.matmul(Z,W)
    Y = softmaxNN(B)
    Ew = Y - t
    #Back_Propagation
    dW = np.matmul(Z.T, Ew)
    W_copy=W
    W_copy[0,:]=0
    W=W-(eta)*((dW+lambdaT*W_copy)/xSize)
    Ev = tanh_prime(Z) * np.matmul(Ew,W.T)
    dV = np.matmul(x.T, Ev)
    V_copy=V
    V_copy[0,:]=0
    V=V-(eta)*((dV+lambdaT*V_copy)/xSize)
    #loss = -np.mean ( t * np.log(Y) + (1 - t) * np.log(1 - Y) )
    y_predicted_snn=classifyOutput(Y).reshape(tSize,1)
    accuracy_snn=accuracy(tClass,y_predicted_snn,tSize)
    return  V,W
def single_layer_neural_network():
    #Tuning param values
    layerSizeTune=[100,200,300,400]
    etaTune=[0.1,0.2,0.3,0.4,0.5]
    lambdaTune=[0.01,0.05,0.09,0.1,0.2]
    init_batch_size = 100
    total_batch=int(len(train_data_insert_dim) / init_batch_size)
    initial=0
    global validate_accuracy
    validate_accuracy=np.zeros(shape=[70,1])
    m=0
    global V
    global W
    lambdaT=0.3
    eta=0.1
    n_hidden=300
    V = np.random.normal(loc=0.0, scale=0.1, size=(n_in,n_hidden))
    W = np.random.normal(loc=0.0, scale=0.1, size=(n_hidden,n_out))
    for l in range(epochs):
        initial=0
        batch_size=init_batch_size
        for m in range(total_batch):
            train_batch=train_data_insert_dim[initial:batch_size,:]
            train_test_batch=train_labels[initial:batch_size,:]
            tProb=probability_Labels(train_test_batch)
            initial=batch_size
            batch_size=batch_size+init_batch_size
            V, W = snn_train(train_batch,tProb,train_test_batch,V,W,lambdaT,eta)
    prediction_train=snn_predict(train_data_insert_dim,V,W)
    train_len=int(len(prediction_train))
    prediction_train_classify=classifyOutput(prediction_train).reshape(train_len,1)
    accuracyTrainSnn=accuracy(train_labels,prediction_train_classify,TRAIN_DATA_LABELS_SIZE)
    prediction_val=snn_predict(validate_data_insert_dim,V,W)
    p_len=int(len(prediction_val))
    prediction_classify_val=classifyOutput(prediction_val).reshape(p_len,1)
    accuracyValSnn=accuracy(validate_labels,prediction_classify_val,VALIDATE_DATA_LABELS_SIZE)
    prediction_test=snn_predict(test_data_insert_dim,V,W)
    t_len=int(len(prediction_test))
    prediction_test_classify=classifyOutput(prediction_test).reshape(t_len,1)
    accuracyTestSnn=accuracy(test_labels,prediction_test_classify,TEST_DATA_LABELS_SIZE)
    prediction_usps=snn_predict(ImageX_insert_dim,V,W)
    u_len=int(len(prediction_usps))
    prediction_usps_classify=classifyOutput(prediction_usps).reshape(u_len,1)
    accuracyUSPS=accuracy(ImageY,prediction_usps_classify,u_len)
    print("ACCURACY ON TRAIN DATA::",accuracyTrainSnn)
    print("ACCURACY ON VALIDATION DATA::",accuracyValSnn)
    print("ACCURACY ON TEST DATA::",accuracyTestSnn)
    print("ACCURACY ON USPS DATA::",accuracyUSPS)
def CNN(ImageX,ImageY):
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    session = tf.InteractiveSession()
    #Initializing placeholders for x and y value
    #None --> First dimension can be of any size
    
    x = tf.placeholder(tf.float32, shape = [None, 784]) 
    
    y = tf.placeholder(tf.float32, shape = [None,None])
    

    #First convolution layer
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    x_image = tf.reshape(x,[-1, 28, 28, 1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    #Second convolution layer
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)


    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    #To reduce overfitting
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    #Adding a layer
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    epochsCnn=20000
    step_size=100
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        for i in range(epochsCnn):
            batch = mnist.train.next_batch(50)
            if i % step_size == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    x: batch[0], y: batch[1], keep_prob: 1.0})
                print('step %d, training accuracy %g' % (i, train_accuracy))
            train_step.run(feed_dict={x: batch[0], y: batch[1], keep_prob:
            0.5})
        print('MNIST Test accuracy %g' % accuracy.eval(feed_dict={
            x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0}))
        
        print('USPS Test accuracy %g' % accuracy.eval(feed_dict={
            x: ImageX, y: ImageY, keep_prob: 1.0}))
ITERATIONS=10
learning_rate = 0.3
epochs = 3
lambdaVal=0.1
eta=0.9
TRAIN_LABELS_FILENAME='train-labels-idx1-ubyte.gz'
TRAIN_IMAGES_FILENAME='train-images-idx3-ubyte.gz'
TEST_LABELS_FILENAME='t10k-labels-idx1-ubyte.gz'
TEST_IMAGES_FILENAME='t10k-images-idx3-ubyte.gz'
MNIST_DATA_LABELS_SIZE=60000
TRAIN_DATA_LABELS_SIZE=50000
VALIDATE_DATA_LABELS_SIZE=10000
TEST_DATA_LABELS_SIZE=10000
IMAGE_SIZE=28
CLASS_SIZE=10
IMAGE_FEATURE_SIZE=784
n_in=785
n_hidden=300
n_out=10
mnist_data = extract_data('data/'+TRAIN_IMAGES_FILENAME,MNIST_DATA_LABELS_SIZE)
mnist_labels = extract_labels('data/'+TRAIN_LABELS_FILENAME,MNIST_DATA_LABELS_SIZE)
train_data,validate_data=np.split(mnist_data, [TRAIN_DATA_LABELS_SIZE])
train_labels,validate_labels=np.split(mnist_labels, [TRAIN_DATA_LABELS_SIZE])
train_labels=train_labels.reshape(TRAIN_DATA_LABELS_SIZE,1)
train_labels_prob=probability_Labels(train_labels)
validate_labels=validate_labels.reshape(VALIDATE_DATA_LABELS_SIZE,1)
test_data = extract_data('data/'+TEST_IMAGES_FILENAME,TEST_DATA_LABELS_SIZE)
test_labels = extract_labels('data/'+TEST_LABELS_FILENAME,TEST_DATA_LABELS_SIZE).reshape(TEST_DATA_LABELS_SIZE,1)
#Single Layer Neural Network Params
train_data_insert_dim=np.insert(train_data,0,1,axis=1)
validate_data_insert_dim=np.insert(validate_data,0,1,axis=1)
test_data_insert_dim=np.insert(test_data,0,1,axis=1)
#Load USPS data
ImageX = np.zeros(shape=(20000,784))
ImageY = np.zeros(shape=(20000,1))
imgCount = 0
for root, directories, filenames in os.walk('data/proj3_images/Numerals'):
    for directory in directories:
        for root, directories, filenames in os.walk('data/proj3_images/Numerals/'+directory):
            for filename in filenames: 
                if filename == "Thumbs.db" or filename == "2.list":
                    ignore = 1
                else:
                    img = Image.open('data/proj3_images/Numerals/' + directory + '/' + filename)
                    img = img.resize((28, 28))
                #imgdata = np.array(img.getdata())
       #imgdata = 1 - np.square(imgdata)/65536
                    imgdata = 1 - np.array(img.getdata())/255
                    ImageX[imgCount,:] = imgdata
                    ImageY[imgCount,0] = directory
                    imgCount += 1
ImageYProb=probability_Labels_USPS(ImageY)
ImageX_insert_dim=np.insert(ImageX,0,1,axis=1)
logistic_regression()
single_layer_neural_network()
CNN(ImageX,ImageYProb)
