
from __future__ import print_function
import mxnet as mx
import numpy as np
# import matplotlib.pyplot as plt
import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Network declaration as symbols. The following pattern was based
# on the article, but feel free to play with the number of nodes
# and with the activation function
data = mx.symbol.Variable('data')
fc1  = mx.symbol.FullyConnected(data = data, name='fc1', num_hidden=8192)
act1 = mx.symbol.Activation(data = fc1, name='relu1', act_type="relu")
fc2  = mx.symbol.FullyConnected(data = act1, name = 'fc2', num_hidden = 4096)
act2 = mx.symbol.Activation(data = fc2, name='relu2', act_type="relu")
fc3  = mx.symbol.FullyConnected(data = act2, name = 'fc3', num_hidden = 2048)
act3 = mx.symbol.Activation(data = fc3, name='relu3', act_type="relu")
fc4  = mx.symbol.FullyConnected(data = act3, name = 'fc4', num_hidden = 1024)
act4 = mx.symbol.Activation(data = fc4, name='relu4', act_type="relu")
fc5  = mx.symbol.FullyConnected(data = act4, name = 'fc5', num_hidden = 512)
act5 = mx.symbol.Activation(data = fc5, name='relu5', act_type="relu")
fc6  = mx.symbol.FullyConnected(data = act5, name = 'fc6', num_hidden = 256)
act6 = mx.symbol.Activation(data = fc6, name='relu6', act_type="relu")
fc7  = mx.symbol.FullyConnected(data = act6, name='fc7', num_hidden=1)

mlp=mx.symbol.LinearRegressionOutput(data=fc7, name='softmax')

#Use SoftmaxOutput for classification problems
# mlp = mx.symbol.SoftmaxOutput(data=fc4, name='softmax')

batch_size = 2000
shape=11

trainIt=mx.io.CSVIter(
            data_csv="procData/test_train.csv",
            data_shape=(shape),
            label_csv='procData/test_train_label.csv',
            batch_size=batch_size,
            name='softmax')
testIt=mx.io.CSVIter(
            data_csv="procData/test_test.csv",
            data_shape=(shape),
            label_csv='procData/test_test_label.csv',
            batch_size=batch_size,
            name='softmax')

print(trainIt.getdata())
print(trainIt.getlabel())

model = mx.model.FeedForward(
    ctx = mx.gpu(0),      # Run on GPU 0
    symbol = mlp,         # Use the network we just defined
    num_epoch = 2000,       # Train for 10 epochs
    learning_rate = 0.05,  # Learning rate
    momentum = 0.95,       # Momentum for SGD with momentum
    wd = 0.00001,         # Weight decay for regularization
    initializer = mx.initializer.Normal(sigma=0.01),
    )
model.fit(
    X=trainIt,  # Training data set
    eval_data=testIt,  # Testing data set. MXNet computes scores on test set every epoch
    eval_metric=mx.metric.RMSE(), #Use Root Mean Squre to evaluate. Smaller values are preferable
    batch_end_callback = mx.callback.Speedometer(batch_size, 50))  # Logging module to print out progress

# Uncomment to view an example
# plt.imshow((X_show[0].reshape((28,28))*255).astype(np.uint8), cmap='Greys_r')
# plt.show()
# print 'Result:', model.predict(X_test[0:1])[0].argmax()

# Now it prints how good did the network did for this configuration
print('RMSE:', model.score(testIt,mx.metric.RMSE()))

[prob, data1, label1]=model.predict(
            X=testIt,
            num_batch=None,
            return_data=True)

print(np.array(prob[1:100])*100*15)
# print(data1)
print(np.array(label1[1:100])*100*15)

#save model
prefix = 'mymodel'
iteration = 100
model.save(prefix, iteration)
