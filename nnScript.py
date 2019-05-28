import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import time

def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer

    # Output:
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""

    sig = 1.0 / (1.0 + np.exp(-1.0 * z))
    return sig  # your code here


def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the
       training set
     test_data: matrix of training set. Each row of test_data contains
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - feature selection"""

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary
    # mat = loadmat('/Users/xianzhou/Desktop/574 Assign2/Assignment2/basecode/mnist_all.mat')

    # loads the MAT object as a Dictionary
    # Split the training sets into two sets of 50000 randomly sampled training examples and 10000 validation examples.
    # Your code here.
    train_number = []
    train_total = np.zeros((1, mat['train0'].shape[1] + 1), float)
    # The train_total is created for concatenate. The line of 0s will delete afterwards
    for i in range(0, 10):
        key = 'train' + str(i)
        train_ori = mat[key]
        train_number.append(train_ori.shape[0])
        label = np.zeros((train_number[i], 1), float) + float(i)
        train_ori = np.concatenate((train_ori, label), axis=1)
        train_total = np.concatenate((train_total, train_ori), axis=0)

    train_total = train_total[1:, :]  # delete the first row, which is all 0s

    train_total = train_total[np.random.permutation(train_total.shape[0]), :]
    # Randomly permutate the rows to randomly split the training examples and validation examples.
    train_data = train_total[:50000, : -1]
    train_label = train_total[:50000, -1:]
    validation_data = train_total[50000:, : -1]
    validation_label = train_total[50000:, -1:]

    test_number = []
    test_total = np.zeros((1, mat['test0'].shape[1] + 1), float)
    for i in range(0, 10):
        key = 'test' + str(i)
        test_ori = mat[key]
        test_number.append(test_ori.shape[0])
        label = np.zeros((test_number[i], 1), float) + float(i)
        test_ori = np.concatenate((test_ori, label), axis=1)
        test_total = np.concatenate((test_total, test_ori), axis=0)

    test_total = test_total[1:, :]  # delete the first row, which is all 0s

    # Feature selection
    # Your code here.
    feature = []
    for i in range(0, 784):
        feature.append(not (all(train_total[:, i] == train_total[0, i])))
    feature = np.array(feature)
    # We select the column that not all values the same
    # feature is a boolean list used for data Feature selection

    train_data = train_data[:, feature]
    validation_data = validation_data[:, feature]
    test_data = test_total[:, :-1]
    test_label = test_total[:, -1:]
    test_data = test_data[:, feature]

    train_label = train_label.astype(int)
    validation_label = validation_label.astype(int)
    test_label = test_label.astype(int)

    print('preprocess done')

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log
    %   likelihood error function with regularization) given the parameters
    %   of Neural Networks, thetraining data, their corresponding training
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.

    % Output:
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    # Your code here
    n = train_data.shape[0]  # n represent the number of training data
    x_bias = np.ones((1, 1), float)
    h_bias = np.ones((1, 1), float)
    # Adding bias to input layer (x_bias) and hidden layer (h_bias)
    grad_w2 = np.zeros_like(w2)
    grad_w1 = np.zeros_like(w1)

    for i in range(0, n):
        # forward
        x = train_data[i:i + 1, :]  # x_i 1* 717
        x_with_bias = np.concatenate((x, x_bias), axis=1)
        x_with_bias = np.transpose(x_with_bias)
        h = sigmoid(np.dot(w1, x_with_bias))  # shape of h not change after sigmoid function
        h_with_bias = np.concatenate((h, h_bias), axis=0)
        o = sigmoid(np.dot(w2, h_with_bias))
        y_gt = np.zeros((10, 1), float)
        y_gt[int(train_label[i, 0]), 0] = 1.0  # 1.0 means probability equals to 1.0
        # I have already change the train_label value to int, here add int just in case
        j_i = -(y_gt * np.log(o) + (1 - y_gt) * np.log(1 - o))
        j_i = j_i.sum(axis=0)
        obj_val = obj_val + j_i

        # backward
        grad_ji_w2 = np.dot((o - y_gt), np.transpose(h_with_bias))
        grad_w2 = grad_w2 + grad_ji_w2

        grad_ji_w1_p1 = (1 - h) * h * (np.dot(np.transpose(w2[:, :-1]), o - y_gt))  # n_hidden * 1
        grad_ji_w1_p2 = np.transpose(x_with_bias)  # 1 * (n_input + 1)
        grad_ji_w1 = np.dot(grad_ji_w1_p1, grad_ji_w1_p2)  # n_hidden * (n_input + 1)
        grad_w1 = grad_w1 + grad_ji_w1

    grad_w2 = (grad_w2 + w2 * lambdaval) / float(n)
    grad_w1 = (grad_w1 + w1 * lambdaval) / float(n)
    obj_val = obj_val / float(n) + (sum((w1 * w1).flatten()) + sum((w2 * w2).flatten())) * lambdaval / float(n) / 2.0

    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()), 0)

    return (obj_val, obj_grad)


def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden
    %     layer to unit i in output layer.
    % data: matrix of data. Each row of this matrix represents the feature
    %       vector of a particular image

    % Output:
    % label: a column vector of predicted labels"""

    labels = np.array([])
    # Your code here
    n = data.shape[0]
    x_bias = np.ones((1, 1), float)
    h_bias = np.ones((1, 1), float)
    # Adding bias to input layer (x_bias) and hidden layer (h_bias)

    # w1 is reshaped to (n_hidden, n_input + 1)
    # w2 is reshaped to (n_class, n_hidden + 1)
    # no need to transpose w

    for i in range(0, n):
        x = data[i:i + 1, :]  # x_i 1* 717
        x_with_bias = np.concatenate((x, x_bias), axis=1)
        x_with_bias = np.transpose(x_with_bias)
        h = sigmoid(np.dot(w1, x_with_bias))  # shape of h not change after sigmoid function
        h_with_bias = np.concatenate((h, h_bias), axis=0)
        o = sigmoid(np.dot(w2, h_with_bias))
        labels = np.append(labels, o.argmax())

    labels = labels.astype(int)
    labels = labels.reshape((n, 1))

    return labels


"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
#t1 = time.time()
#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 76

# set the number of nodes in output unit
n_class = 10

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden)
initial_w2 = initializeWeights(n_hidden, n_class)

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

# set the regularization hyper-parameter
lambdaval = 1

args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

# Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter': 50}  # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

# In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
# and nnObjGradient. Check documentation for this function before you proceed.
# nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


# Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

# Test the computed parameters

predicted_label = nnPredict(w1, w2, train_data)

# find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, validation_data)

# find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, test_data)

# find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')

