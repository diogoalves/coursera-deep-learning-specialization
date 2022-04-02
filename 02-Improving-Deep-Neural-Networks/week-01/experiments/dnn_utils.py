import numpy as np
import matplotlib.pyplot as plt
import h5py

def load_data():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def load_flatten_and_standarized_data():
    train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

    # Reshape the training and test examples 
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

    # Standardize data to have feature values between 0 and 1.
    train_x = train_x_flatten/255.
    test_x = test_x_flatten/255.

    return train_x, train_y, test_x, test_y, classes



def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy
    
    Arguments:
    Z -- numpy array of any shape
    
    Returns:
    A -- output of sigmoid(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """
    
    A = 1/(1+np.exp(-Z))
    cache = Z
    
    return A, cache

def relu(Z):
    """
    Implement the RELU function.

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """
    
    A = np.maximum(0,Z)
    
    assert(A.shape == Z.shape)
    
    cache = Z 
    return A, cache


def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ



###################################################################################################
# graph utils

def plot_costs(costs, learning_rate=0.0075):
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

# def plot_learning_grah()

def plot_optimization_learning_curve(costs):
    plt.plot(np.squeeze(costs['training']))
    plt.plot(np.squeeze(costs['test']))
    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.title('Optimization learning curve')
    plt.show()

def plot_perfomance_learning_curve(accuracies):
    plt.plot(np.squeeze(accuracies['training']))
    plt.plot(np.squeeze(accuracies['test']))
    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.title('Optimization learning curve')
    plt.show()

######

def accuracy(X, y, parameters):
    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    p = np.zeros((1,m))
    
    probas, caches = L_model_forward(X, parameters)

    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    
    accuracy = np.sum((p == y)/m)
    # print("Accuracy: "  + str(accuracy))
        
    return accuracy

################################################################################

def gradients_to_vector(gradients, layer_dims):
    """
    Roll all our gradients dictionary into a single vector satisfying our specific required shape.
    """
    
    L = len(layer_dims)
    vector = None
    count = 0
    for l in range(1, L):
        if count == 0:
            vector = np.reshape(gradients['dW' + str(l)], (-1, 1))
            count += 1
        else:
            vector = np.concatenate((vector, np.reshape(gradients['dW' + str(l)], (-1, 1))), axis=0)
        vector = np.concatenate((vector, np.reshape(gradients['db' + str(l)], (-1, 1))), axis=0)

    return vector

def vector_to_gradients(vector, layer_dims):
    """
    Unroll all our parameters dictionary from a single vector satisfying our specific required shape.
    """
    parameters = {}

    L = len(layer_dims)            # number of layers in the network

    cursor = 0
    for l in range(1, L):
        n_rows = layer_dims[l]
        n_cols = layer_dims[l-1]
        W_end = cursor + (n_rows * n_cols)
        B_end = W_end + n_rows
        parameters['dW' + str(l)] = vector[cursor : W_end].reshape((n_rows, n_cols))
        parameters['db' + str(l)] = vector[W_end: B_end].reshape((n_rows, 1))
        cursor = B_end

        assert(parameters['dW' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['db' + str(l)].shape == (layer_dims[l], 1))


    return parameters


def parameters_to_vector(parameters, layer_dims):
    """
    Roll all our parameters dictionary into a single vector satisfying our specific required shape.
    """
    L = len(layer_dims)
    vector = None
    count = 0
    for l in range(1, L):
        if count == 0:
            vector = np.reshape(parameters['W' + str(l)], (-1, 1))
            count += 1
        else:
            vector = np.concatenate((vector, np.reshape(parameters['W' + str(l)], (-1, 1))), axis=0)
        vector = np.concatenate((vector, np.reshape(parameters['b' + str(l)], (-1, 1))), axis=0)

    return vector

def vector_to_parameters(vector, layer_dims):
    """
    Unroll all our parameters dictionary from a single vector satisfying our specific required shape.
    """
    parameters = {}

    L = len(layer_dims)            # number of layers in the network

    cursor = 0
    for l in range(1, L):
        n_rows = layer_dims[l]
        n_cols = layer_dims[l-1]
        W_end = cursor + (n_rows * n_cols)
        B_end = W_end + n_rows
        parameters['W' + str(l)] = vector[cursor : W_end].reshape((n_rows, n_cols))
        parameters['b' + str(l)] = vector[W_end: B_end].reshape((n_rows, 1))
        cursor = B_end

        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))


    return parameters