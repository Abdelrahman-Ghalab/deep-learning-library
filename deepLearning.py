#this is my deep learning library
__author__      = "Abdelrahman Ghalab"



import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing


#normalize
def normalize(data):
    """
    Arguments:
    the data to be scaled
    data should be in this form: rows are queries and columns are features

    Returns:
    scaled data with mean =0 and std = 1
    """

    normalized_data = preprocessing.scale(data)

    return normalized_data


#initialize the neural network with the depth
def initialize_parameters_random(layers_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the size of each layer.

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                    b1 -- bias vector of shape (layers_dims[1], 1)
                    ...
                    WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                    bL -- bias vector of shape (layers_dims[L], 1)
    """

    parameters = {}
    L = len(layers_dims)            # number of layers

    for l in range(1, L):

        parameters['W' + str(l)] = np.random.randn(layers_dims[l],layers_dims[l-1])*10
        parameters['b' + str(l)] = np.zeros((layers_dims[l],1))


    return parameters



def initialize_parameters_he(layers_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the size of each layer.

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                    b1 -- bias vector of shape (layers_dims[1], 1)
                    ...
                    WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                    bL -- bias vector of shape (layers_dims[L], 1)
    """

    parameters = {}
    L = len(layers_dims) - 1 # number of layers

    for l in range(1, L + 1):

        parameters['W' + str(l)] = np.random.randn(layers_dims[l],layers_dims[l-1])*np.sqrt(2./layers_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layers_dims[l],1))

    return parameters


#sigmoid function
def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """

    s = 1 / (1 + np.exp(-z))


    return s



#forward propagation
def forward_prop(X, parameters, layers_dims):
    """
    Arguments:
    parameters -- python dictionary containing our parameters
    X -- input data of shape (n, number of examples)
    layer_dims

    Returns:
    cache -- a dictionary containing "Z1", "A1", "Z2", "A2",...
    and An

    """
    cache = {}
    L = len(layers_dims)-1       #number of layers
    Z = np.dot(parameters["W1"],X)+parameters["b1"]
    A = np.tanh(Z)
    cache["Z1"]= Z
    cache["A1"]=A
    for l in range(2,L):
        Z = np.dot( parameters["W"+str(l)],A)+parameters["b"+str(l)]
        A = np.tanh(Z)
        cache["Z"+str(l)] = Z
        cache["A"+str(l)] = A

    Zn = np.dot(parameters["W"+str(L)],A)+parameters["b"+str(L)]
    An = sigmoid(Zn)
    cache["Z"+str(L)] = Zn
    cache["A"+str(L)] = An

    return An, cache


#compute cost
def compute_cost(An, Y):
    """
    Computes the cross-entropy cost given in equation (13)

    Arguments:
    An -- The sigmoid output of the second activation, of shape (1, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    parameters -- python dictionary containing your parameters W1, b1, W2 and b2

    Returns:
    cost
    """

    m = Y.shape[0] # number of example



    # Compute the cross-entropy cost
    logprobs = np.multiply(np.log(An), Y) + np.multiply((1 - Y), np.log(1 - An))
    cost = - np.sum(logprobs) / m

    cost = np.squeeze(cost)     # makes sure cost is the dimension we expect.
                                # E.g., turns [[17]] into 17
    #assert(isinstance(cost, float))

    return cost


def compute_cost_with_regularization(An, Y, parameters, lambd,layers_dims ):
    """

    Arguments:
    An -- post-activation, output of forward propagation, of shape (output size, number of examples)
    Y -- "true" labels vector, of shape (output size, number of examples)
    parameters -- python dictionary containing parameters of the model

    Returns:
    cost - value of the regularized loss function
    """
    m = Y.shape[0]
    L = len(layers_dims)-1

    cross_entropy_cost = compute_cost(An, Y) # This gives you the cross-entropy part of the cost

    L2_regularization_cost = 0
    for l in range(1,L+1):
        L2_regularization_cost += (np.sum(np.square(parameters["W"+str(l)])))*(lambd/(2*m))

    cost = cross_entropy_cost + L2_regularization_cost

    return cost

#backward propagation

def backward_propagation(parameters, cache, X, Y, layers_dims):
    """

    Arguments:

    parameters -- python dictionary containing our parameters
    cache -- a dictionary containing "Z1", "A1", "Z2", "A2"...
    X -- input data of shape (n, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    layers-dims to know the number of layers
    Returns:
    grads -- python dictionary containing your gradients with respect to different parameters
    """
    m = X.shape[1]

    grads = {}
    L = len(layers_dims)-1
    # Backward propagation: calculate dW1, db1, dW2, db2,...
    dZ = cache["A"+str(L)] -Y

    for l in reversed(range(2,L+1)):
        dWn = (1 / m) * np.dot(dZ, cache["A"+str(l-1)].T)
        grads["dW"+str(l)] = dWn
        dbn = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        grads["db"+str(l)] = dbn
        dZ = np.multiply(np.dot(parameters["W"+str(l)].T, dZ), 1 - np.power(cache["A"+str(l-1)], 2))



    dW1 = (1 / m) * np.dot(dZ, X.T)
    db1 = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    grads["dW1"] = dW1
    grads["db1"] = db1



    return grads


def backward_propagation_with_regularization(parameters, cache, X, Y, layers_dims, lambd):
    """
    Arguments:
    X -- input dataset, of shape (input size, number of examples)
    Y -- "true" labels vector, of shape (output size, number of examples)
    cache -- cache output from forward_propagation()
    lambd -- regularization hyperparameter, scalar

    Returns:
    gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
    """


    m = X.shape[1]

    grads = {}


    L = len(layers_dims)-1
    # Backward propagation: calculate dW1, db1, dW2, db2,...
    dZ = cache["A"+str(L)] -Y

    for l in reversed(range(2,L+1)):
        dWn = (1 / m) * np.dot(dZ, cache["A"+str(l-1)].T) + (lambd * parameters["W"+str(l)]) / m
        grads["dW"+str(l)] = dWn
        dbn = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        grads["db"+str(l)] = dbn
        dZ = np.multiply(np.dot(parameters["W"+str(l)].T, dZ), 1 - np.power(cache["A"+str(l-1)], 2))



    dW1 = (1 / m) * np.dot(dZ, X.T) + (lambd * parameters["W1"]) / m
    db1 = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    grads["dW1"] = dW1
    grads["db1"] = db1



    return grads

#update parameters
def update_parameters(parameters, grads, layers_dims,learning_rate=1.2):
    """
    Updates parameters using the gradient descent update rule given above

    Arguments:
    layers_dims to know the number of layers
    parameters -- python dictionary containing your parameters
    grads -- python dictionary containing your gradients

    Returns:
    parameters -- python dictionary containing your updated parameters
    """

    L = len(layers_dims)-1
    # Update rule for each parameter
    for l in range(1,L+1):
        parameters["W"+str(l)] -= learning_rate * grads["dW"+str(l)]
        parameters["b"+str(l)] -= learning_rate * grads["db"+str(l)]

    return parameters

def initialize_adam(parameters) :
    """
    Initializes v and s as two python dictionaries with:
                - keys: "dW1", "db1", ..., "dWL", "dbL"
                - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.

    Arguments:
    parameters -- python dictionary containing your parameters.
                    parameters["W" + str(l)] = Wl
                    parameters["b" + str(l)] = bl

    Returns:
    v -- python dictionary that will contain the exponentially weighted average of the gradient.
                    v["dW" + str(l)] = ...
                    v["db" + str(l)] = ...
    s -- python dictionary that will contain the exponentially weighted average of the squared gradient.
                    s["dW" + str(l)] = ...
                    s["db" + str(l)] = ...

    """

    L = len(parameters) // 2 # number of layers in the neural networks
    v = {}
    s = {}

    # Initialize v, s. Input: "parameters". Outputs: "v, s".
    for l in range(L):
        v["dW" + str(l + 1)] = np.zeros_like(parameters["W" + str(l + 1)])
        v["db" + str(l + 1)] = np.zeros_like(parameters["b" + str(l + 1)])

        s["dW" + str(l+1)] = np.zeros_like(parameters["W" + str(l + 1)])
        s["db" + str(l+1)] = np.zeros_like(parameters["b" + str(l + 1)])


    return v, s


def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate=0.01,
                                beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Update parameters using Adam

    Arguments:
    parameters -- python dictionary containing your parameters:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients for each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    v -- Adam variable, moving average of the first gradient, python dictionary
    s -- Adam variable, moving average of the squared gradient, python dictionary
    learning_rate -- the learning rate, scalar.
    beta1 -- Exponential decay hyperparameter for the first moment estimates
    beta2 -- Exponential decay hyperparameter for the second moment estimates
    epsilon -- hyperparameter preventing division by zero in Adam updates

    Returns:
    parameters -- python dictionary containing your updated parameters
    v -- Adam variable, moving average of the first gradient, python dictionary
    s -- Adam variable, moving average of the squared gradient, python dictionary
    """

    L = len(parameters) // 2                 # number of layers in the neural networks
    v_corrected = {}                         # Initializing first moment estimate, python dictionary
    s_corrected = {}                         # Initializing second moment estimate, python dictionary

    # Perform Adam update on all parameters
    for l in range(L):
        # Moving average of the gradients. Inputs: "v, grads, beta1". Output: "v".
        v["dW" + str(l + 1)] = beta1 * v["dW" + str(l + 1)] + (1 - beta1) * grads['dW' + str(l + 1)]
        v["db" + str(l + 1)] = beta1 * v["db" + str(l + 1)] + (1 - beta1) * grads['db' + str(l + 1)]

        # Compute bias-corrected first moment estimate. Inputs: "v, beta1, t". Output: "v_corrected".
        v_corrected["dW" + str(l + 1)] = v["dW" + str(l + 1)] / (1 - np.power(beta1, t))
        v_corrected["db" + str(l + 1)] = v["db" + str(l + 1)] / (1 - np.power(beta1, t))

        # Moving average of the squared gradients. Inputs: "s, grads, beta2". Output: "s".
        s["dW" + str(l + 1)] = beta2 * s["dW" + str(l + 1)] + (1 - beta2) * np.power(grads['dW' + str(l + 1)], 2)
        s["db" + str(l + 1)] = beta2 * s["db" + str(l + 1)] + (1 - beta2) * np.power(grads['db' + str(l + 1)], 2)

        # Compute bias-corrected second raw moment estimate. Inputs: "s, beta2, t". Output: "s_corrected".
        s_corrected["dW" + str(l + 1)] = s["dW" + str(l + 1)] / (1 - np.power(beta2, t))
        s_corrected["db" + str(l + 1)] = s["db" + str(l + 1)] / (1 - np.power(beta2, t))

        # Update parameters. Inputs: "parameters, learning_rate, v_corrected, s_corrected, epsilon". Output: "parameters".
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * v_corrected["dW" + str(l + 1)] / np.sqrt(s["dW" + str(l + 1)] + epsilon)
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * v_corrected["db" + str(l + 1)] / np.sqrt(s["db" + str(l + 1)] + epsilon)

    return parameters, v, s


#test
def predict(parameters, X, layers_dims):
    """
    Using the learned parameters, predicts a class for each example in X

    Arguments:
    parameters -- python dictionary containing your parameters
    X -- input data of size (n_x, m)

    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """

    # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.

    An, cache = forward_prop(X, parameters,layers_dims)
    predictions = np.round(An)

    return predictions

def computer_classifier_accuracy(predictions, Y_test):

    false = np.count_nonzero(predictions-Y_test)

    accuracy = (Y_test.shape[0]-false)/Y_test.shape[0]

    return accuracy

#visualize
