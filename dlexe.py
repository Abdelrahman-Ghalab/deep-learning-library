#this is my deep learning library
__author__      = "Abdelrahman Ghalab"



from deepLearning import*
import time

def read_data():

    data = np.genfromtxt('data.csv',delimiter=',')
    X = data [:,0:9]                #9 is the number of variables
    X = normalize(X)
    X = np.transpose(X)                 #to have it in the form of rows: features, columns: queries
    Y = (data[:,9])



    train_size = int(np.ceil(X.shape[1]*0.8))                       #0.8 is the percentage of training to testing
    X_train = X[:,0:train_size]
    Y_train= Y[0:train_size]
    X_test = X[:,train_size:]
    Y_test = Y[train_size:]

    return (X_train, Y_train, X_test, Y_test)


def nn_model_with_mini_batches(X_train, Y_train, X_test, Y_test,layers_dims, learning_rate, num_epochs, lambd):
    """
    n-layer neural network model which can be run in different optimizer modes.

    Arguments:
    X -- input data, of shape (n, number of examples)
    Y -- true "label" vector (1 / 0), of shape (1, number of examples)
    layers_dims -- python list, containing the size of each layer
    learning_rate -- the learning rate, scalar.
    mini_batch_size -- the size of a mini batch
    beta -- Momentum hyperparameter
    beta1 -- Exponential decay hyperparameter for the past gradients estimates
    beta2 -- Exponential decay hyperparameter for the past squared gradients estimates
    epsilon -- hyperparameter preventing division by zero in Adam updates
    num_epochs -- number of epochs

    """

    parameters = initialize_parameters_he(layers_dims)
    v,s = initialize_adam(parameters)

    cost_accumulator = np.array([])

    for i in range(num_epochs):
        minibatches = random_mini_batches(X_train, Y_train)

        for minibatch in minibatches:

            (minibatch_X, minibatch_Y) = minibatch

            An, cache = forward_prop(minibatch_X, parameters, layers_dims)

            #cost = compute_cost(An, Y_train)

            cost = compute_cost_with_regularization(An, minibatch_Y, parameters, lambd,layers_dims)




                #grads = backward_propagation(parameters, cache, X_train, Y_train, layers_dims)

            grads = backward_propagation_with_regularization(parameters, cache, minibatch_X, minibatch_Y, layers_dims, lambd =0.2)

            #parameters = update_parameters(parameters, grads, layers_dims, learning_rate)



            parameters,v,s =update_parameters_with_adam(parameters, grads, v ,s , t=1)

        if(i%1000==0):
            cost_accumulator = np.append(cost_accumulator, cost)
            print("Cost after iteration {}: {}".format(i, cost))


    predictions = predict(parameters, X_test, layers_dims)

    accuracy = computer_classifier_accuracy(predictions, Y_test)

    print("classifier accuracy = ",accuracy*100, "%")
    plt.plot(cost_accumulator)
    plt.show()

def nn_model(X_train, Y_train, X_test, Y_test, layers_dims,learning_rate, num_iterations, lambd):
    """
    n-layer neural network model which can be run in different optimizer modes.

    Arguments:
    X -- input data, of shape (n, number of examples)
    Y -- true "label" vector (1 / 0), of shape (1, number of examples)
    layers_dims -- python list, containing the size of each layer
    learning_rate -- the learning rate, scalar.
    mini_batch_size -- the size of a mini batch
    beta -- Momentum hyperparameter
    beta1 -- Exponential decay hyperparameter for the past gradients estimates
    beta2 -- Exponential decay hyperparameter for the past squared gradients estimates
    epsilon -- hyperparameter preventing division by zero in Adam updates
    num_iterations -- number of iterations

    """

    parameters = initialize_parameters_he(layers_dims)
    v,s = initialize_adam(parameters)

    cost_accumulator = np.array([])

    for i in range(num_iterations):

        An, cache = forward_prop(X_train, parameters, layers_dims)

        #cost = compute_cost(An, Y_train)

        cost = compute_cost_with_regularization(An, Y_train, parameters, lambd,layers_dims)
        if(i%1000==0):
            cost_accumulator = np.append(cost_accumulator, cost)
            print("Cost after iteration {}: {}".format(i, cost))


        #grads = backward_propagation(parameters, cache, X_train, Y_train, layers_dims)

        grads = backward_propagation_with_regularization(parameters, cache, X_train, Y_train, layers_dims, lambd =0.2)

        #parameters = update_parameters(parameters, grads, layers_dims, learning_rate)


        parameters,v,s =update_parameters_with_adam(parameters, grads, v ,s , t=1)

    predictions = predict(parameters, X_test, layers_dims)

    accuracy = computer_classifier_accuracy(predictions, Y_test)

    print("classifier accuracy = ",accuracy*100, "%")
    plt.plot(cost_accumulator)
    plt.show()


t1 = time.time()
(X_train, Y_train, X_test, Y_test) = read_data()
layers_dims = (9,8,5,1)                     #architecture of the network
                                            #first number has to be equal to the number of variables
nn_model_with_mini_batches(X_train, Y_train, X_test, Y_test,layers_dims,0.1, 5000, 0.2)
t2 = time.time()
print(t2-t1)
