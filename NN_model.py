import numpy as np
from activation_functions import *


class NN_model:
    def __init__(self, layers_sizes=(10, 1), learning_rate=0.075, max_iter=2000, activation='relu', verbose=True) -> None:
        self.layer_sizes = layers_sizes
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.activation = activation
        self.verbose = verbose

        self.data = {}
        self.parameters = {}
        self.number_of_layers = len(layers_sizes)

    def process_data(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Reshapes the input arrays for the simplicity of implementation and creates a few data parameters to use later.

        Parameters
        ----------
        X : np.ndarray
            Training set, numpy array of shape (number_of_examples, number_of_features) 

        y : np.ndarray
            Target values for the test set, numpy array of shape (number_of_examples, 1)

        Returns
        -------
        data : dictionary
            A dictionary containing processed data:
                X : numpy array of shape (number_of_features, number_of_examples) 
                y : numpy array of shape (1, number_of_examples) 
                m : number_of_training_examples
                n_x : number_of_features
        """

        X = X.T
        m = X.shape[1]
        n_x = X.shape[0]

        y = y.reshape(1, m)

        data = {
            'X':    X,
            'y':    y,
            'm':    m,
            'n_x':  n_x
        }

        return data

    def initialize_parameters(self, layer_dims: tuple) -> dict:
        """
        Randomly initializes parameters for the NN.

        Parameters
        ----------
        layer_dims : tuple
            A tuple containing numbers of units in the consecutive layers

        Returns
        -------
        parameters : dictionary
            A dictionary containing parameters W1...WL, b1...bL for the NN.
                Wn : numpy array of weights for n-th layer
                bn : numpy array of biases for n-th layer
        """
        L = len(layer_dims)  # total number of layers
        parameters = {}

        for l in range(1, L):
            parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
            parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

        return parameters

    def forward_propagation_step(self, A_prev: np.ndarray, W: np.ndarray, b: np.ndarray, activation: str = 'relu') -> tuple:
        """
        Computes a forward propagation step for one layer.

        Parameters
        ----------
        A_prev : np.array
            Activations from the previous layer of shape (size_of_previous_layer, number_of_examples)

        W : np.ndarray
            Weights of shape (size_of_current_layer, size_of_previous_layer)

        b : np.ndarray
            Biases og shape (size_of_current_layer, 1)

        activation : str
            A choice of a activation function (sigmoid, tanh, ReLU, leaky ReLU)

        Returns
        -------
        Z : np.ndarray
            Pre-activation parameter, input for an activation function of size (size_of_current_layer, 1)

        cache : tuple
            A tuple containing A_prev, W, B and Z for the backpropagation
        """
        Z = W @ A_prev + b

        if activation == 'tanh':
            A = np.tanh(Z)

        cache = {
            'A_prev': A_prev,
            'W': W,
            'b': b,
            'Z': Z
        }

        return A, cache

    def forward_propagation(self, X: np.ndarray, parameters: dict, number_of_layers: int, activation_function: str) -> tuple:
        """
        Performs a forward propagation.

        Parameters
        ----------
        X : np.ndarray
            Input values of shape (number_of_features, number_of_examples)

        parameters : dictionary
            Weights and biases W1...WL, b1...bL

        number_of_layers : int
            A number of layers

        activation : str
            A choice of a activation function for hidden layers (sigmoid, tanh, ReLU, leaky ReLU)

        Returns
        -------
        AL : np.ndarrat
            Predictions

        caches : list
            Caches for BP
        """
        caches = []
        A = X

        for l in range(1, number_of_layers):
            A, cache = self.forward_propagation_step(A, parameters['W' + str(l)], parameters['b' + str(l)], activation_function)
            caches.append(cache)

        AL, cache = self.forward_propagation_step(A, parameters['W' + str(number_of_layers)],
                                                  parameters['b' + str(number_of_layers)], activation_function)

        return AL, caches

    def compute_cost(self, AL: np.ndarray, y: np.ndarray) -> float:
        """
        Computes cost.

        Parameters
        ----------
        AL : np.ndarray
            Probabilities corresponding to the label predictions of shape (1, number_of_examples)

        y : np.ndarray
            Target values of predictions of shape (1 , number_of_examples)

        Returns
        -------
        cost : int
            Cross-entropy cost
        """
        m = y.shape[1]
        cost = np.sum(y * np.log(AL) + (1+y) * np.log(1 - AL)) / (-m)
        cost = np.squeeze(cost)

        return cost

    def backpropagation_step(self, dA: np.ndarray, cache: dict, activation: str = 'relu') -> tuple:
        """
        Performs a backpropagation step, computes gradients.

        Parameters
        ----------
        dA : np.ndarray
            Gradient computed ealier.

        cache : dictionary
            Tuple of values A_prev, W, b and Z calculated during the forward pass.

        Returns
        -------
        dA_prev : np.ndarray
            Gradient of the cost w.r.t the activation

        dW : np.ndarray
            Gradient of the cost w.r.t W

        db : np.ndarray
            Gradient of the cost w.r.t b 
        """
        A_prev, W, b, Z = cache['A_prev'], cache['W'], cache['b'], cache['Z']
        m = A_prev.shape[1]

        if activation == 'relu':
            dZ = dA * relu_backward(dA, Z)
        elif activation == 'sigmoid':
            dZ = dA * sigmoid_backward(dA, Z)

        dW = dZ @ A_prev.T / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = W.T @ dZ

        return dA_prev, dW, db

    def backpropagation(self, AL: np.ndarray, y: np.ndarray, caches: list, number_of_layers: int) -> dict:
        """
        Perfoms a full backpropagation.

        Parameters
        ----------
        AL : np.ndarray
            Output of the forward propagation

        y : nd.ndarray
            Target prediction values

        caches : dictionary
            List of caches from the forward propagation

        number_of_layers : int
            A number of layers

        Returns
        -------
        grads : dictionary
            A dictionary of gradients from every step of BP
        """
        grads = {}
        m = AL.shape[1]
        y = y.reshape(AL.shape)

        dAL = -(y / AL - (1 - y) / (1 - AL))

        current_cache = caches[number_of_layers-1]
        dA_prev_temp, dW_temp, db_temp = self.backpropagation_step(dAL, current_cache, 'sigmoid')
        grads["dA" + str(number_of_layers-1)] = dA_prev_temp
        grads["dW" + str(number_of_layers)] = dW_temp
        grads["db" + str(number_of_layers)] = db_temp

        for l in reversed(range(number_of_layers-1)):
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.backpropagation_step(dA_prev_temp, current_cache, 'relu')
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp

        return grads

    def update_parameters(self, params: dict, grads: dict, learning_rate: float = 0.0075) -> dict:
        """
        Update parameters using gradient descent.

        Parameters
        ----------
        params : dictionary
            Parameters W1...WL, b1...bL

        grads : dictionary
            Output of the BP, gradients dA1...dAL, dW1...dWL, db1...dbL

        learning_rate : float
            A learning rate.

        Returns
        -------
        parameters : dictionary
            Updated parameters.
        """
        parameters = params.copy()
        L = len(parameters) // 2

        for l in range(L):
            parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads['dW' + str(l+1)]
            parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads['db' + str(l+1)]

        return parameters

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fits model to the training data.

        Parameters
        ----------
        X : np.array
            The input data as a numpy array of shape (number_of_training_examples, number_of_features)

        y : np.array
            Target values for the training examples of shape (number_of_training_examples, )
        """
        self.data = self.process_data(X, y)
        self.parameters = self.initialize_parameters(self.layer_sizes)

        for i in range(self.max_iter):
            AL, caches = self.forward_propagation(X, self.parameters, self.number_of_layers, self.activation)
            cost = self.compute_cost(AL, y)
            grads = self.backpropagation(AL, y, caches, self.number_of_layers)
            self.parameters = self.update_parameters(self.parameters, grads, self.learning_rate)

            if self.verbose and i % 100 == 0:
                print(f'Cost after interation {i}: {np.squeeze(cost)}')

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts outputs on the set X.

        Parameters
        ----------
        X : np.array
            Input data of shape (number_of_training_examples, number_of_features)

        Returns
        -------
        predictions : np.ndarray
        """
        data = self.process_data(X, np.zeros(1))
        X = data['X']

        predictions = self.forward_propagation(X, self.parameters, self.number_of_layers, self.activation)

        return predictions
