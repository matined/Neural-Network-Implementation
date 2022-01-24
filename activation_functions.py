import numpy as np


def relu(Z):
    """
    ReLU function.
    """
    return np.maximum(0, Z)


def sigmoid(Z):
    """
    Sigmoid function.
    """
    return 1 / (1 + np.exp(-Z))


def relu_backward(dA, Z):
    """
    Backward propagation for a single RELU unit.
    """
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0

    return dZ


def sigmoid_backward(dA, Z):
    """
    Backward propagation for a single SIGMOID unit.
    """
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)

    return dZ
