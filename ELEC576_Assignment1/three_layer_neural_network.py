__author__ = 'tan_nguyen & Yu-Che Lin'

import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt

figureFolder = "figure"


def generate_data():
    """
    generate data
    :return: X: input data, y: given labels
    """
    np.random.seed(0)
    # X, y = datasets.make_moons(200, noise=0.20)
    # X, y = datasets.make_circles(200, noise=0.20)
    X, y = datasets.make_gaussian_quantiles(n_samples=300, n_features=2, n_classes=3, random_state=0)
    return X, y


def plot_decision_boundary(pred_func, X, y, actName, numberLayer, numberUnit, string):
    """
    plot the decision boundary
    :param pred_func:   function used to predict the label
    :param X:           input data
    :param y:           given labels
    :param actName:     type name
    :param numberLayer: number of hidden layers
    :param numberUnit:  number of units in hidden layer
    :param string:      string for the title
    :return:
    """
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.title(string)
    plt.savefig(figureFolder + "/" + actName + "_" + numberLayer + "_" + numberUnit + ".png")
    plt.show()


########################################################################################################################
########################################################################################################################
# YOUR ASSIGNMENT STARTS HERE
# FOLLOW THE INSTRUCTION BELOW TO BUILD AND TRAIN A 3-LAYER NEURAL NETWORK
########################################################################################################################
########################################################################################################################
class NeuralNetwork(object):
    """
    This class builds and trains a neural network
    """

    def __init__(self, nn_input_dim, nn_hidden_dim, nn_output_dim, actFun_type='tanh', reg_lambda=0.01, seed=0):
        """
        :param nn_input_dim:    input dimension
        :param nn_hidden_dim:   the number of hidden units
        :param nn_output_dim:   output dimension
        :param actFun_type:     type of activation function. 3 options: 'tanh', 'sigmoid', 'relu'
        :param reg_lambda:      regularization coefficient
        :param seed:            random seed
        """
        self.nn_input_dim = nn_input_dim
        self.nn_hidden_dim = nn_hidden_dim
        self.nn_output_dim = nn_output_dim
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda

        # initialize the weights and biases in the network
        np.random.seed(seed)
        self.W1 = np.random.randn(self.nn_input_dim, self.nn_hidden_dim) / np.sqrt(self.nn_input_dim)
        self.b1 = np.zeros((1, self.nn_hidden_dim))
        self.W2 = np.random.randn(self.nn_hidden_dim, self.nn_output_dim) / np.sqrt(self.nn_hidden_dim)
        self.b2 = np.zeros((1, self.nn_output_dim))

    def actFun(self, z, type):
        """
        actFun computes the activation functions
        :param z: net input
        :param type: Tanh, Sigmoid, or ReLU
        :return: activations
        """

        # YOU IMPLEMENT YOUR actFun HERE
        if type == "tanh":
            return np.tanh(z)
        elif type == "sigmoid":
            return 1 / (1 + np.exp(-z))
        elif type == "relu":
            return np.maximum(z, 0)

        return None

    def diff_actFun(self, z, type):
        """
        diff_actFun compute the derivatives of the activation functions wrt the net input
        :param z: net input
        :param type: Tanh, Sigmoid, or ReLU
        :return: the derivatives of the activation functions wrt the net input
        """

        # YOU IMPLEMENT YOUR diff_actFun HERE
        if type == "tanh":
            return 1 - np.square(self.actFun(z, type))
        elif type == "sigmoid":
            function = self.actFun(z, type)
            return function * (1 - function)
        elif type == "relu":
            return np.where(z > 0, 1, 0)

        return None

    def feedforward(self, X, actFun):
        """
        feedforward builds a 3-layer neural network and computes the two probabilities,
        one for class 0 and one for class 1
        :param X: input data
        :param actFun: activation function
        :return:
        """

        # YOU IMPLEMENT YOUR feedforward HERE
        # print("X:", X.shape)          # X:  (200, 2)
        # print("W1:", self.W1.shape)   # W1: (2, 3)
        # print("b1:", self.b1.shape)   # b1: (1, 3)
        # print("W2:", self.W2.shape)   # W2: (3, 2)
        # print("b2:", self.b2.shape)   # b2: (1, 2)
        self.z1 = np.dot(X, self.W1) + self.b1
        # print("z1:", self.z1.shape)   # z1: (200, 3)
        self.a1 = actFun(self.z1)
        # print("a1:", self.a1.shape)   # a1: (200, 3)

        self.z2 = np.dot(self.a1, self.W2) + self.b2
        # print("z2:", self.z2.shape)   # z2: (200, 2)
        self.probs = np.exp(self.z2) / np.sum(np.exp(self.z2), axis=1, keepdims=True)  # Softmax
        # print("probability:", self.probs.shape) # probability: (200, 2) ===> y prediction(probability)

        return None

    def calculate_loss(self, X, y):
        """
        calculate_loss compute the loss for prediction
        :param X: input data
        :param y: given labels
        :return: the loss for prediction
        """
        num_examples = len(X)
        self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
        # Calculating the loss

        # YOU IMPLEMENT YOUR CALCULATION OF THE LOSS HERE
        num_output = len(y)
        # print("# of output:", num_output) # y: (200, 2) ===> y prediction(prob)
        probOneHotEncoding = self.probs[np.arange(num_output), y]
        data_loss = - np.sum(np.log(probOneHotEncoding))
        # print(data_loss)

        # Add regularization term to loss (optional)
        data_loss += self.reg_lambda / 2 * (np.sum(np.square(self.W1)) + np.sum(np.square(self.W2)))
        data_loss = (1. / num_examples) * data_loss
        # print(data_loss)
        return data_loss

    def predict(self, X):
        """
        predict infers the label of a given data point X
        :param X: input data
        :return: label inferred
        """
        self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
        return np.argmax(self.probs, axis=1)

    def backprop(self, X, y):
        """
        backprop run backpropagation to compute the gradients used to update the parameters in the backward step
        :param X: input data
        :param y: given labels
        :return: dL/dW1, dL/b1, dL/dW2, dL/db2
        """

        # IMPLEMENT YOUR BACKPROP HERE (Lecture3 P.42)
        num_examples = len(X)
        num_output = len(y)

        delta2 = self.probs.copy()
        delta2[range(num_output), y] -= 1                               # (prob-y)

        typeDiff = self.diff_actFun(self.z2, type=self.actFun_type)     # diff_type(z2)
        delta2 *= typeDiff    # not sure to add or not                  # (prob-y) * diff_type(z2)
        dW2 = np.dot(self.a1.T, delta2)                                 # dW2 = dL/dW2
        db2 = np.sum(delta2, axis=0, keepdims=True)                     # db2 = dL/db2

        typeDiff = self.diff_actFun(self.z1, type=self.actFun_type)
        delta1 = np.dot(delta2, self.W2.T) * typeDiff
        dW1 = np.dot(X.T, delta1)                                       # dW1 = dL/dW1
        db1 = np.sum(delta1, axis=0, keepdims=True)                     # db1 = dL/db1

        return dW1, dW2, db1, db2

    def fit_model(self, X, y, epsilon=0.01, num_passes=20000, print_loss=True):
        """
        fit_model uses backpropagation to train the network
        :param X: input data
        :param y: given labels
        :param epsilon:
        :param num_passes: the number of times that the algorithm runs through the whole dataset
        :param print_loss: print the loss or not
        :return:
        """
        # Gradient descent.
        dataLosses = []
        index = []
        for i in range(0, num_passes):
            # Forward propagation
            self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
            # Backpropagation
            dW1, dW2, db1, db2 = self.backprop(X, y)

            # Add derivatives of regularization terms (b1 and b2 don't have regularization terms)
            dW2 += self.reg_lambda * self.W2
            dW1 += self.reg_lambda * self.W1

            # Gradient descent parameter update
            self.W1 += -epsilon * dW1
            self.b1 += -epsilon * db1
            self.W2 += -epsilon * dW2
            self.b2 += -epsilon * db2

            # Optionally print the loss.
            # This is expensive because it uses the whole dataset, so we don't want to do it too often.
            if print_loss and i % 1000 == 0:
                dataLoss = self.calculate_loss(X, y)
                print("Loss after iteration %i: %f" % (i, dataLoss))
                index.append(i)
                dataLosses.append(dataLoss)
        return dataLosses, index

    def visualize_decision_boundary(self, X, y, actName, numberLayer, numberUnit, string):
        """
        visualize_decision_boundary plot the decision boundary created by the trained network
        :param X:           input data
        :param y:           given labels
        :param actName:     type name
        :param numberLayer: number of hidden layers
        :param numberUnit:  number of units in hidden layer
        :param string:      string for the title
        :return:
        """
        plot_decision_boundary(lambda x: self.predict(x), X, y, actName, numberLayer, numberUnit, string)


def main():
    # # generate and visualize Make-Moons dataset
    X, y = generate_data()
    plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    plt.title("Original Data")
    plt.savefig(figureFolder + "/original_data.png")
    plt.show()

    data = {}
    df = pd.DataFrame(data)
    actType = ["tanh", "sigmoid", "relu"]
    for act in actType:
        for i in range(3, 4):
            string = "Type: " + act + " Hidden Layer: 1" + " Units: " + str(i)
            print("\n\n" + string)
            model = NeuralNetwork(nn_input_dim=2, nn_hidden_dim=i, nn_output_dim=2, actFun_type=act)
            df[act + "_1_" + str(i)], index = model.fit_model(X, y)
            model.visualize_decision_boundary(X, y, act, str(1), str(i), string)
    df["Iteration"] = index
    df.set_index("Iteration", inplace=True)
    df.to_csv("dataLoss_1_n.csv", index=True)


if __name__ == "__main__":
    main()
