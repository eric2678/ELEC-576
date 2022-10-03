from three_layer_neural_network import NeuralNetwork, generate_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

figureFolder = "figure"


class DeepNeuralNetwork(NeuralNetwork):
    """
    This class inherit NeuralNetwork
    """
    def __init__(self, nn_input_dim, nn_hidden_layer_num, nn_hidden_dim, nn_output_dim, actFun_type='tanh',
                 reg_lambda=0.01, seed=0):
        """
        :param nn_input_dim:            input dimension
        :param nn_hidden_layer_num:     number of hidden layers
        :param nn_hidden_dim:           the number of hidden units
        :param nn_output_dim:           output dimension
        :param actFun_type:             type of activation function. 3 options: 'tanh', 'sigmoid', 'relu'
        :param reg_lambda:              regularization coefficient
        :param seed:                    random seed
        """
        self.nn_input_dim = nn_input_dim
        self.nn_hidden_layer_num = nn_hidden_layer_num
        self.nn_hidden_dim = nn_hidden_dim
        self.nn_output_dim = nn_output_dim
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda

        # initialize the weights and biases in the network
        np.random.seed(seed)
        self.W = []
        self.b = []

        for i in range(0, nn_hidden_layer_num + 1):
            if i == 0:
                inputLayer, outputLayer = nn_input_dim,  nn_hidden_dim
            elif i == nn_hidden_layer_num:
                inputLayer, outputLayer = nn_hidden_dim, nn_output_dim
            else:
                inputLayer, outputLayer = nn_hidden_dim, nn_hidden_dim

            self.W.append(np.random.randn(inputLayer, outputLayer) / np.sqrt(inputLayer))
            self.b.append(np.zeros((1, outputLayer)))

    def feedforward(self, X, actFun):
        """
        feedforward builds a 3-layer neural network and computes the two probabilities,
        one for class 0 and one for class 1

        Args:
            X:      input data
            actFun: activation function
        """
        self.layers = []
        for i in range(0, self.nn_hidden_layer_num + 1):
            inputData = X if i == 0 else self.a
            layer = Layer(inputData, self.W[i], self.b[i], self.actFun_type)
            self.layers.append(layer)
            self.z, self.a = self.layers[i].feedforward(actFun)

        self.probs = np.exp(self.z) / np.sum(np.exp(self.z), axis=1, keepdims=True)  # Softmax

        return None

    def backprop(self, X, y):
        """
        backprop run backpropagation to compute the gradients used to update the parameters in the backward step
        :param X: input data
        :param y: given labels
        :return: dL/dW1, dL/b1, dL/dW2, dL/db2
        """

        # IMPLEMENT YOUR BACKPROP HERE (Lecture3 P.42)
        num_examples = len(X)
        dWs = []
        dbs = []
        deltas = []

        delta = self.probs.copy()
        delta[range(num_examples), y] -= 1  # (prob-y)
        for i in range(self.nn_hidden_layer_num, -1, -1):
            diffType = self.diff_actFun(self.layers[i].z, type=self.actFun_type)
            delta = delta * diffType if i == self.nn_hidden_layer_num else np.dot(delta, self.W[i+1].T) * diffType
            deltas.insert(0, delta)
            dW, db = self.layers[i].backprop(delta)
            dWs.insert(0, dW)
            dbs.insert(0, db)
        return dWs, dbs

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
        total = 0
        for i in range(0, self.nn_hidden_layer_num + 1):
            total += np.sum(np.square((self.W[i])))
        data_loss += self.reg_lambda / 2 * total
        data_loss = (1. / num_examples) * data_loss
        # print(data_loss)
        return data_loss


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
            dWs, dbs = self.backprop(X, y)

            # Add derivatives of regularization terms (b1 and b2 don't have regularization terms)
            for j in range(self.nn_hidden_layer_num, -1, -1):
                dWs[j] += self.reg_lambda * self.W[j]

            # Gradient descent parameter update
            for j in range(0, self.nn_hidden_layer_num + 1):
                self.W[j] += -epsilon * dWs[j]
                self.b[j] += -epsilon * dbs[j]

            # Optionally print the loss.
            # This is expensive because it uses the whole dataset, so we don't want to do it too often.
            if print_loss and i % 1000 == 0:
                dataLoss = self.calculate_loss(X, y)
                print("Loss after iteration %i: %f" % (i, self.calculate_loss(X, y)))
                index.append(i)
                dataLosses.append(dataLoss)
        return dataLosses, index


class Layer(object):
    def __init__(self, X, W, b, type='tanh'):
        """
        :param X:       neuron input
        :param W:       weight of input layer
        :param b:       bias of input layer
        :param type:    activation function type
        """
        self.X = X
        self.W = W
        self.b = b
        self.actFun_type = type

    def feedforward(self, actFun):
        """
        Implements the feedforward steps for a single layer in the network

        Args:
            actFun: activation function

        Returns:
            z, a
        """
        self.z = np.dot(self.X, self.W) + self.b
        self.a = actFun(self.z)
        return self.z, self.a

    def backprop(self, delta):
        """
        Implements the feedforward steps for a single layer in the network

        Args:
            delta:

        Returns: dL/dW, dL/b
        """
        dW = np.dot(self.X.T, delta)  # dW1 = dL/dW1
        db = np.sum(delta, axis=0, keepdims=True)  # db1 = dL/db1

        return dW, db

    def get_z(self):
        return self.z


def main():
    """
    This network will be trained with the Make_Moons dataset using different number of layers,
    different layer sizes, different activation functions and, in general, different network
    configurations.
    """
    # generate and visualize Make-Moons dataset
    X, y = generate_data()
    plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    plt.title("Original Data")
    plt.savefig(figureFolder + "/original_data.png")
    plt.show()

    # start training
    data = {}
    df = pd.DataFrame(data)
    actType = ["tanh", "sigmoid", "relu"]
    # units = 3
    for act in actType:
        for i in range(1, 5, 1):
            for units in range(3, 10, 1):
                string = "Type: " + act + " Hidden Layer: " + str(i) + " Units: " + str(units)
                print("\n\n" + string)
                model = DeepNeuralNetwork(nn_input_dim=2, nn_hidden_layer_num=i, nn_hidden_dim=units, nn_output_dim=3,
                                          actFun_type=act)
                df[act + "_" + str(i) + "_" + str(units)], index = model.fit_model(X, y, epsilon=0.01, num_passes=20000)
                model.visualize_decision_boundary(X, y, act, str(i), str(units), string)
    df["Iteration"] = index
    df.set_index("Iteration", inplace=True)
    df.to_csv("dataLoss_n_n.csv", index=True)


if __name__ == "__main__":
    main()
