import copy
import time
import MathAndStats as ms
import random
import Neuron as unit


class FeedforwardNetwork:

    # output_type is either "classification", "regression", or "autoencoder". Defaults to classification
    # logistic_nodes and logistic_output are boolean variables determining of the nodes are linear or logistic
    def __init__(self, out_k, clsses, output_type, logistic_nodes, logistic_output):
        self.hidden_layers = []
        self.output_layer = []
        self.output_type = output_type
        self.class_list = clsses
        self.logistic_nodes = logistic_nodes
        self.logistic_output = logistic_output
        self.ffn = None
        self.regularize = False
        if output_type == "autoencoder":
            self.name = "autoencoder"
        else:
            self.name = "mlp"
        # in the case of regression, overwrite these inputs
        if output_type == "regression":
            self.out_k = 1
            logistic_output = False
        else:
            self.out_k = out_k

    # given a list of nodes per layer in the network (counting input/output layers), set up the nodes
    def makeNodes(self, nodes_by_layer):
        pass
        self.hidden_layers = []
        for layer in range(1, len(nodes_by_layer)):
            if layer == 1:
                inputs = nodes_by_layer[0]
            else:
                inputs = nodes_by_layer[layer-1]
            # append a list for the nodes of this layer
            self.hidden_layers.append([])
            for node in range(nodes_by_layer[layer]):
                self.hidden_layers[layer-1].append(unit.Neuron(inputs, self.logistic_nodes))
        if len(nodes_by_layer) != 0:
            inputs = nodes_by_layer[-1]
        else:
            inputs = nodes_by_layer[0]
        # output layer
        self.output_layer = []
        for output_node in range(self.out_k):
            self.output_layer.append(unit.Neuron(inputs, self.logistic_output))
            if not ((self.output_type == "regression") or (self.output_type == "autoencoder")):
                self.output_layer[output_node].setClass(self.class_list[output_node])

    # given a list for all weights in the network, assign the weights to the nodes
    def setWeights(self, nodes_by_layer, weights):
        if (self.output_layer == []) or (self.hidden_layers == []):
            self.makeNodes(nodes_by_layer)
        weight_num = 0
        for layer in range(len(self.hidden_layers)):
            for node in range(len(self.hidden_layers[layer])):
                for weight_index in range(len(self.hidden_layers[layer][node].weights)):
                    self.hidden_layers[layer][node].weights[weight_index] = weights[weight_num]
                    weight_num += 1
        for output_node in range(len(self.output_layer)):
            for weight_index in range(len(self.output_layer[output_node].weights)):
                self.output_layer[output_node].weights[weight_index] = weights[weight_num]
                weight_num += 1

    def getWeights(self):
        weight_vector = []
        for layer in range(len(self.hidden_layers)):
            for node in range(len(self.hidden_layers[layer])):
                for weight in range(len(self.hidden_layers[layer][node].weights)):
                    weight_vector.append(self.hidden_layers[layer][node].weights[weight])
        return weight_vector

    def getParameters(self, nodes_by_layer):
        num_weights = 0
        for layer in range(1, len(nodes_by_layer)):
            num_weights += (nodes_by_layer[layer] * (nodes_by_layer[layer - 1] + 1))
        num_weights += self.out_k * (nodes_by_layer[-1] + 1)
        return num_weights, -1, 1

    def getHiddenLayerOutput(self, new_obs, layer_num):
        data = []
        # bias node
        data.append(1.0)
        for hidden_node_num in range(len(self.hidden_layers[layer_num])):
            data.append(self.hidden_layers[layer_num][hidden_node_num].getOutput(new_obs))
        return data

    def predict(self, new_obs):
        if self.output_type == "regression":
            return self.regress(new_obs)
        elif self.output_type == "autoencoder":
            return self.reproduce(new_obs)
        else:
            # I have classify return (class, probability) as a tuple for use in tuning, but
            # predict will simply return class
            return self.classify(new_obs)[0]

    def reproduce(self, new_obs):
        # get the output for each hidden node of each hidden layer
        hidden_outputs = []
        if len(self.hidden_layers) > 0:
            for layer in range(len(self.hidden_layers)):
                # if first layer, take data inputs
                if layer == 0:
                    hidden_outputs = self.getHiddenLayerOutput(new_obs, 0)
                else:
                    # else, take outputs from previous layer
                    hidden_outputs = self.getHiddenLayerOutput(hidden_outputs, layer)
        else:
            # there are no hidden layers
            hidden_outputs = new_obs
        outputs = []
        for output_node in range(self.out_k):
            outputs.append(self.output_layer[output_node].getOutput(hidden_outputs))
        if self.ffn is None:
            return outputs
        # if there is a feedforward network stacked ontop, return the result from that
        return self.ffn.predict(outputs)

    def regress(self, new_obs):
        # get the output for each hidden node of each hidden layer
        hidden_outputs = []
        if len(self.hidden_layers) > 0:
            for layer in range(len(self.hidden_layers)):
                # if first layer, take data inputs
                if layer == 0:
                    hidden_outputs = self.getHiddenLayerOutput(new_obs, 0)
                else:
                    # else, take outputs from previous layer
                    hidden_outputs = self.getHiddenLayerOutput(hidden_outputs, layer)
        else:
            # there are no hidden layers
            hidden_outputs = new_obs
        return self.output_layer[0].getOutput(hidden_outputs)

    def classify(self, new_obs):
        if self.output_type == "autoencoder":
            return self.ffn.classify(new_obs)
        hidden_outputs = []
        if len(self.hidden_layers) > 0:
            for layer in range(len(self.hidden_layers)):
                # if first layer, take data inputs
                if layer == 0:
                    hidden_outputs = self.getHiddenLayerOutput(new_obs, 0)
                else:
                    # else, take outputs from previous layer
                    hidden_outputs = self.getHiddenLayerOutput(hidden_outputs, layer)
        else:
            # there are no hidden layers
            hidden_outputs = new_obs
        #return self.output_layer[0].getOutput(hidden_outputs)
        classes = {}
        for output_num in range(len(self.output_layer)):
            classes[self.output_layer[output_num].clss] = self.output_layer[output_num].getOutput(hidden_outputs)
        decision = sorted(classes.items(), key=lambda elem: elem[1], reverse=True)
        return decision[0]

    def getFitness(self, testing_set, nodes_by_layer, chromosome):
        # assign chromosome
        self.setWeights(nodes_by_layer, chromosome)
        fitness = -1 * ms.testRegressor(self, testing_set)[1]
        self.hidden_layers = []
        self.output_layer = []
        return fitness

    def addFFNetwork(self, ff, tune, training_set, hidden_layer_nodes, eta, alpha_momentum, iterations):
        if self.output_type == "autoencoder":
            # if the given network is already tuned, simply place on top of the autoencoder
            if not tune:
                self.ffn = ff
                return
            training_output = []
            for example in range(len(training_set)):
                predicted = self.predict(training_set[example][:-1])
                predicted.append(training_set[example][-1])
                training_output.append(predicted)
            self.ffn = ff
            print("Tuning stacked FFN on top of",len(self.hidden_layers),"layer autoencoder")
            self.ffn.backpropogation(training_output, hidden_layer_nodes, eta, alpha_momentum, iterations)