import MathAndStats as ms
import random


class Neuron:

    def __init__(self, num_inputs, is_logistic):
        self.weights = []
        self.is_logistic = is_logistic
        self.num_inputs = num_inputs
        self.clss = ' '
        self.initializeWeights(-0.3, 0.3)

    def setClass(self, clss):
        self.clss = clss

    def initializeWeights(self, min, max):
        self.weights = []
        # the +1 is to have a bias node
        for i in range(self.num_inputs+1):
            self.weights.append(random.uniform(min, max))

    def getOutput(self, new_inputs):
        # calculate the weighted linear sum
        sum = ms.weightedSum(new_inputs, self.weights, len(new_inputs))
        # if a logistic unit, return the logistic(sum)
        if self.is_logistic:
            return ms.logistic(sum)
        else:
            return sum