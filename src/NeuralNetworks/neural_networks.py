"""Neural Networks"""

# Author: Vandana Bachani <vandana.bvj@gmail.com>
# Created on Feb 25, 2012

from __future__ import division
import random
import sys
from math import exp
from copy import deepcopy


class NeuralNetwork:
    """Class implementing Neural Networks.
    
    Parameters
    ----------
    input_data: utilities.InputData
        Array of data points, labels and the attribute map for the given dataset.
    valid: utilities.InputData
        Validation Set
    layers_list: int[], optional
        Number of hidden layers and number of hidden units in each layer.
    learn_rate: float, optional(default = 0.005)
        The learning rate, eta, which dictates the speed of updation of weights.
    epoch: int, optional(default = 500)
        Maximum number of epochs
    stop_criterion: int, optional(default = 20)
        Number of consecutive validation set points of reducing accuracy.
    """
    
    def __init__(self, input_data, valid, layers_list=[], learn_rate=0.005, 
                 epoch=500, stop_criterion=20):
        self.input = input_data.rows
        self.valid = valid
        self.attrmap = input_data.attrmap
        self.layers_list = layers_list
        self.classes = list(input_data.labels)
        self.learn_rate = learn_rate #can be made adaptive
        self.epoch = epoch
        self.stop_criterion = stop_criterion
        self.stop_epoch = 0
        
    def construct_network(self):
        """The method to construct the network structure given the network 
        configuration, i.e. initializing the layers and units.
        
        """
        self.layers = []
        num_of_layers = 0
        ind = 0
        if self.layers_list:
            l = len(self.layers_list)
        else:
            l = 0
        if l > 0:
            num_of_layers = self.layers_list[0]
        if num_of_layers > 0:
            self.layers.append(self.initialize_input_layer(ind, num_of_layers))
            ind += 1
            for i in range(l):
                if l > i+1:
                    self.layers.append(self.initialize_hidden_layer(ind, 
                                                                    self.layers_list[i], 
                                                                    self.layers_list[i+1], 
                                                                    self.layers[ind-1]))
                else:
                    self.layers.append(self.initialize_hidden_layer(ind, 
                                                                    self.layers_list[i], 
                                                                    len(self.classes), 
                                                                    self.layers[ind-1]))
                ind += 1
        else:
            self.layers.append(self.initialize_input_layer(ind, 
                                                           len(self.classes)))
            ind += 1
        self.layers.append(self.initialize_output_layer(ind, 
                                                        self.layers[ind-1]))
        ind += 1
    
    def initialize_hidden_layer(self, ind, num_units, num_units_next, 
                                prev_layer):
        """Method to initialize a hidden layer of the neural network.
        Parameters
        ----------
        ind: int
            Layer Index
        num_units: int
            Number of units in the layer
        num_units_next: int
            Number of units in the layer next to this layer for weights.
        prev_layer: Layer
            Reference to the previous layer in the network structure.
        """
        hidden_layer = Layer(ind, num_units=num_units, 
                            num_units_next_layer=num_units_next, 
                            prev_layer=prev_layer)
        return hidden_layer
    
    def initialize_input_layer(self, ind, num_units_next):
        """Method to initialize the input layer of the neural network.
        Parameters
        ----------
        ind: int
            Layer Index
        num_units_next: int
            Number of units in the layer next to this layer for weights.
        """
        input_layer = Layer(ind, num_units_next_layer=num_units_next, 
                            is_input=True)
        num_units = 0
        labels = ['bias']
        for i in self.attrmap:
            if i['typ'] == 'discrete':
                num_units += len(i['val'])
                lbl = i['name']
                labels.extend([lbl]*len(i['val']))
            else:
                num_units += 1
                labels.append(i['name'])
        input_layer.num_units = num_units
        for i in range(1, num_units+1):
            input_layer.units.append(Unit(1, num_units_next, labels[i]))
        return input_layer
            
    def initialize_output_layer(self, ind, prev_layer):
        """Method to initialize a hidden layer of the neural network.
        Parameters
        ----------
        ind: int
            Layer Index
        prev_layer: Layer
            Reference to the previous layer in the network structure.
        """
        l = len(self.classes)
        output_layer = Layer(ind, num_units=l, prev_layer=prev_layer, 
                             is_output=True)
        return output_layer
    
    def print_network(self):
        """Method to print the network the units with weights learnt.
        """
        for i in self.layers:
            i.print_layer()
        
    def train_network(self):
        """The back-propagation algorithm implementation to train the neural
        network.
        
        It uses gradient descent to update the weights of individual units in 
        the network.
        """
        goodnnets = []
        err_increase_count = 0
        prev_i = -1
        prev_mse = sys.maxint
        for i in range(self.epoch):
            for j in self.input:
                for k in self.layers:
                    if k.is_input:
                        k.update_unit_values(j, self.attrmap, True)
                        continue
                    if k.is_output:
                        k.update_unit_values(j, self.classes, True)
                    k.update_unit_values()
                l = len(self.layers)
                output_layer = self.layers[l-1]
                final_errors = []
                for p in output_layer.units:
                    if p.label != "bias":
                        final_errors.append(p.rvalue - p.value)
                        p.delta = (p.rvalue - p.value)
                for k in reversed(self.layers):
                    if not k.is_input:
                        k.prev_layer_ref.update_unit_weights(final_errors, 
                                                         self.learn_rate, k)
            (acc, mse) = self.predict(self.valid)
            if mse <= prev_mse:
                goodnnets.append((deepcopy(self), acc, mse))
                prev_mse = mse
            else:
                if (prev_i+1) == i:
                    err_increase_count += 1
                else:
                    err_increase_count = 0
            prev_i = i
            
            if err_increase_count > self.stop_criterion:
                self.stop_epoch = i
                break
        (goodone, a, m) = min(goodnnets, key=lambda x:x[2])
        self.goodone = goodone
        #self.goodone.print_network()
        #print "Accuracy: ", a, "MSE: ", m
    
    def predict(self, test_data):
        """Method to predict the class for the given test data.
        
        Parameters
        ----------
        test_data: utilities.InputData
            Test data to be classified.
        """
        errors = []
        for i in test_data.rows:
            for k in self.layers:
                if k.is_input:
                    k.update_unit_values(i, self.attrmap, True)
                    continue
                if k.is_output:
                    k.update_unit_values(i, self.classes, True)
                k.update_unit_values()
            l = len(self.layers)
            output_layer = self.layers[l-1]
            value_list = []
            mse = 0
            for k in output_layer.units:
                if k.label != "bias":
                    mse += 0.5*((k.rvalue - k.value)**2)
                    if k.value > 0.5:
                        value_list.append(1)
                    else:
                        value_list.append(0)
            y = decode_output(value_list, self.classes)
            if y != i['target']:
                errors.append(i)
        acc = (1 - float(len(errors))/len(test_data.rows))*100
        #print "Accuracy: ", acc, "MSE: ", mse
        return (acc, mse)


class Layer:
    """Class implementing a layer in the Neural Network.
    
    Parameters
    ----------
    ind: int
        Layer index
    num_units: int, optional(default = 0)
        Number of units in the layer
    num_units_next_layer: int, optional(default = 0)
        Number of units in the layer next to this layer for weights. 0 for 
        output layer.
    prev_layer: Layer
        Reference to the previous layer in the network structure. None for 
        input layer.
    is_input: boolean
        Set to true if layer is the input layer of the network.
    is_output: boolean
        Set to true if layer is the output layer of the network.
    """
    def __init__(self, ind, num_units=0, num_units_next_layer=0, 
                 prev_layer=None, is_input=False, is_output=False):
        self.units = []
        self.ind = ind
        self.num_units = num_units
        self.prev_layer_ref = prev_layer
        self.is_input = is_input
        self.is_output = is_output
        self.num_units_next = num_units_next_layer
        self.biasunit = Unit(1, self.num_units_next, "bias")
        self.units.append(self.biasunit)
        if not is_input:
            for i in range(1, num_units+1):
                self.units.append(Unit(1, self.num_units_next))
        if not is_output:
            self.errors = []
    
    def update_unit_values(self, data=None, classormap=None, init=False):
        """Method to update the values of the units in the layer.
        Parameters
        ----------
        data: List, optional(default = None)
            The list of a training data instance.
        classormap: List, optional(default = None)
            If the layer is input layer attribute map is passed. If the layer 
            is the output layer list of classes is passed.
        init: boolean, optional(default = False)
            Set to true if called to initialize the input and output layers. 
            set to false for the hidden units.
        """
        if self.is_input and init:
            if not data:
                print "pass the training data row"
            x = data['attrs']
            n = 1
            for i in range(len(x)):
                attr = classormap[i]
                l = len(attr['val'])
                if attr['typ'] == 'discrete':
                    values = encode_input(x, classormap, i)
                    for j in range(n, n+l):
                        self.units[j].value = values[j - n]
                    n = n + l
                else:
                    values = x[i]
                    self.units[n].value = values
                    n += 1
            return
        if self.is_output and init:
            values = encode_output(data['target'], classormap)
            for j in range(1, self.num_units+1):
                self.units[j].rvalue = values[j-1]
                
            return 
        for j in range(1, self.num_units+1):
            y = 0
            for i in self.prev_layer_ref.units:
                y += i.weights[j-1] * i.value
            if not self.is_output:
                y = sigmoid(y)
            self.units[j].value = y

    def print_layer(self):
        """Method to print a layer in the network.
        """
        print '[',
        print self.ind, ', units:[',
        l = len(self.units)
        for i in self.units:
            i.print_unit()
            if l > 1:
                print ', ',
            l -= 1
        print ']]'
    
    def update_unit_weights(self, errors, learn_rate, next_layer):
        """Method to update the values of the units in the layer.
        Parameters
        ----------
        errors: List
            The errors in the output units.
        learn_rate: float
            The learning rate which is used to update the weights by 
            back-propagating the error using gradient descent.
        next_layer: Layer
            Reference to the next layer in the network structure.
        """
        for i in self.units:
            error_prop = 0
            for j in range(len(i.weights)):
                if not self.is_input:
                    if i.label != "bias":
                        error_prop += i.weights[j] * next_layer.units[j+1].delta
                
                # the momentum constant, alpha = 0.8
                i.weights[j] += learn_rate * next_layer.units[j+1].delta * i.value + 0.8*i.prevdwts[j]
                i.prevdwts[j] = learn_rate * next_layer.units[j+1].delta * i.value + 0.8*i.prevdwts[j]
            val = i.value * (1-i.value) * error_prop
            i.delta = val


class Unit:
    """Class implementing a unit in the Neural Network.
    
    Parameters
    ----------
    value: float
        Value at the unit.
    lbl: String, optional(default = "")
        Label of the unit, defined only for output units.
    is_output: boolean, optional(default = False)
        Set to true if layer is the output layer of the network.
    """
    def __init__(self, value, n, lbl="", is_output=False):
        self.value = value
        if is_output:
            self.rvalue = 0
        else:
            self.weights = []
            self.prevdwts = []
            for i in range(n):
                self.weights.append(randomweight())
                # prevdwts: previous weight deltas which influence the current 
                # weight updates when using momentum for weight updates.
                self.prevdwts.append(0)
        self.label = lbl
        self.delta = 0
    
    def print_unit(self):
        """Method to print a unit's state in the network.
        """
        print '[',
        print 'label:', self.label,
        print ', value:', self.value,
        if hasattr(self, 'rvalue'):
            print ', rvalue:', self.rvalue,
        else:
            print ', weights:', self.weights,
        print ', delta:', self.delta,
        print ']',

        
def encode_input(input_data_attrs, attrmap, i):
    """General method to encode a discrete attributes values using one-hot 
    encoding.
    Parameters
    ----------
    input_data_attrs: List
        List of attribute values at a data point.
    attrmap: Dict
        Attribute Map.
    i: int
        Index in the attribute list of the attribute to be decoded.
    
    Returns
    -------
    List: A bit vector (list of 0/1).
    """
    values = []
    val = input_data_attrs[i]
    attr_vals = attrmap[i]['val']
    l = len(attr_vals)
    j = attr_vals.index(val)
    for k in range(l):
        if k == j:
            values.append(1)
        else:
            values.append(0)
    return values

def encode_output(rvalue, classes):
    """Function to encode a class using one-hot encoding.
    Parameters
    ----------
    rvalue: int
        The index of the class in the classes list, which needs to be encoded.
    classes: List
        List of classes for the given classification problem.
    
    Returns
    -------
    List: A bit vector (list of 0/1)
    """
    i = classes.index(rvalue)
    values = []
    for j in range(len(classes)):
        if j != i:
            values.append(0)
        else:
            values.append(1)
    return values

def decode_output(rvalue_list, classes):
    """Function to decode a class name given the bit vector i.e. the 
    values at the output units.
    Parameters
    ----------
    rvalue_list: List
        The bit vector of the values at all the output nodes.
    classes: List
        List of classes for the given classification problem.
    
    Returns
    -------
    String: The class name as predicted by the neural network output units.
    """
    if 1 not in rvalue_list:
        return ""
    i = rvalue_list.index(1)
    classvalue = classes[i]
    return classvalue

def sigmoid(val):
    """Function to calculate the sigmoid of a number.
    Parameters
    ----------
    val: float
        The real number for which the sigmoid needs to be calculated.
    
    Returns
    -------
    float: sigmoid value of the number.
    """
    val1 = float(1)/(1+exp(-val))
    return val1

def randomweight():
    """Function to get a random weight, for weight initialization for the units 
    of the neural network.
    Returns
    -------
    float: random number between -0.01 and 0.01
    """
    return random.uniform(-0.01, 0.01)