#!/usr/local/bin/python3

# Oystein Kapperud, 2017

import numpy as np
import copy

class Multilayer:
	'''
	A class for supervised training and testing of a multilayer neural network with one hidden layer.
	'''
	def __init__(self, inputs, targets, nhidden, beta = 1.0, eta = 0.1, momentum = 0.0, iterations = 100, crossvalidation = False, k = 10):
		"""
		The constructor for the class Multilayer.
		
		in the following: i is the number of input parameters, n is the number of complete input sets, and o is the number of output classifications.
		Parameters:
		inputs (numpy array): (n,i) numpy array with input data
		targets (numpy array): (n,o) numpy array with binary values (0/1) where the 1 in row x corresponds to the target of the input data in row x of 'inputs'
		nhidden (int): the number of nodes in the hidden layer
		beta (float): parameter in the activation function (default: 1.0)
		eta (float): learning rate (default: 0.1)
		momentum (float): momentum parameter (default: 0.0)
		iterations (int): the number of the times the network is trained on the full dataset before validation error is calculated
		crossvalidation (boolean): if True the program will perform a k-fold cross-validation, if False the data will be split training:validation:test 50:25:25 (default: False)
		k (int): the number of subsets in the crossvalidation (if crossvalidation is True) (default: 10)
		"""
		
		self.k = k
		self.crossvalidation = crossvalidation
		self.nhidden = nhidden
		
		self.beta = beta
		self.eta = eta
		self.momentum = momentum
		self.iterations = iterations
		
		# Randomize input/target order
		order = list(range(np.shape(inputs)[0]))
		np.random.shuffle(order)
		self.inputs = inputs[order,:]
		self.targets = targets[order,:]
		
		self.old_output_update = 0
		self.old_hidden_update = 0
		
		if not crossvalidation:
			# Split data into 3 sets
			
			# Training set
			self.train_inputs = inputs[::2]
			self.train_targets = targets[::2]

			# Validation checks how well the network is performing and when to stop
			self.validation_inputs = inputs[1::4]
			self.validation_targets = targets[1::4]

			# Test data is used to evaluate how good the completely trained network is.
			self.test_inputs = inputs[3::4]
			self.test_targets = targets[3::4]

		else:
			# K-fold validation
			
			print('K-fold cross-validation, k=%d' %(self.k))
			print(self.nhidden, 'hidden nodes\n')
			print('Training network. Please wait...')
			self.input_subsets = []
			self.target_subsets = []
			for i in range(k): # Divide inputs and targets into k subsets
				self.input_subsets.append(self.inputs[i::k])
				self.target_subsets.append(self.targets[i::k])
			
			

	def run(self):
		if not self.crossvalidation:
			self.initialize()
			self.stop = False
			print('50:25:25 split')
			print(self.nhidden, 'hidden nodes\n')
			print('Training network. Please wait...')
			while not self.stop:
				self.train()
			self.confusion(self.test_inputs, self.test_targets)
		
		else:
			
			# k-fold cross-validation
			
			self.best_validation_error = float('Inf')
			self.best_hidden_weights = None
			self.best_output_weights = None
			for i in range(self.k): # subsets[i] and subsets[i+1]/[0] will serve as validation and test subsets, respectively, for k separate runs.
				training_input_subsets = copy.deepcopy(self.input_subsets)
				training_target_subsets = copy.deepcopy(self.target_subsets)
				
				validation = i
				if i +1 != self.k:
					test = i+1
				else:
					test = 0
				
				self.validation_inputs = self.input_subsets[validation]
				self.validation_targets = self.target_subsets[validation]
				
				self.test_inputs = self.input_subsets[test]
				self.test_targets = self.target_subsets[test]
				
				
				
				if validation > test:
					del training_input_subsets[validation]
					del training_target_subsets[validation]
				
					del training_input_subsets[test]
					del training_target_subsets[test]
				else:
					del training_input_subsets[test]
					del training_target_subsets[test]
					
					del training_input_subsets[validation]
					del training_target_subsets[validation]
					
				
				self.train_inputs = np.concatenate(training_input_subsets) # self.train_inputs is now a concatenation of all the subsets not not chosen for validation  or testing
				self.train_targets = np.concatenate(training_target_subsets) # same
				self.initialize()
				self.stop = False
				while not self.stop:
					self.train()
					#print(self.validation_errors[-1])
					
				print('\n%d' %(i))
				validation_error = self.validation_errors[-1]
				print('Validation error: ', self.validation_errors[-1])
				if validation_error < self.best_validation_error: # store the relevant parameters for the the run with the lowest validation error
					self.best_validation_error = validation_error
					self.best_hidden_weights = self.hidden_weights
					self.best_output_weights = self.output_weights
					self.best_test_inputs = self.test_inputs
					self.best_test_targets = self.test_targets
					self.best_i = i
				self.confusion(self.test_inputs, self.test_targets)
			self.hidden_weights = self.best_hidden_weights # set the weights to the values found in the run with lowest validation error
			self.output_weights = self.best_output_weights
			print('\nLowest validation error: nr ', self.best_i)
			self.confusion(self.best_test_inputs, self.best_test_targets)
					
					
				
				
				
	def initialize(self):
		'''
		This function does some necessary initializations
		'''
		self.ninput_sets_train = self.train_inputs.shape[0] # number of input sets in the training subset
		self.ninput_sets_validation = self.validation_inputs.shape[0] # number of input sets in the validation subset
	
		self.noutputs = self.train_targets.shape[1] # number of output nodes
		self.ninput_nodes = self.train_inputs.shape[1] # number if input nodes

		self.train_inputs = np.append(self.train_inputs, np.ones((self.ninput_sets_train, 1), dtype = np.float64), axis = 1)	# add a bias input node with constant value 1
		self.validation_inputs = np.append(self.validation_inputs, np.ones((self.ninput_sets_validation, 1), dtype = np.float64), axis = 1) # # add a bias input node with constant value 1
	
		self.train_errors = []
		self.validation_errors = []
		
		self.hidden_weights = np.random.rand(self.ninput_nodes +1, self.nhidden) # set the weights (including a bias weight) for the input to the hidden layer to random numbers between -1 and 1
		self.output_weights = np.random.rand(self.nhidden +1, self.noutputs) # # set the weights (including a bias weight) for the input to the output layer to random numbers between -1 and 1
		
		# run the input forwards through the network once to get the errors with the random weights (pre-training)
		train_outputs = self.forward(self.train_inputs)
		validation_outputs = self.forward(self.validation_inputs)
		self.train_errors.append(self.find_error(train_outputs, self.train_targets))
		self.validation_errors.append(self.find_error(validation_outputs, self.validation_targets))			
	
		
	def find_error(self, output_set, target_set):
		'''
		This function takes as argument an output set (output_set) and a target set (target_set) and returns the error.
		'''
		return np.sum( (output_set-target_set)**2 )
        
		
	def train(self):
		'''
		this function trains the network
		parameters:
		iterations (int): the number of times the network trains on the full input set before the validation error is calculated
		
		this function is called repeatedly from the 'run' function until the validation errors starts to increase, at which point the boolean
		vairable self.stop is set to True
		'''
		for i in range(self.iterations):
			self.train_outputs = self.forward(self.train_inputs)
			self.find_delta()
			self.update_weights()
		self.train_errors.append(self.find_error(self.train_outputs, self.train_targets))
		
		validation_outputs = self.forward(self.validation_inputs)
		self.validation_errors.append(self.find_error(validation_outputs, self.validation_targets))
		
		
		if self.validation_errors[-1] > self.validation_errors[-2]:
			self.validation_errors = self.validation_errors[:-1]
			self.stop = True
		
		
			

	def forward(self, input_set):
		'''
		this function computes and returns the output of a given input, given the current weights in the network
		'''
		
		hiddenlayer_summed_weighted_inputs = np.dot(input_set, self.hidden_weights)
		self.hidden_activations = self.g(hiddenlayer_summed_weighted_inputs)
		self.hidden_activations = np.append(self.hidden_activations, np.ones((input_set.shape[0],1), dtype = np.float64), axis = 1) 
		outputlayer_summed_weighted_inputs = np.dot(self.hidden_activations, self.output_weights)
		
		output_activations = self.g(outputlayer_summed_weighted_inputs)
		return output_activations
	
	def find_delta(self):
		'''
		this function calculates the delta values
		'''
		self.output_delta = (self.train_outputs - self.train_targets)*self.g_derivative(self.train_outputs)
		self.hidden_delta = np.dot(self.output_delta, np.transpose(self.output_weights)[:,:-1])*self.g_derivative(self.hidden_activations[:,:-1])
		
	def update_weights(self):
		'''
		this function udates the weights
		'''
		self.output_update = self.eta*np.dot(np.transpose(self.hidden_activations), self.output_delta)
		self.hidden_update = self.eta*np.dot(np.transpose(self.train_inputs), self.hidden_delta)
		self.output_weights = self.output_weights - self.output_update + self.momentum*self.old_output_update
		self.hidden_weights = self.hidden_weights -  self.hidden_update + self.momentum*self.old_hidden_update
		self.old_outout_update = self.output_update
		self.old_hidden_update = self.hidden_update
		
	def confusion(self, inputs, targets):
		'''
		this function calculates and prints the confusion matrix and the percentage of correct outputs
		'''
		
		inputs = np.append(inputs, np.ones((inputs.shape[0], 1), dtype = np.float64), axis = 1) # add bias input
		outputs = self.forward(inputs)
		outputs_list = np.argmax(outputs, axis=1)
		targets_list = np.argmax(targets, axis=1)
		correct = 0
		wrong = 0
		
		confusion = np.zeros((targets.shape[1], outputs.shape[1]), dtype = np.int)
		for i in range(len(outputs_list)):
			target = targets_list[i]
			output = outputs_list[i]
			confusion[target][output] += 1
			if target == output:
				correct += 1
			else:
				wrong += 1

			

		print('\nTest set confusion matrix:')
		print(confusion)
		percent_correct = 100*correct/(correct+wrong)
		print('%.2f percent correct outputs' %(percent_correct))
		

	
	def g(self,summed_weighted_inputs):
		'''
		activation function (sigmoid)
		'''
		return 1/(1+np.exp(-self.beta*summed_weighted_inputs))


	def g_derivative(self, activations):
		'''
		the derivative of the activation function
		'''
		return self.beta*activations*(1-activations)
