#!/usr/local/bin/python3
import numpy as np
from mlp import Multilayer

'''
This file was provided by the instructors for the course INF4490.
'''

filename = 'movements_day1-3.dat'

movements = np.loadtxt(filename,delimiter='\t')

# Subtract arithmetic mean for each sensor. We only care about how it varies:
movements[:,:40] = movements[:,:40] - movements[:,:40].mean(axis=0)

# Find maximum absolute value:
imax = np.concatenate(  ( movements.max(axis=0) * np.ones((1,41)) ,
                          np.abs( movements.min(axis=0) * np.ones((1,41)) ) ),
                          axis=0 ).max(axis=0)
						  

# Divide by imax, values should now be between -1,1
movements[:,:40] = movements[:,:40]/imax[:40]


# Generate target vectors for all inputs 2 -> [0,1,0,0,0,0,0,0]
target = np.zeros((np.shape(movements)[0],8));
for x in range(1,9):
	indices = np.where(movements[:,40]==x)
	target[indices,x-1] = 1


# Randomly order the data
order = list(range(np.shape(movements)[0]))
np.random.shuffle(order)

inputs = movements[order,:][:,:-1]
targets = target[order,:]

# Try networks with different number of hidden nodes:
hidden = 8


# Initialize the network:
net = Multilayer(inputs, targets, hidden, momentum = 0.0, crossvalidation=False)

# Training and testing:
net.run()

