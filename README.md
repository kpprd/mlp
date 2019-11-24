# mlp
Python version: 3.7
Testrun: Macbook Pro OS 10.15.1

A program for supervised training and testing of a multilayer neural network with one hidden layer (see e.g. Marsland 2015)

How to run the program:
Create an instance of the class Multilayer with constructor arguments corresponding to the desired properties (see documentation for details).

Brief description:
The file movements.py reads the input file, as in the pre-code. Note that I have moved the splitting of the data into sets to the constructor of the Multilayer class, as this makes it easier to deal with the k-fold validation.

All the calculations are done with matrix operations using numpy arrays. The weights are updated only after the accumulated error has been calculated for the full training set once. Then this is repeated for the given number of iterations, at which time the validation error is calculated. If the validation error is higher than after the precious calculation (indicating that the model has started to overfit to the training data), the program is stopped and the confusion matrix and percent of correct outputs (in the test set) is calculated and printed.

For the k-fold cross-validation, the data is divided into k subsets. The program loops over values of i ranging from 0 to k-1, so that for each iteration subset i and subset i+1 (or 0 if i is the last subset) are designated as the validation set and test set, respectively. The rest of the susbsets are merged to form the training set. The program then train the model for each value of i in the same manner as described above, and reports the confusion matrices and correct percentages (for the various test sets). It also reports the confusion matrix for the model with the lowest validation error (as per Marshland 2.2.2)

References:
Marsland, Stephen (2015) Machine Learning: An Algorithmic Approach. CRC Press
