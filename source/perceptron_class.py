import numpy as np
from random import choice
import matplotlib.pyplot as plt

def train_perceptron(training_data):
    '''
    Train a perceptron model given a set of training data
    :param training_data: A list of data points, where training_data[0]
    contains the data points and training_data[1] contains the labels.
    Labels are +1/-1.
    :return: learned model vector
    '''
    X = training_data[0]
    y = training_data[1]
    model_size = X.shape[1]
    w = np.zeros(model_size)#np.random.rand(model_size)
    iteration = 1
    while True:
        # compute results according to the hypothesis

        # get incorrect predictions (you can get the indices)

        # Check the convergence criteria (if there are no misclassified
        # points, the PLA is converged and we can stop.)

        # Pick one misclassified example.

        # Update the weight vector with perceptron update rule

        iteration += 1

    return w

def print_prediction(model,data):
    '''
    Print the predictions given the dataset and the learned model.
    :param model: model vector
    :param data:  data points
    :return: nothing
    '''
    result = np.matmul(data,model)
    predictions = np.sign(result)
    for i in range(len(data)):
        print("{}: {} -> {}".format(data[i][:2], result[i], predictions[i]))


if __name__ == '__main__':

    rnd_x = np.array([[0,1,1],\
                      [0.6,0.6,1],\
                      [1,0,1],\
                      [1,1,1],\
                      [0.3,0.4,1],\
                      [0.2,0.3,1],\
                      [0.1,0.4,1],\
                      [0.5,-0.1,1]])

    rnd_y = np.array([1,1,1,1,-1,-1,-1,-1])
    rnd_data = [rnd_x,rnd_y]

    trained_model = train_perceptron(rnd_data)
    print_prediction(trained_model, rnd_x)


