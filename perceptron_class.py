import numpy as np
import matplotlib.pyplot as plt
from utilities import Utilities


def train_perceptron(training_data, weights_rand_init=False):
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

    if weights_rand_init:
        weights = np.random.rand(model_size)
    else:
        weights = np.zeros(model_size)

    max_n_iter = 2000
    early_stopping_iterations = 1000

    best_accuracy = -np.inf
    best_accuracy_iteration = 0
    best_weights = None
    iteration = 1
    while True:
        # compute results according to the hypothesis
        out = np.matmul(X, weights)

        # get incorrect predictions (you can get the indices)
        out = np.sign(out) # [-1, 0, 1]
        deltas = (y - out) / 2.0 # [-2, -1, 1, 2] -> [-1, -0.5, 0.5, 1]
        incorrect_pred_indices = np.where(deltas != 0.0)[0]

        # Check the convergence criteria (if there are no misclassified
        # points, the PLA is converged and we can stop.)
        n_incorrect_predictions = len(incorrect_pred_indices)

        accuracy = 1.0 - (1.0 * n_incorrect_predictions / len(X))

        print("Iteration : {} - Accuracy : {}".format(iteration, accuracy))

        if n_incorrect_predictions == 0:
            best_accuracy = accuracy
            best_accuracy_iteration = iteration
            best_weights = weights
            break

        if accuracy < best_accuracy: # NOTE: convergence criteria
            if (iteration - best_accuracy_iteration) >= early_stopping_iterations:
                break
        else:
            best_accuracy = accuracy
            best_accuracy_iteration = iteration
            best_weights = weights

        # Pick one misclassified example.
        selected_index = incorrect_pred_indices[np.random.choice(list(range(len(incorrect_pred_indices))))]

        # Update the weight vector with perceptron update rule
        weights += np.sum(np.expand_dims(deltas[[selected_index, ]], axis=1) * X[[selected_index, ]], axis=0) # NOTE: update with single misclassified example.

        if iteration >= max_n_iter:
            break

        iteration += 1

    weights = best_weights

    print("Best Iteration : {} - Best Accuracy : {}".format(best_accuracy_iteration, best_accuracy))

    return weights

def print_prediction(model, data):
    '''
    Print the predictions given the dataset and the learned model.
    :param model: model vector
    :param data:  data points
    :return: nothing
    '''

    result = np.matmul(data, model)
    predictions = np.sign(result)
    for i in range(len(data)):
        print("{}: {} -> {}".format(data[i][:2], result[i], predictions[i]))

    return predictions

def plot_prediction(data, model, predictions):
    plt.scatter([data[i][1] for i in range(len(data)) if predictions[i]==1],[data[i][2] for i in range(len(data)) if predictions[i]==1],marker="o",c="green")
    plt.scatter([data[i][1] for i in range(len(data)) if predictions[i]!=1],[data[i][2] for i in range(len(data)) if predictions[i]!=1],marker="x",c="red")
    x1 = np.linspace(-0.5, 1.2, 50)    
    x2 = -(model[1]*x1 + model[0])/model[2]
    plt.plot(x1,x2)
    plt.xlabel("feature1")
    plt.ylabel("feature2")
    plt.title("Decision boundary and samples")
    plt.show()


if __name__ == '__main__':
    seed = 73
    np.random.seed(seed)


    dataset_type = "small"

    data, labels = Utilities.read_dataset("data/PLA_data-20240326/data_{}.npy".format(dataset_type), "data/PLA_data-20240326/label_{}.npy".format(dataset_type))

    trained_model = train_perceptron((data, labels),True)
   #predictions = print_prediction(trained_model, data)

    #plot_prediction(data, trained_model, predictions)
    print(trained_model)