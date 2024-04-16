import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utilities import Utilities


class Linear(object):

    def __init__(self, n_features):

        self.n_features = n_features

        self.weights, self.bias = self.__initialize_weights_and_biases()

    def __initialize_weights_and_biases(self):

       std_v = 1.0 / np.sqrt(1.0)

       weights = np.random.uniform(low=-std_v, high=std_v, size=self.n_features)
       bias = np.random.uniform(low=-std_v, high=std_v)

       return weights, bias

    def forward(self, x):

        out = np.dot(x, self.weights) + self.bias

        return out

    def backward(self, x, dout, learning_rate, weight_decay=None):

        dweights = (1.0 / x.shape[0]) * np.dot(x.T, dout)
        dbias = (1.0 / x.shape[0]) * np.sum(dout)

        if weight_decay is not None:
            dweights += weight_decay * self.weights

        self.weights -= learning_rate * dweights
        self.bias -= learning_rate * dbias
 
    def __call__(self, x):

        return self.forward(x)


class Architecture(object):

    def __init__(self, n_features):

        self.n_features = n_features

        self.linear = Linear(self.n_features)

    def __sigmoid(self, x):

        return 1.0 / (1.0 + np.exp(-x))

    def __bce_loss(self, preds, labels, eps=1e-9):

        loss = -1.0 * np.mean((labels * np.log(preds + eps)) + ((1.0 - labels) * np.log(1.0 - preds + eps)))

        return loss

    def forward(self, x, y=None):

        out = self.linear(x)
        out = self.__sigmoid(out)

        loss = None
        if y is not None:
            loss = self.__bce_loss(out, y)

        return out, loss

    def backward(self, x, out, y, learning_rate, weight_decay=None):

        dout = out - y
        self.linear.backward(x, dout, learning_rate, weight_decay=weight_decay)

    def __call__(self, x, y=None):

        return self.forward(x, y=y)


class Model(object):

    def __init__(self, n_features, probability_threshold):

        self.n_features = n_features

        self.probability_threshold = probability_threshold

        self.arch = Architecture(self.n_features)

    def __minibatch(self, x, y, batch_size):

        if batch_size is None:
            return [x, ], [y, ]

        n_samples = len(x)
        n_splits = float(n_samples) / batch_size

        x = np.split(x, n_splits)
        y = np.split(y, n_splits)

        return x, y

    def __shuffle(self, x, y):

        indices = list(range(len(x)))
        np.random.shuffle(indices)

        x = x[indices]
        y = y[indices]

        return x, y

    def fit(self, train_x, train_y, val_x, val_y, n_epochs, early_stopping_n_epochs, learning_rate, learning_rate_patience, learning_rate_drop_factor, batch_size=None, weight_decay=None):

        best_val_accuracy = -np.inf
        best_val_loss = np.inf
        best_epoch = 0
        best_params = None

        for epoch in range(n_epochs):
            mbs_train_x, mbs_train_y = self.__minibatch(train_x, train_y, batch_size)

            train_losses = []
            train_accuracies = []

            for mb_train_x, mb_train_y in zip(mbs_train_x, mbs_train_y):
                mb_train_x, mb_train_y = self.__shuffle(mb_train_x, mb_train_y)

                mb_train_preds, mb_train_loss = self.arch(mb_train_x, y=mb_train_y)
                self.arch.backward(mb_train_x, mb_train_preds, mb_train_y, learning_rate=learning_rate, weight_decay=weight_decay)

                mb_train_accuracy = (mb_train_preds >= self.probability_threshold).astype(np.float32) == mb_train_y

                train_losses.append(mb_train_loss)
                train_accuracies.extend(list(mb_train_accuracy))

            train_loss = np.mean(train_losses)
            train_accuracy = np.mean(train_accuracies)

            val_preds, val_loss = self.arch(val_x, y=val_y)

            val_accuracy = np.mean((val_preds >= self.probability_threshold).astype(np.float32) == val_y)

            if val_accuracy == 1.0:
                best_val_accuracy = val_accuracy
                best_val_loss = val_loss
                best_epoch = epoch
                best_params = (self.arch.linear.weights.copy(), self.arch.linear.bias.copy())
                break

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_val_loss = val_loss
                best_epoch = epoch
                best_params = (self.arch.linear.weights.copy(), self.arch.linear.bias.copy())
                print("Epoch : {} - Train Loss : {} - Val Loss : {} - Train Accuracy : {} - Val Accuracy : {}".format(epoch, round(train_loss, 5), round(val_loss, 5),
                                                                                                                      round(train_accuracy, 5), round(val_accuracy, 5)))
            else:
                if (epoch - best_epoch) >= learning_rate_patience:
                    prev_learning_rate = learning_rate
                    learning_rate *= learning_rate_drop_factor

                    # print("Learning Rate : {} -> {}".format(prev_learning_rate, learning_rate))

                if (epoch - best_epoch) >= early_stopping_n_epochs:
                    break

        self.arch.linear.weights, self.arch.linear.bias = best_params

    def predict(self, x):

        preds, _ = self.arch(x, y=None)
        preds = (preds >= self.probability_threshold).astype(np.float32)

        return preds

    def analyse(self, x, y):

        preds, loss = self.arch(x, y=y)
        preds = (preds >= self.probability_threshold).astype(np.float32)

        accuracy = np.mean(preds == y)

        print("Loss : {} - Accuracy : {}".format(loss, accuracy))


if __name__ == "__main__":
    seed = 73
    np.random.seed(seed)

    probability_threshold = 0.5
    n_epochs = 100000
    learning_rate = 50.0
    early_stopping_n_epochs = 200
    learning_rate_patience = 100
    learning_rate_drop_factor = 0.5
    batch_size = None # NOTE: None to disable #32
    weight_decay = 1e-3  # NOTE: None to disable # 1e-8

    train_data, train_labels, val_data, val_labels, classes = Utilities.get_rice_cammeo_osmancik_data("data/rice_cammeo_osmancik/Rice_Cammeo_Osmancik.arff", validation_ratio=0.15)

    print("N Train : {}".format(len(train_data)))
    print("N Val   : {}".format(len(val_data)))

    n_features = train_data.shape[1]

    model = Model(n_features, probability_threshold)

    model.fit(train_data, train_labels, val_data, val_labels, n_epochs, early_stopping_n_epochs, learning_rate, learning_rate_patience, learning_rate_drop_factor, batch_size=batch_size, weight_decay=weight_decay)
    model.analyse(val_data, val_labels)
