# Machine Learning HW2 Ridge Regression

import matplotlib.pyplot as plt
import numpy as np

# Parse the file and return 2 numpy arrays


def load_data_set(filename):
    arr = np.loadtxt(filename)
    x, y = np.split(arr, [-1], axis=1)
    return x, y

# Split the data into train and test examples by the train_proportion
# i.e. if train_proportion = 0.8 then 80% of the examples are training and 20%
# are testing


def train_test_split(x, y, train_proportion):
    num_train = int(x.shape[0] * train_proportion)
    x_train, x_test = x[:num_train, :], x[num_train:, :]
    y_train, y_test = y[:num_train, :], y[num_train:, :]
    return x_train, x_test, y_train, y_test

# Find theta using the modified normal equation
# Note: lambdaV is used instead of lambda because lambda is a reserved word in python


def normal_equation(x, y, lambdaV):
    # your code
    x_t = np.transpose(x)
    n = x.shape[1]
    beta = np.dot(np.dot(np.linalg.inv(
        np.dot(x_t, x) + lambdaV * np.identity(n)), x_t), y)
    return beta

# Extra Credit: Find theta using gradient descent


def gradient_descent(x, y, lambdaV, num_iterations, learning_rate):
    num_features = x.shape[1]
    beta = np.array([[0] for i in range(num_features)])
    for _ in range(num_iterations):
        gradient_sum = np.array([[0.0] for i in range(num_features)])
        for row, label in zip(x, y):
            gradient_sum = gradient_sum - learning_rate * \
                row[:, np.newaxis] * (np.dot(row, beta) - label)
        beta = beta + gradient_sum
    return beta

# Given an array of y and y_predict return loss


def get_loss(y, y_predict):
    diff = y - y_predict
    loss = np.dot(diff.T, diff) / len(y)
    return loss[0][0]

# Given an array of x and theta predict y


def predict(x, theta):
    # your code
    y_predict = x.dot(theta)
    return y_predict

# Find the best lambda given x_train and y_train using 4 fold cv


def cross_validation(x_train, y_train, lambdas, n_folds=4):
    valid_losses = []
    training_losses = []
    num_examples = x_train.shape[0]
    num_per_fold = num_examples//n_folds
    # your code
    for lambda_ in lambdas:
        valid_loss_sum = 0
        training_loss_sum = 0
        for i in range(n_folds):
            testing_start = i*num_per_fold
            beta = normal_equation(np.concatenate((x_train[0:testing_start], x_train[testing_start+num_per_fold:]), axis=0), np.concatenate(
                (y_train[0:testing_start], y_train[testing_start+num_per_fold:]), axis=0), lambda_)
            training_loss_sum += get_loss(np.concatenate((y_train[0:testing_start], y_train[testing_start+num_per_fold:]), axis=0), predict(
                np.concatenate((x_train[0:testing_start], x_train[testing_start+num_per_fold:]), axis=0), beta))
            valid_loss_sum += get_loss(y_train[testing_start: testing_start+num_per_fold], predict(
                x_train[testing_start: testing_start+num_per_fold], beta))
        valid_losses.append(valid_loss_sum/n_folds)
        training_losses.append(training_loss_sum/n_folds)

    return np.array(valid_losses), np.array(training_losses)


def bar_plot(best_beta):
    x = range(1, best_beta.shape[0]+1)
    plt.bar(x=x, height=best_beta.flatten())
    plt.title("Final beta bar graph")
    plt.show()


if __name__ == "__main__":

    # step 1
    # If we don't have enough data we will use cross validation to tune hyperparameter
    # instead of a training set and a validation set.
    x, y = load_data_set("dataRidge.txt")  # load data
    x_train, x_test, y_train, y_test = train_test_split(x, y, 0.8)
    # Create a list of lambdas to try when hyperparameter tuning
    lambdas = [2**i for i in range(-3, 9)]
    lambdas.insert(0, 0)
    # Cross validate
    valid_losses, training_losses = cross_validation(x_train, y_train, lambdas)
    # Plot training vs validation loss
    plt.plot(lambdas[1:], training_losses[1:], label="training_loss")
    # exclude the first point because it messes with the x scale
    plt.plot(lambdas[1:], valid_losses[1:], label="validation_loss")
    plt.legend(loc='best')
    plt.xscale("log")
    plt.yscale("log")
    plt.title("lambda vs training and validation loss")
    plt.show()

    best_lambda = lambdas[np.argmin(valid_losses)]

    # step 2: analysis
    normal_beta = normal_equation(x_train, y_train, 0)
    best_beta = normal_equation(x_train, y_train, best_lambda)
    large_lambda_beta = normal_equation(x_train, y_train, 512)
    # your code get l2 norm of normal_beta
    normal_beta_norm = np.linalg.norm(normal_beta)
    # your code get l2 norm of best_beta
    best_beta_norm = np.linalg.norm(best_beta)
    # your code get l2 norm of large_lambda_beta
    large_lambda_norm = np.linalg.norm(large_lambda_beta)
    print(best_lambda)
    print("L2 norm of normal beta:  " + str(normal_beta_norm))
    print("L2 norm of best beta:  " + str(best_beta_norm))
    print("L2 norm of large lambda beta:  " + str(large_lambda_norm))
    print("Average testing loss for normal beta:  " +
          str(get_loss(y_test, predict(x_test, normal_beta))))
    print("Average testing loss for best beta:  " +
          str(get_loss(y_test, predict(x_test, best_beta))))
    print("Average testing loss for large lambda beta:  " +
          str(get_loss(y_test, predict(x_test, large_lambda_beta))))
    bar_plot(best_beta)

    # Step3: Retrain a new model using all sampling in training, then report error on testing set
    # your code !
    final_beta = normal_equation(x_train, y_train, best_lambda)
    print("Final testing loss:  " +
          str(get_loss(y_test, predict(x_test, final_beta))))

    # Step Extra Credit: Implement gradient descent, analyze and show it gives the same or very similar beta to normal_equation
    # to prove that it works
