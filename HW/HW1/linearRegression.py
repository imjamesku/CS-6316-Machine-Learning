# Machine Learning HW1
import matplotlib.pyplot as plt
import numpy as np
# more imports


# Parse the file and return 2 numpy arrays


def load_data_set(filename):
    arr = np.loadtxt(filename)
    x, y = np.split(arr, [-1], axis=1)
    plt.plot(x[:, 1], y, '.')
    plt.show()
    return x, y

# Find theta using the normal equation


def normal_equation(x, y):
    x_t = np.transpose(x)
    theta = np.dot(np.dot(np.linalg.inv(np.dot(x_t, x)), x_t), y)
    return theta

# Find thetas using stochiastic gradient descent
# Don't forget to shuffle


def stochiastic_gradient_descent(x, y, learning_rate, num_iterations):
    # your code
    thetas = []
    theta = np.array([
        [1],
        [1]
    ])
    for _ in range(num_iterations):
        x, y = unison_shuffled_copies(x, y)
        for row, label in zip(x, y):
            theta = theta - learning_rate * \
                row[:, np.newaxis] * (np.dot(row, theta) - label)
        thetas.append(theta)
        # print(thetas)
    return thetas

# Find thetas using gradient descent


def gradient_descent(x, y, learning_rate, num_iterations):
    thetas = []
    theta = np.array([
        [1],
        [1]
    ])
    for _ in range(num_iterations):
        gradient_sum = np.array([
            [0.0],
            [0.0]
        ])
        for row, label in zip(x, y):
            gradient_sum = gradient_sum - learning_rate * \
                row[:, np.newaxis] * (np.dot(row, theta) - label)
        theta = theta + gradient_sum
        thetas.append(theta)
    return thetas


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

# Find thetas using minibatch gradient descent
# Don't forget to shuffle


def minibatch_gradient_descent(x, y, learning_rate, num_iterations, batch_size):
    thetas = []
    theta = np.array([
        [1],
        [1]
    ])
    gradient_sum = np.array([
        [0],
        [0]
    ])
    count = 0
    for _ in range(num_iterations):
        x, y = unison_shuffled_copies(x, y)
        # print(x, y)
        for row, label in zip(x, y):
            count += 1
            gradient_sum = gradient_sum - learning_rate * \
                row[:, np.newaxis] * (np.dot(row, theta) - label)
            if count == batch_size:
                count = 0
                theta = theta + gradient_sum
                gradient_sum = np.array([
                    [0],
                    [0]
                ])
                # print(theta)
        theta = theta + gradient_sum
        thetas.append(theta)
        # print(thetas)
    return thetas

# Given an array of x and theta predict y


def predict(x, theta):
    # your code
    y_predict = np.dot(x, theta)
    return y_predict

# Given an array of y and y_predict return loss


def get_loss(y, y_predict):
    # your code
    diff = y - y_predict
    loss = np.dot(diff.T, diff) / len(y)
    return loss

# Given a list of thetas one per epoch
# this creates a plot of epoch vs training error


def plot_training_errors(x, y, thetas, title):
    accuracies = []
    epochs = []
    losses = []
    epoch_num = 1
    for theta in thetas:
        losses.append(get_loss(y, predict(x, theta)))
        epochs.append(epoch_num)
        epoch_num += 1
    plt.scatter(epochs, losses)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(title)
    plt.show()

# Given x, y, y_predict and title,
# this creates a plot


def plot(x, y, theta, title):
    # plot
    y_predict = predict(x, theta)
    plt.scatter(x[:, 1], y)
    plt.plot(x[:, 1], y_predict)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    x, y = load_data_set('regression-data.txt')
    # plot
    plt.scatter(x[:, 1], y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Scatter Plot of Data")
    plt.show()

    theta = normal_equation(x, y)
    plot(x, y, theta, "Normal Equation Best Fit")

    # Try different learning rates and number of iterations
    thetas = gradient_descent(x, y, 0.003, 10)
    plot(x, y, thetas[-1], "Gradient Descent Best Fit")
    # print(thetas)
    plot_training_errors(
        x, y, thetas, "Gradient Descent Mean Epoch vs Training Accuracy")

    # Try different learning rates and number of iterations
    thetas = stochiastic_gradient_descent(x, y, 0.01, 10)
    plot(x, y, thetas[-1], "Stochiastic Gradient Descent Best Fit")
    # print(thetas)
    plot_training_errors(
        x, y, thetas, "Stochiastic Gradient Descent Mean Epoch vs Training Accuracy")

    thetas = minibatch_gradient_descent(x, y, 0.01, 20, 50)
    plot(x, y, thetas[-1], "Minibatch Gradient Descent Best Fit")
    # print(thetas)
    plot_training_errors(
        x, y, thetas, "Minibatch Gradient Descent Mean Epoch vs Training Accuracy")
