import linearRegression

x, y = linearRegression.load_data_set("regression-data.txt")
# print(x)
# print(y)
# print(x.shape)
theta = linearRegression.normal_equation(x, y)
print(theta)
# theta = linearRegression.stochiastic_gradient_descent(x, y, 0.01, 5)
# theta = linearRegression.gradient_descent(x, y, 0.003, 10)
# thetas = linearRegression.minibatch_gradient_descent(x, y, 0.00001, 10, 10)
# print(thetas)