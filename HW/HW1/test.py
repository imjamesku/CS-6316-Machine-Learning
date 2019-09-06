import RegressionTemplate

x, y = RegressionTemplate.load_data_set("regression-data.txt")
# print(x)
# print(y)
# print(x.shape)
theta = RegressionTemplate.normal_equation(x, y)
print(theta)