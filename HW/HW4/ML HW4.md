# ML HW4

## 1.1 Hand-crafted Feature Engineering

| Dataset      | Selected Features | Iterations | Test Loss |
| ------------ | ----------------- | ---------- | --------- |
| Circle       | $x_1^2$, $x_2^2$  | 290        | 0.001     |
| Exclusive or | $x_1x_2$          | 578        | 0.001     |
| Gaussian     | $x_1$, $x_2$      | 57         | 0.001     |
| Spiral       | $x_2$             | 1          | 0.465     |

![](https://i.imgur.com/uqF7JCh.jpg)
![](https://i.imgur.com/gAb1CQs.jpg)
![](https://i.imgur.com/tT4vE12.jpg)
![](https://i.imgur.com/7qcBQN8.jpg)

## Interpretation
For the circle, $x_1^2$ and $x_2^2$ work because we can infer the distance between the data point from the origin, which determines the label.

For xor, $x_1 x_2$ works well for classifying the data because the label is essentially the sign of $x_1 x_2$

For the Gaussian dataset, $x1$ and $x2$ works because we just need a linear separation for this dataset.

For the spiral dataset, I couldn't find a good selection of features with 0 hidden layers that yields a good classification.

## 1.2 Regularization
### Task A
#### Circle
![](https://i.imgur.com/85bEEnK.jpg)
![](https://i.imgur.com/aJLyeQw.jpg)
![](https://i.imgur.com/uDZZLfz.jpg)

With the circle dataset, there isn't much of a difference between the three.

#### Exclusive or
![](https://i.imgur.com/eQ0WDSM.jpg)
![](https://i.imgur.com/g5u0ixW.jpg)
![](https://i.imgur.com/fG6Xv9P.jpg)

With the exclusive or dataset, different regularization methods led to slightly different decision boundaries.

#### Gaussian
![](https://i.imgur.com/70HVw84.jpg)
![](https://i.imgur.com/fzvnMag.jpg)
![](https://i.imgur.com/vogaGZa.jpg)

#### Interpretation
These configurations work because regularization helps filter out unimportant features by giving them lower weights.

### Task B
Yes. But there were some features that I didn't select that still got a little bit of weight.

## 1.3 Automated Feature Engineering with Neural Network
### Circle
![](https://i.imgur.com/3rzerVj.jpg)
I used a total of 3 hidden layers, ReLU for the activation function, and L2 for regularization.
This works because the first layer essentially learned boundaries at different slopes and the second layer learned the shape of a circle from those lines.
### Exclusive or
![](https://i.imgur.com/ZkFRFRA.jpg)
Used 2 hidden layers, ReLU for activation, L2 for regularization, with a learning rate of 0.01 and the regularization rate being 0.1.

As we can see in the screen shot, this works because the first layer learned the subspaces separated by the diagonals of the space and the second layer just takes the intersections of them, which have the same shape as xor.

## 1.4 Spiral Challenge

![](https://i.imgur.com/ae8Qy19.jpg)
My interpretation is that the first layer learned some basic/local features of the network and the second layer was then able to extract higher level features from the first layer, allowing the third layer to take advantage of that and learn features that was able to classify the data.

## 2.3 Multilayer Perceptron (MLP)
### Summary
![](https://i.imgur.com/d4s9lQY.jpg)
Just like the previous DNN I built in tensorflow playgound, the the first layer captures some characteristics in the input image and the second layer infers higher level characteristics from the first layer. I also added dropout layers to prevent overfitting.

### Plots
![](https://i.imgur.com/1PNAgRE.jpg)
![](https://i.imgur.com/miYX1Wx.jpg)
The validation accuracy being higher than the training accuracy suggests that my model might have overfit the data, meaning I reducing the number of epochs may increase the accuracy.

### Weights Visualization
![](https://i.imgur.com/WQcFcjY.png)
![](https://i.imgur.com/Xjn0UMu.png)
We can roughly see the outline of a piece of clothing or a top wear in the first picture and the outline of a high-heel in the second picture. This suggests the weights are capturing the certain features that help identify which class the input image is.

### Why isn’t MSE a good choice for a loss function in for this problem?
MSE doesn’t punish misclassifications enough. We want the model to have the highest accuracy possible, so MSE isn't the best choice for classification problems.
## 2.4 Convolutional Neural Network (CNN)
### Summary
![](https://i.imgur.com/JYiXz2n.jpg)
I used 4 convolutional layers to reduce the size of data and extract features, 3 drop out layers to for regularization, and fully connected layers at the end.
### Loss and Accuracy
![](https://i.imgur.com/kqmqdW4.png)
![](https://i.imgur.com/sIRmsKp.png)

Performance did not improve much on the validation data after 5 epochs despite continuous performance improvement in train accuracy. This suggests the model might've been overfitting and more iterations may not improve the accuracy of the model.

### • How many matrices are outputted by your first convolutional layer when it receives a single testing image?
32
### What are the dimensions of these matrices?
$26 \times 26$
### What are the dimensions of one of these matrices after it passes through your first maxpooling layer?
As can be seen in the summary after the first max pooling, the dimensions will be $4 \times 4$

# Sample Questions
## Question 1. Neural Nets and Regression
### (a)
Assigned to all units L
### (b)
First layer: L
Last unit: S
### \(c\)
![](https://i.imgur.com/M8WqR2v.jpg)

## Question 2. Neural Nets
### (a)
![](https://i.imgur.com/62oUOQa.png)
![](https://i.imgur.com/Sdc5fLZ.png)
![](https://i.imgur.com/EcJzlpx.jpg)

