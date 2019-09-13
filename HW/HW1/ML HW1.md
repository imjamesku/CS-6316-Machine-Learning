# ML HW1
## 1 Linear Algebra Review 
![](https://i.imgur.com/smeTQ0z.jpg)
![](https://i.imgur.com/NtLcl6J.jpg)
## 2 Linear Regression Model Fitting (Programming)
### Data Set
![](https://i.imgur.com/7luy5pD.png)

### Normal Equation
theta = `[[3.00774324]
 [1.69532264]]`
![](https://i.imgur.com/OAAGEda.png)


### Gradient Descent
#### Results
learning rate: 0.01
number of iterations: 10
theta = `array([[-113.02169208],
       [ -59.01472316]])`
![](https://i.imgur.com/Hy8dIrK.png)
![](https://i.imgur.com/mZeUaA2.png)



learning rate: 0.006
number of iterations: 10
theta = `array([[2.94306258],
       [1.81368982]])`
![](https://i.imgur.com/aNv3sFO.png)
![](https://i.imgur.com/SLWEI8Z.png)



learning rate: 0.003
number of iterations: 10
theta = `array([[2.91107894],
       [1.87981306]])]`
![](https://i.imgur.com/Al1Efp2.png)
![](https://i.imgur.com/REFq9dm.png)

#### Observation
As shown above, too large a learning rate resulted in the loss function not converging, while learning rates of 0.003 and 0.006 were small enough for the loss function to converge.



### Stochastic Gradient Descent
learning rate: 1
number of iterations: 10
theta = `array([[3.08500241],
       [1.73524859]])`
![](https://i.imgur.com/AgfQQRO.png)
![](https://i.imgur.com/5aRIhnm.png)



learning rate: 0.1
number of iterations: 10
theta = `array([[2.9940233 ],
       [1.68317826]])`
![](https://i.imgur.com/ZJlSxrQ.png)
![](https://i.imgur.com/HdvpUSh.png)



learning rate: 0.01
number of iterations: 5
theta = `array([[2.93335447],
       [1.83177765]])`
![](https://i.imgur.com/K6Ec8D2.png)
![](https://i.imgur.com/dVArK5Z.png)

#### Observation
As we can see in the above figure, even with a learning rate of 1, SGD still managed to fit pretty well to the data points. But if we look at the loss function, we can see that does not increase as epoch increases, so 1 is still too large for the loss function to converge. When the learning rate is brought down to 0.1 or 0.01, the loss function converges and no longer jumps up and down.



### Minibatch Gradient Descent
#### Result
learning rate: 0.001,
number of iterations: 20,
batch size: 50,
theta = `array([[2.88485029],
       [1.90411717]])`
![](https://i.imgur.com/ups1FOY.png)
![](https://i.imgur.com/K0Skvsl.png)


learning rate: 0.01
number of iterations: 20,
batch size: 50
theta = `array([[3.00127283],
       [1.71374226]])`
![](https://i.imgur.com/91G4fox.png)
![](https://i.imgur.com/d2jkhOM.png)


#### Observation
Converged faster when the learning rate was larger.


## 3 Sample Exam Questions:

### 3.1
The MSE will be 0 as every point in the data set sits perfectly on the same line, so we can use linear regression to find a line to fit the data set perfectly.

### 3.2
#### (a)
![](https://i.imgur.com/6uOSEGY.jpg)

#### (b)
![](https://i.imgur.com/gLgW9aW.jpg)
![](https://i.imgur.com/YUgEUMB.jpg)

\(c\)
The second model, since it produced a smaller error

### 3.3
(e) A, because the smaller the training data set is, the easier it is for us to find a line that fits the data points. Two points, for example, can be perfectly fit with 0 training error.

(f) B, because as we increase the size of the training data set, the model should generalize better and be able to make better predictions on testing data.
