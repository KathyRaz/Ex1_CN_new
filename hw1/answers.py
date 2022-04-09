r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**
We observe that increasing k beyond 5 results in a sharp and continuous decline in the KNN model performance on the unseen data.
When k=1 the model classifies each test sample based on is closest smaple from the training set. Given the data in MNIST is of handwritten digits, it appears that the most similar training set samples (1-3) are most often the same digit as the test example.
When increasing k to say 100, the many training samples have the incorrect label negatively affect KNN accuracy.
At the extremal values of k:
If k is very small the danger is that we overfit, while if k is very large (number of data points) then we aren't really looking at any neighbors.

"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**

The selection of $\Delta > 0$ is arbitrary for the Linear SVM loss $L(\mat{W})$ we have defined, since the _direction_ of
the optimal $w$ (where $L$ is minimal) would be the same for any given $\Delta$. Only the direction affects the inference predictions.

"""

part3_q2 = r"""
**Your answer:**

1. The classifier has learnt a "generic" image for each of the classes. During inference it "compares" (dot product) the image 
with each of the classes, and returns the image that is closest to the generic class image that it has learned. For that reason we can see,
for example, that the model predicts 7 when the number is 2: in the instance that can be seen above we can see that 2 is similar to the
generic image that the model has learned (7 with no cross on it).

2. This interpretation is different since in the KNN implementation, we compare the given sample to its K closest samples.
In the linear classifier, we are comparing the given sample to a pre-learnt generic sample.

"""

part3_q3 = r"""
**Your answer:**

1. Based on the graph of the training set loss, the learning rate seems to be good. If it were too low, the loss would not
have been close to convergence (it would still be declining more steeply with the same number of epochs). If it were too hih,
we may have seen the loss zig-zaging in a non-stable way, since the optimal loss would be missed at every step.

2. The model is slightly overfitted to the training set. We can see the the training accuracy is higher than the validation
accuracy, but the difference is not very large (87.% vs 90%).

"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
**Your answer:**

The ideal pattern in the residual plot is for the residual values to be equally and randomly spaced around the horizontal axis (x-axis). This is given that the vertical axis (y - y_hat) indicates the residual error. So we would like our residuals to be around the lower single digits of the vertical axis (y-axis) and equally distributed on both sides of the horizontal axis (x-axis).

By observing the performance of our trained linear regression model on the hidden test-set (orange points) we observe that it performs better than our previous top-5 features heuristic. On the test set is down to 12.25 versus 26.98. We observe test predictions of the trained model are closer to the x-axis. Last, there are fewer outliers in our trained model than in the plot of the top-5 features heuristic.

"""

part4_q2 = r"""
**Your answer:**
1. Logarithmic scale enables us to search a bigger space quickly. At first, we do not know the range for the lambda 
hyper-parameter. A faster way than linspace is trying dramatically different values at different scale, e.g. 1, 10, 100, 1000, 
which come from a logarithmic scale.

2. During the cross validation (training) phase, we performed 3-fold cross validation on the training set.
During each phase we performed a grid search over two hyperparams: the degree & lambda.
For the degree we go over 3 potential values [1 2 3].
For lambda we go over all 20 potential values in np.logspace(-3, 2, base=10, num=20).
In total, we have 3 * 20 hyperparam combinations over 3-fold cross validation culminating in 3*3*20 = 180 total fits.
"""

# ==============
