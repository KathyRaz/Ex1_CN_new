r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**
When increasing the K, we will get worse results for the new points in the test part, as increasing k leads to improved generalization for unseen data. When K is small, the model is overfitted, and less generalized. This is because, for small K, we are more influenced by each point separately, including the more anomalous points (outliers). But, eventually, as K is increased further, we decide the label of the unseen data by the majority of points surrounding it, and thus, each point has a smaller weight in the decision. Eventually, when we increase K, the model is too general and could lead to a decrease in accuracy. It is especially decreased when the groups vary in size, and small groups could disappear completely for the unseen data.


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


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part4_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============
