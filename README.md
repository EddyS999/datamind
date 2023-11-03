# Datamind
This project is an implementation of some mathematics functions for machine learning algorithm. I chose to the approach used in scikit-learn to implement the functions.
## Overview 
For the moment, we only implement a simple Linear Regression which is a fundamental technique in machine learning. It models the relationship between a dependent variable (target) and an independent variable (feature) using a linear equation f(x) = ax+b. This implementation provides a simple and educational example of how the algorithm works.
## Features
- Fits a simple linear regression model to your data.
- Generates cost history to track the optimization process.
- Provides a scoring function to evaluate the model's performance.
## Usage

Here, we're using datamind with a generated dataset from sklearn. 

```python
# Import 
from datamind import SimpleLinearRegression
from sklearn.datasets import make_regression
import matplotlib .pyplot as plt 
import numpy as np

#test dataset
np.random.seed(0)
x,y = make_regression(n_samples=100, n_features = 1, noise = 10) 
X = np.hstack((x, np.ones(x.shape)))#ajout d'un biais
y = y.reshape(y.shape[0],1)

# Linear regression with fit()
model = SimpleLinearRegression()
final = model.fit(X, y)

# Scoring 
prediction = model.main_function(X, final)
print(model.score(y, prediction))

```
## Results
![Figure_2](https://github.com/EddyS999/datamind/assets/71152540/4c4c56a8-2373-4a33-9e01-01d661099c18)
![Figure_1](https://github.com/EddyS999/datamind/assets/71152540/96aa291f-807a-4477-a58d-42e15d671796)
