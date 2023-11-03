# Simple linear regression implementation
This is a Python project of implementation of the Simple Linear Regression algorithm, a basic machine learning algorithm for predicting a continuous target variable based on a single feature. I chose to the approach used in scikit-learn to implement the functions.
## Overview 
Simple Linear Regression is a fundamental technique in machine learning. It models the relationship between a dependent variable (target) and an independent variable (feature) using a linear equation f(x) = ax+b. This implementation provides a simple and educational example of how the algorithm works.
## Features
- Fits a simple linear regression model to your data.
- Generates cost history to track the optimization process.
- Provides a scoring function to evaluate the model's performance.
## Usage

Here, we're using library with generated data from sklearn. 

\```python
# Import 
from datamin import SimpleLinearRegression
from sklearn.datasets import make_regression
import matplotlib .pyplot as plt 
import numpy as np

# Linear regression with fit()
model = SimpleLinearRegression()
final = model.fit(X, y)

# Scoring 
prediction = model.main_function(X, final)
print(model.score(y, prediction))

\```
