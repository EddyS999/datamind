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
teta_final = model.fit(X, y)
#print(teta_final)

prediction = model.predict(X,y)
print(prediction)

# Scoring 
#prediction = model.main_function(X, final)
#print(model.score(y, prediction))
