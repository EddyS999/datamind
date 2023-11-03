from datamind import SimpleLinearRegression as slr
from sklearn.datasets import make_regression
import matplotlib .pyplot as plt 
import numpy as np



####### test du package  ######

np.random.seed(0)
x,y = make_regression(n_samples=100, n_features = 1, noise = 10) 
X = np.hstack((x, np.ones(x.shape)))#ajout d'un biais
y = y.reshape(y.shape[0],1)
modele = slr()

final = modele.fit(X, y)
cost_history = modele.cost_history(X, y)
#print(final)
#print(cost_history)
prediction = modele.main_function(X, final)
print(modele.score(y, prediction))
'''
plt.plot(range(1000), cost_history)
plt.show()

plt.scatter(x,y)
plt.plot(x, prediction, c='r')
plt.show()
'''