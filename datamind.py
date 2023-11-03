import numpy as np
import matplotlib.pyplot as plt 


class SimpleLinearRegression:

    def __init__(self) -> None:
        pass

    def main_function(self, X, O):
        return X.dot(O)

    #descente de gradient matricielle
    def fit(self, X, y, learning_rate=0.01, max=1000):
        np.random.seed(0)
        m_O = 0 #vect
        O = np.random.randn(2,1) #vecteur teta contenant 2 hypers a et b paramètres (aX+b)
        m = len(y)
        for i in range(max):
            fx = X.dot(O)#self.cost(X,y,O)
            gradient = 1/m * X.T.dot(fx - y)#transposé 
            O = O - learning_rate * gradient
            #fx = 1/(2*m) * np.sum((X.dot(O) - y)**2)
            max -= 1
        m_O = O
        return m_O
    
    #Permet de vérifier l'efficacite de la descente de gradient
    def cost_history(self, X, y, learning_rate=0.01, max=1000):
        np.random.seed(0)
        O = np.random.randn(2,1)
        m = len(y)
        cost_history = np.zeros(max)
        for i in range(max):
            fx = X.dot(O)#self.cost(X,y,O)
            gradient = 1/m * X.T.dot(fx - y)
            O = O - learning_rate * gradient
            cost_history[i] = 1/(2*m) * np.sum((X.dot(O) - y)**2) #self.cost(X,y,O)
            #fx = 1/(2*m) * np.sum((X.dot(O) - y)**2)
            max -= 1
       
        return cost_history
    

    def score(self, y, prediction):
        u = ((y - prediction)**2).sum()
        v = ((y - y.mean())**2).sum()
        return 1 - u/v


    def predict(self, X, y):
        return X.dot(self.fit(X, y))




class LinearRegression:

    def __init__(self) -> None:
        pass
