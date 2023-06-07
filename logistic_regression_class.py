import numpy as np
class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.w = None
        
    

    def sigmoid(self, summ):
        return 1/(1+np.exp(-summ)) #המרה לפונקציה לוגיסטית

    def gradient(self, X, p, y):
        return np.dot(X.T, (p - y)) / y.shape[0]  #הנגזרת של log loss

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        
        for i in range(self.n_iters): #gradient descent
            p = self.sigmoid(np.dot(X, self.w)) #הסתברויות הנחזות
            gradient_val = self.gradient(X, p, y)
            self.w -= (gradient_val * self.learning_rate)
        return self
    
    def predict(self, X):
        p = self.sigmoid(np.dot(X, self.w)) 
        return np.round(p)
   