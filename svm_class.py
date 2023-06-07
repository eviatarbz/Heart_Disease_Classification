import numpy as np


class SVM:
    def __init__(self, learning_rate=0.01, lambda_param=0.05, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def _init_weights_bias(self, X):
        #לקבוע משקל והטיה התחלתיים
        n_features = X.shape[1]
        self.w = np.zeros(n_features)
        self.b = 0

    def _get_cls_map(self, y):
        #משנה ערכי 0 ל- 1-
        return np.where(y == 0, -1, 1)

    def _satisfy_constraint(self, x, idx):
        #האם שורה צריכה לקבל קנס (האם היא מיסקלסיפייד)
        linear_model = np.dot(x, self.w) + self.b 
        return self.cls_map[idx] * linear_model >= 1
    
    def _get_gradients(self, constrain, x, idx):
        #פונקציה שמוצאת את הגרדיאנט עבור w ו b מסויימים
        if constrain: #אם נקודה מסווגת נכון
            dw = self.lambda_param * self.w
            db = 0
            return dw, db
        #אם לא (גרדיאנט שונה)
        dw = self.lambda_param * self.w - np.dot(self.cls_map[idx], x)
        db = - self.cls_map[idx]
        return dw, db
    
    def _update_weights_bias(self, dw, db):
        #עדכון של המקדם ומשקל בהתאם לגרדיאנט
        self.w -= self.lr * dw
        self.b -= self.lr * db
    
    def fit(self, X, y):
        self._init_weights_bias(X)
        self.cls_map = self._get_cls_map(y)

        for _ in range(self.n_iters):
            for idx, x in enumerate(X):
                constrain = self._satisfy_constraint(x, idx) #האם הנקודה מקיימת את התנאי
                dw, db = self._get_gradients(constrain, x, idx) #הנגזרות
                self._update_weights_bias(dw, db)
    
    def predict(self, X):
        #יצירת hyperplane וחיזוי לפיו
        estimate = np.dot(X, self.w) + self.b
        prediction = np.sign(estimate)
        return np.where(prediction == -1, 0, 1)

