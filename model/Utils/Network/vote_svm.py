import numpy as np

from sklearn.svm import LinearSVC

class VoteSVM():
    def __init__(self, svm_number: int, max_iter: int):
        self.svm_number = svm_number
        self.max_iter = max_iter
        self.svms = [LinearSVC(max_iter=max_iter) for _ in range(svm_number)]
    
    def fit(self, X, y):
        assert len(X) == self.svm_number
        
        for i in range(self.svm_number):
            # print('Fitting SVM', i, '...')
            x = np.array(X[i])
            self.svms[i].fit(x, y)
        
    def predict(self, X):
        assert len(X) == self.svm_number

        y_preds = []
        for i in range(self.svm_number):
            x = np.array(X[i])
            y_preds.append(self.svms[i].predict(x))
        
        y_pred = []
        for i in range(len(X[0])):
            one_num = 0
            for j in range(self.svm_number):
                if y_preds[j][i] == 1:
                    one_num += 1
            if one_num > 2:
                y_pred.append(1)
            else:
                y_pred.append(0)
        
        return y_pred