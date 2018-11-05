import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.svm import SVC

np.random.seed(22)

means = [[2, 2], [4, 2]]
cov = [[.3, .2], [.2, .3]]
N = 10
X0 = np.random.multivariate_normal(means[0], cov, N)  # class 1
X1 = np.random.multivariate_normal(means[1], cov, N)  # class -1
X = np.concatenate((X0.T, X1.T), axis=1)  # all data
y = np.concatenate((np.ones((1, N)), -1*np.ones((1, N))), axis=1)  # labels


y1 = y.reshape((2*N,))
X1 = X.T
clf = SVC(kernel='linear', C=1e5)

clf.fit(X1, y1)
w = clf.coef_
b = clf.intercept_
print('w = ', w)
print('b = ', b)
