import pandas as ps
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np




digits = ps.read_csv('data/train.csv')
X = digits.drop(['label'], axis = 1)
Y = digits['label']


#clf = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(5,2), random_state=1)
#clf.fit(X, Y)
#print(clf.classes_)


im = X.loc[1]  # picked a row at random
im = np.reshape(im, (28, 28), order="F")  #"F" means read/write by 1st index changing fastest, last index slowest.
plt.imshow(im,cmap='gray')
plt.plot()


#print(clf.predict_proba([X.iloc[1]]))
#print(Y.loc[1])
#print
#print(clf.predict_proba(X.iloc[2]))
#print(Y.loc[2])
#print
#print(clf.predict_proba([X.iloc[3]]))
#print(Y.loc[3])

