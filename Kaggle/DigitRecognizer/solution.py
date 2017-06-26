import pandas as ps
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import numpy as np

def ShowDigit(digit):
    im = digit
    im = np.reshape(im, (28, 28), order="F")  # "F" means read/write by 1st index changing fastest, last index slowest.
    plt.imshow(im, cmap='gray')
    plt.plot()

def predict(classifier, X, expectedAns=None):
    clf = classifier #goshadi
    if (expectedAns!=None):
        ShowDigit(expectedAns)
        print("expected Answer is: " + expectedAns)

    print("predict proba: " + clf.predict_proba([X]))
    print("predict: " + clf.predict([X]))

digits = ps.read_csv('data/train.csv')
X = digits.drop(['label'], axis = 1)
Y = digits['label']


sgd_clf = MLPClassifier(solver='sgd', hidden_layer_sizes=(30,2), random_state=1)
sgd_clf.fit(X, Y)


predict(sgd_clf, X.iloc[2], Y.iloc[2])




