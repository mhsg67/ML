import pandas as ps
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import numpy as np

def showDigit(digit):
    im = digit
    im = np.reshape(im, (28, 28))  # "F" means read/write by 1st index changing fastest, last index slowest.
    plt.imshow(im, cmap='gray')
    plt.plot()
    plt.show()

def predict(classifier, X, expectedAns=None):
    showDigit(X)
    clf = classifier #goshadi
    if (expectedAns!=None):
        print("expected Answer is: " + str(expectedAns))

    print("predict proba: " + str(clf.predict_proba([X])))
    print("predict: " + str(clf.predict([X])))

digits = ps.read_csv('data/train.csv')
X = digits.drop(['label'], axis = 1)
Y = digits['label']


sgd_clf = MLPClassifier(solver='sgd', hidden_layer_sizes=(30,2), random_state=1)
sgd_clf.fit(X, Y)


predict(sgd_clf, X.iloc[2], Y.iloc[2])




