'''
https://www.kaggle.com/c/digit-recognizer
'''

import pandas as ps
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import numpy as np
from sklearn.externals import joblib
import os

def showDigit(digit):
    im = digit
    im = np.reshape(im, (28, 28))  # "F" means read/write by 1st index changing fastest, last index slowest.
    plt.imshow(im, cmap='gray')
    plt.plot()
    plt.show()

def predict(classifier, X, expectedAns=None,toPrint=False):
    #showDigit(X)
    clf = classifier #goshadi
    if (expectedAns!=None and toPrint):
        print("expected Answer is: " + str(expectedAns))

    result = clf.predict([X])[0]
    if (toPrint):
        print("predict proba: " + str(clf.predict_proba([X])))
        print("predict: " + str(result))

    if (expectedAns == None):
        return result

    if (result == expectedAns):
        return 1
    else:
        return 0

def output_submission(clf):
    X = ps.read_csv('data/test.csv')
    with open('solution.csv', 'w') as file:
        file.write("ImageId,Label\n")
        for i in range(0,X.__len__()):
            result = predict(clf,X.iloc[i])
            file.write(str(i+1) + "," + str(result) + "\n")



digits = ps.read_csv('data/train.csv')
X = digits.drop(['label'], axis = 1)
Y = digits['label']

clf = None
model_file_name = "model_150_logistic.pkl"
if os.path.isfile(model_file_name):
    clf = joblib.load(model_file_name)
else:
    sgd_clf = MLPClassifier(solver='sgd', hidden_layer_sizes=(150,), activation='logistic', random_state=1)
    sgd_clf.fit(X, Y)
    clf = sgd_clf
    joblib.dump(clf, model_file_name)

output_submission(clf)

# tp = 0
# size = digits.__len__()
# for i in range(0,size):
#     tp = tp + (int)(predict(clf,X.iloc[i],Y.iloc[i]))
# print((tp)/(size*1.)*100)

