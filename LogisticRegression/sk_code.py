from numpy import loadtxt
from sklearn import linear_model as lrm


def run():
    data = loadtxt('data/ex2data1.txt', delimiter=',')
    x = data[:, 0:2]
    x_train = x[:-1]
    x_test = x[-1:]

    y = data[:, 2]
    y_train = y[:-1]
    y_test = y[-1:]

    lr = lrm.LogisticRegression()
    lr.fit(x_train, y_train)

    if lr.predict(x_test) == y_test[0]:
        print("YES")
    else:
        print("NO")


if __name__ == "__main__":
    run()