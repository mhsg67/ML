from numpy import loadtxt, where, exp, array, transpose, dot, zeros, ones,append
from pylab import scatter, show, legend, xlabel, ylabel


def sigmoid(func):
    return 1/(1 + exp(func))


def hypothesis(theta,x):
    return sigmoid(dot(x, theta.T))


def cost(output, expected):
    diff = (output - expected)
    power_diff = diff*diff
    power_diff.mean()/2

'''
    theta should be n*1
    x should be m*n
'''
def gd(theta, x, y, num_iter,alpha):
    num_test = len(y)
    num_feature = len(theta)
    for iter in range(num_iter):
        ht = []
        for k in range(num_test):
            ht[k] = hypothesis(theta,x[k])

        for j in range(num_feature):
            sum = 0
            for i in range(num_test):
                sum += (ht[i] - y[i]) * x[i][j]
            theta[j] = theta[j] - alpha * sum

    return theta

def calculate(x,y):
    num_feature = len(x)
    o = ones((len(y), 1))
    x = append(o, x, 1)
    theta = zeros(num_feature+1)
    alpha = 0.3

    return gd(theta,x,y,20,alpha)


def run():
    data = loadtxt('data/ex2data1.txt', delimiter=',')
    x = data[:, 0:2]
    y = data[:, 2]

    theta = calculate(x,y)

    print("theta is:")
    print(theta)

    totalCost = 0

def test():
    x = array([[1,2],[3,4],[5,6]])
    o = ones((3,1))
    temp = append(o, x,1)
    print(temp)
    theta = array ([1,1,1])
    y = [0, 1, 1]


if __name__ == "__main__":
    run()

'''data = loadtxt('data/ex2data1.txt', delimiter=',')

X = data[:, 0:2]
y = data[:, 2]

pos = where(y == 1)
neg = where(y == 0)

scatter(X[pos, 0], X[pos, 1], marker='o', c='g')
scatter(X[neg, 0], X[neg, 1], marker='x', c='r')
xlabel('Exam 1 score')
ylabel('Exam 2 score')
legend(['Not Admitted', 'Admitted'])
show()'''

