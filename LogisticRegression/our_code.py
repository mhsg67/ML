from numpy import loadtxt, exp, array, transpose, log,  dot, zeros, ones, append


def sigmoid(func):
    func = min(func, 200)
    func = max(func, -200)

    return 1/(1 + exp(-1*func))


def hypothesis(theta,x):
    return sigmoid(dot(x, transpose(theta)))


def cost_func(x, y, theta):
    num_test = len(x)
    sum = 0
    for i in range(num_test):
        ht = hypothesis(theta, x[i])
        if not(ht == 0 or ht == 1):
            sum += (y[i]*log(ht)) + ((1 - y[i])*log(1 - ht))

    return (-1*sum)/num_test

'''
    theta should be n*1
    x should be m*n
'''

def gd(theta, x, y, num_iter,alpha):
    num_test = len(y)
    num_feature = len(theta)
    for iter in range(num_iter):
        ht = zeros(num_test)
        for k in range(num_test):
            ht[k] = hypothesis(theta,x[k])

        for j in range(num_feature):
            sum = 0
            for i in range(num_test):
                sum += (ht[i] - y[i]) * x[i][j]
            theta[j] = theta[j] - alpha * sum

    return theta


def calculate(x, y):
    num_feature = len(x[0])
    theta = zeros(num_feature) #(1,n+1)
    alpha = 0.3

    return gd(theta, x, y, 22, alpha)


def run():
    data = loadtxt('data/ex2data1.txt', delimiter=',')
    x = data[:99, 0:2]
    y = data[:99, 2] #(m,1)
    z = data[99:, 0:2]
    ones_col = ones((len(x), 1))
    x_new = append(ones_col, x, 1) #(m,n+1)

    z_pri = [52.34800398794107,60.76950525602592]

    theta = calculate(x_new,y)

    print(hypothesis(theta, z_pri))


if __name__ == "__main__":
    run()


