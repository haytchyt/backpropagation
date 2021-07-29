#coded by Harrison McDonagh
#to run on windows use command py backpropagation.py

import math
import random
random.seed(0)

# gives a random number where a <= randomNumber < b
def randomNumber(a, b):
    return (b-a)*random.random() + a

# gives a matrix
def genMatrix(I, J, fill=0.0):
    matrix = []
    for i in range(I):
        matrix.append([fill]*J)
    return matrix

# the sigmoid function, tanh is better than 1/(1+e^-x)
def getSigmoid(x):
    return math.tanh(x)

# derivative the sigmoid function, i.e output y
def getDsigmoid(y):
    return 1.0 - y**2

class newNetwork:
    def __init__(self, ni, nh, no):
        # n(x) is number of input, hidden, and output nodes
        self.ni = ni + 1 # for bias node
        self.nh = nh
        self.no = no

        # activation
        self.ai = [1.0]*self.ni
        self.ah = [1.0]*self.nh
        self.ao = [1.0]*self.no

        # create weights, w(x) is weight of input, and output nodes
        self.wi = genMatrix(self.ni, self.nh)
        self.wo = genMatrix(self.nh, self.no)
        # set to random vaules
        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j] = randomNumber(-0.2, 0.2)
        for j in range(self.nh):
            for k in range(self.no):
                self.wo[j][k] = randomNumber(-2.0, 2.0)

        # change in weights for momentum, c(x) for change in input, and output nodes
        self.ci = genMatrix(self.ni, self.nh)
        self.co = genMatrix(self.nh, self.no)

    def weights(self):
        print('Input weights:')
        for i in range(self.ni):
            print(self.wi[i])
        print()
        print('Output weights:')
        for j in range(self.nh):
            print(self.wo[j])

    def update(self, inputs):
        if len(inputs) != self.ni-1:
            raise ValueError('Incorrect number of inputs')

        # input activations, ai for activations in input nodes
        for i in range(self.ni-1):
            #self.ai[i] = getSigmoid(inputs[i])
            self.ai[i] = inputs[i]

        # hidden activations, ah for activations in hidden nodes
        for j in range(self.nh):
            sum = 0.0
            for i in range(self.ni):
                sum = sum + self.ai[i] * self.wi[i][j]
            self.ah[j] = getSigmoid(sum)

        # output activations, ao for activations in hidden nodes
        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                sum = sum + self.ah[j] * self.wo[j][k]
            self.ao[k] = getSigmoid(sum)

        return self.ao[:]

    def test(self, patterns):
        for p in patterns:
            print(p[0], '->', self.update(p[0]))


    def backPropagation(self, targets, N, M):
        if len(targets) != self.no:
            raise ValueError('Incorrect number of target values')

        # calculate error for output
        outputDeltas = [0.0] * self.no
        for k in range(self.no):
            error = targets[k]-self.ao[k]
            outputDeltas[k] = getDsigmoid(self.ao[k]) * error

        # calculate error for hidden
        hiddenDeltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error = error + outputDeltas[k]*self.wo[j][k]
            hiddenDeltas[j] = getDsigmoid(self.ah[j]) * error

        # update output weights
        for j in range(self.nh):
            for k in range(self.no):
                change = outputDeltas[k]*self.ah[j]
                self.wo[j][k] = self.wo[j][k] + N*change + M*self.co[j][k]
                self.co[j][k] = change

        # update input weights
        for i in range(self.ni):
            for j in range(self.nh):
                change = hiddenDeltas[j]*self.ai[i]
                self.wi[i][j] = self.wi[i][j] + N*change + M*self.ci[i][j]
                self.ci[i][j] = change

        # calculate error
        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5*(targets[k]-self.ao[k])**2
        return error

    def train(self, patterns, iterations=1000, N=0.5, M=0.1):
        # N is learning rate
        # M is momentum factor
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error = error + self.backPropagation(targets, N, M)
            if i % 100 == 0:
                print('error %-.5f' % error)


def demo():
    # XOR function
    pat = [
        [[0,0], [0]],
        [[0,1], [1]],
        [[1,0], [1]],
        [[1,1], [0]]
    ]

    # create a network with two input, two hidden, and one output nodes
    n = newNetwork(2, 2, 1)
    # train it with patterns
    n.train(pat)
    # test with XOR
    n.test(pat)



if __name__ == '__main__':
    demo()