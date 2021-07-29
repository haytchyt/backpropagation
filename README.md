Neural Network
Neural Networks (NN) are a widely used data mining method for classification and grouping. It's an attempt to create a machine that can learn and mimic brain processes. NN learns by example most of the time. If enough examples are provided to NN, it should be able to classify data and possibly uncover new trends or patterns. The basic NN has three layers: input, output, and a hidden layer. Nodes from the input layer are connected to nodes from the hidden layer, and each layer can have a large number of nodes. The output layer nodes are coupled to the nodes from the hidden layer. Weights between nodes are represented by those connections.

Backpropagation Algorithm
The basic idea underlying the BP method is that the output of the NN is compared to the desired output. If the results aren't good enough, the connection (weights) between the layers are changed, and the procedure is repeated until the mistake is small enough.

Version
1.0

Requirement
Python 2.7+

How it use?
git clone https://github.com/haytchyt/backpropagation.git
cd backpropagation
python backpropagation.py
If you want modify experimental dataset, just put your datas on code

def demo():
    #  XOR function
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