# load dataset into memory
# 28x28 = 784 pixels, which starts out as an array 
import numpy as np
import os, glob
from PIL import Image
import sys
# np.set_printoptions(threshold=sys.maxsize) 
def main():
    
    # m = number of test cases that have been added
    trainingPath = "../trainingdata/Reduced MNIST Data/Reduced Trainging data"
    # testingPath = "../trainingdata/Reduced MNIST Data/Reduced Testing data"
    trainingPaths = get_one_pathlist(trainingPath)
    # originalTrainingPaths = trainingPaths.copy()
    # np.random.shuffle(trainingPaths)
    # trainingPaths.reverse()
    # training, testing = get_dataset()
    sc = np.sqrt(2/784)
    sc2 = np.sqrt(2/10)
    W1 = np.random.normal(loc=0, scale=sc,size=(10,784)) # (low=.0001,high=.5,size=(10,784))
    W2 = np.random.normal(loc=0, scale=sc2,size=(10,10)) # randint(low=1, high=10, size=(10,10)) # (loc=0, scale=sc2,size=(10,10)) # (low=.0001, high=.5,size=(10,10))
    
    # common bias is 1
    b1 = np.zeros(shape=(10,1))
    b2 = np.zeros(shape=(10,1))
    # already an np array
    nextPath = trainingPaths.pop()

    learningRate = .001
    
    X = []
    Y = []
    c = 0
    while len(nextPath) > 0:
        
        if c > 100:
            break
        c += 1
        nextTarget = get_parent_path(nextPath)
        nextDataset = path_to_pixels(nextPath)
        # .001 every 100 can adjust a whole cycle,  .1 = every 10, .2 every 5
        
        # Y is subtracted from the final probabilities
        Y.append(fill_at(int(nextTarget)))
        X.append(nextDataset)
        
        # number is the layer. 0 = input, 1 = hidden, 2 = output
        # A is the input set, b is the bias
        # W are weights that are modified by the forward and backward propagation
        # print()
        A0 = np.transpose(X) # m x 784 -> 784 x m

        
        # train the system
        # A2 is a list of probabilities, 10 total, one for each digit. These total up to 1.
        Z1, Z2, A1, A2 = forward_propagation(A0, W1, W2, b1, b2)
        
        
        # take what we learned from the training and apply it to our weights so we can learn
        W1, W2, b1, b2 = backward_propagation(A0, W1, W2, Z1, A1, A2, b1, b2, np.transpose(Y), len(X), learningRate)
        #print(A2)
        # print("YT", np.transpose(Y))
        if c > 90:
            print(nextPath, nextTarget)
            print("A0")
            print(A0)        
            print("A1")
            print(np.around(A1, 3))
            print("A2")
            print(np.around(A2, 3), A2[-1:-1])
            print("W1")
            print(W1)
            print("W2")
            print(W2.shape)
            print("b")
            print(b1,b2)
            # print("Weights",np.max(W1), np.max(W2))
            # print("b avgs",np.average(b1),np.average(b2))
            # print(b1,b2)
            # pop another item off the image stack and attempt to process it too
        try:
            # if c > 10:
            #     nextPath = originalTrainingPaths.pop()
                        
            # else:
            nextPath = trainingPaths.pop()
        except IndexError:
            nextPath = ''
        # nextTarget = get_parent_path(nextPath)
        # nextDataset = path_to_pixels()

def fill_at(y):
     # fake y, which represents the target number    
    # y in vector form, example: 5 is [0,0,0,0,1,0,0,0,0,0]. 2 would be [0,1,0,0,0,0,0,0,0,0]
    Y = np.zeros(10)
    Y[y] = 1
    # return np.transpose(Y)
    return Y

def hidden_layer(A0, W1, b):
    a = np.dot(W1, A0)
    Z1 = a + b # linear regression, produces a 10 x m array
    # A1 = np.tanh(Z1) #  activation function
    A1 = np.vectorize(relu)(Z1) # second activation
    return Z1,A1

# A0 is also X transposed
def forward_propagation(A0, W1, W2, b1, b2):
    # hidden layer
    Z1, A1 = hidden_layer(A0, W1, b1)   
    
    # output layer
    Z2 = np.dot(W2, A1) + b2

    # certainy/probability adjacent
    A2 = softmax(Z2)

    return (Z1, Z2, A1, A2)

def backward_propagation(A0, W1, W2, Z1, A1, A2, b1, b2, Y, m, learningRate):
    # [.05,.1,.15,.7] -> [0,0,0,1] -> .05, .1, .15, -.3]
    
    dZ2 = A2 - Y
    dW2 = 1/m * np.dot(dZ2, np.transpose(A1))
    db2 = 1/m * np.reshape(np.sum(dZ2, axis=1), (b2.shape[0], 1))

    dZ1 = np.dot(np.transpose(W2), dZ2) * np.vectorize(relu_prime)(Z1)
    dW1 = 1/m * np.dot(dZ1, np.transpose(A0))
    db1 = 1/m * np.reshape(np.sum(dZ1, axis=1), (b1.shape[0], 1))
    
    W1 = W1 - learningRate * dW1
    b1 = b1 - learningRate * db1
    W2 = W2 - learningRate * dW2
    b2 = b2 - learningRate * db2

    return (W1,W2,db1,db2)

# non-linear activation function
def softmax(V):
    
    eV = np.exp(V)
    # eV = np.exp(V)
    return eV / np.sum(eV, axis=0)
    

# 1 - g(x) ^ 2
def tanh_prime(V):
    return 1-(np.tanh(V) * np.tanh(V))
# def softmax_prime(V):
#     smv = softmax(V)
#     return np.dot(smv, (np.identity(V.size) - np.sum(smv)))


def relu_prime(v):
    if v < 0: 
        return 0
    if v >= 0:
        return 1

# linear activation function
def relu(v):
    if v < 0:
        return 0
    return v

# def tanh(x):
#     # sigmoid, (0,1)
#     #return 1/(1 + np.exp(np.e, -v)) 
#     # tanh(x) - (-1, 1)
#     top = np.exp(np.e, x) - np.exp(np.e, -x)
#     bottom = np.exp(np.e, x) + np.exp(np.e, -x)
#     return top/bottom


def get_parent_path(path):
    return os.path.basename(os.path.dirname(path))

def image_to_pixels(image):
    pixels = image.getdata()
    return np.array(pixels)
    # return np.ndarray(shape=(1,784), dtype=int, buffer=np.array(pixels))

# def get_both_datasets():
#     trainingPath = "Reduced MNIST Data/Reduced Trainging data"
#     testingPath = "Reduced MNIST Data/Reduced Testing data"

#     return (get_one_dataset(trainingPath), get_one_dataset(testingPath))


def get_one_pathlist(path):
    return glob.glob(os.path.normpath(path+'/**/*.jpg'),recursive=True)

def path_to_pixels(path):
    with Image.open(path) as image:
        return image_to_pixels(image)

# def get_one_dataset(path):
#     data = {}
    
#     for p in glob.glob(path+'/**/*.jpg',recursive=True):
#         with Image.open(p) as im:
#             target = get_parent_path(p)
#             found = data.get(target)
#             if found == None:
#                 found = []
#             found.append(image_to_pixels(im))
#             data[target] = found
 
#     return data

# def menu():


if __name__ == '__main__':
    # menu loop
    # p - forward propagation, then backward propagation, then apply deltas to bias and weights
    # l - learning rate options - "l #" to adjust to a number (0, 1), eg "l 0.45". 
    # p - print results of last test
    # h - this menu
    # e or x - exit
    # option = input().lower()
    # while option != 'x' or option != 'e':

    #     if option == 'f':
    #         pass
    #     option = input().lower()
    main()