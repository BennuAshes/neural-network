import unittest
from nn import *
import numpy as np
from PIL import Image
import glob


class TestNeuralNetwork(unittest.TestCase):
    def test_d(self):
        input = np.array([10,8,9,12,11])
        print(softmax(input))
        print(softmax(input/10))
        print(softmax(input-1))
        print(softmax(input-20))
        print(softmax(input-np.max(input)))
        a = np.array([[10.54727489],
            [28],
            [24],
            [15],
            [16],
            [23],
            [30],
            [22],
            [18],
            [24]])
        print(softmax(a-np.max(a)))
    def test_softmax(self):
        input = np.array([1.3,5.1,2.2,.7,1.1])
        np.testing.assert_almost_equal(softmax(input), [0.0201905, 0.9025377, 0.0496605, 0.0110808, 0.0165306])
    # def test_softmax_larger_numbers(self):
    #     input = np.array([101,105,99,70,50])
    #     b = np.ones(5)
    #     cleaned = input - np.max(input) + b
    #     dirty = softmax(input)
    #     clean = softmax(cleaned)
    #     np.testing.assert_almost_equal(dirty, clean)
    def test_relu(self):
        self.assertEqual(relu(-1),0)
        self.assertEqual(relu(-2),0)
        self.assertEqual(relu(1),1)
        self.assertEqual(relu(53),53)
    def test_relu_prime(self):
        self.assertEqual(relu_prime(32), 1)
        self.assertEqual(relu_prime(0), 0)
        self.assertEqual(relu_prime(-3), 0)
    def test_parent_path(self):
        p = "Reduced MNIST Data/Reduced Trainging data/0/4924.jpg"
        self.assertEqual(get_parent_path(p), '0')

    # def test_pixels(self):
    #     with Image.open('Reduced MNIST Data/Reduced Trainging data/0/5842.jpg') as image:
    #         # d = list(map(lambda rgb : (rgb[0]+rgb[1]+rgb[2])/(3), image.getdata()))
    #         d = list(image.getdata())
    #         # print(len(d))
    #         # print(d)
    # def test_images_to_pixels(self):
    #     answer = [
    #         0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 1, 10, 6, 0, 5, 0, 6, 0, 0, 6, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 0, 21, 7, 6, 0, 0, 3, 0, 2, 0, 27, 5, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 0, 0, 8, 0, 8, 0, 21, 0, 1, 0, 0, 15, 13, 0, 0, 0, 0, 0, 0, 
    #         0, 0, 0, 0, 0, 0, 0, 6, 0, 5, 18, 0, 0, 14, 0, 0, 4, 0, 24, 3, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 241, 255, 255, 250, 255, 0, 6, 0, 0, 11, 18, 1, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 0, 9, 244, 255, 255, 247, 255, 255, 253, 248, 255, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 239, 255, 255, 241, 246, 255, 235, 255, 255, 5, 4, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 254, 253, 255, 247, 255, 247, 255, 255, 255, 254, 255, 254, 255, 0, 4, 0, 0, 0, 0, 4, 0, 0, 12, 8, 0, 10, 0, 0, 255, 255, 245, 255, 245, 255, 255, 255, 245, 255, 255, 254, 247, 0, 3, 0, 0, 0, 0, 2, 9, 0, 0, 0, 0, 14, 12, 255, 230, 253, 255, 0, 0, 255, 246, 255, 249, 255, 250, 255, 248, 19, 0, 0, 0, 0, 0, 0, 0, 3, 0, 11, 5, 0, 0, 255, 245, 255, 243, 8, 19, 0, 0, 0, 0, 246, 
    #         255, 238, 255, 244, 6, 0, 0, 0, 0, 1, 0, 11, 0, 0, 0, 39, 249, 240, 255, 253, 0, 0, 0, 10, 0, 2, 16, 2, 255, 255, 246, 255, 0, 0, 0, 0, 0, 17, 0, 0, 0, 9, 0, 0, 244, 255, 240, 255, 9, 10, 0, 8, 0, 0, 0, 8, 250, 249, 255, 0, 8, 0, 0, 0, 0, 0, 0, 12, 1, 0, 0, 0, 255, 244, 255, 3, 0, 1, 0, 5, 1, 18, 0, 3, 242, 252, 255, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 6, 11, 6, 244, 255, 255, 0, 0, 2, 0, 0, 0, 0, 10, 10, 255, 255, 238, 0, 8, 0, 0, 0, 0, 0, 5, 1, 3, 0, 0, 5, 255, 242, 255, 7, 4, 0, 13, 0, 8, 6, 0, 255, 244, 245, 255, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 4, 1, 0, 255, 247, 255, 0, 3, 0, 0, 15, 0, 0, 5, 254, 253, 255, 244, 8, 0, 0, 0, 0, 0, 6, 2, 0, 0, 0, 10, 17, 255, 255, 0, 8, 0, 10, 5, 0, 0, 255, 244, 255, 250, 255, 255, 0, 8, 0, 0, 0, 0, 4, 0, 0, 16, 4, 0, 0, 242, 242, 255, 0, 0, 9, 0, 0, 255, 237, 255, 254, 255, 242, 6, 3, 0, 0, 0, 0, 0, 6, 0, 1, 1, 0, 0, 2, 255, 255, 238, 255, 255, 248, 255, 255, 255, 255, 255, 255, 251, 2, 0, 11, 4, 0, 0, 0, 0, 0, 2, 12, 0, 0, 17, 0, 251, 243, 255, 255, 255, 252, 248, 253, 248, 255, 243, 255, 252, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 11, 0, 7, 1, 0, 255, 255, 247, 251, 255, 253, 255, 255, 255, 250, 255, 239, 25, 1, 0, 5, 4, 0, 0, 0, 0, 8, 0, 0, 8, 2, 0, 22, 243, 253, 255, 255, 255, 244, 255, 254, 250, 255, 254, 7, 0, 0, 10, 3, 0, 0, 0, 0, 0, 0, 14, 0, 8, 0, 3, 0, 15, 248, 255, 254, 241, 255, 252, 252, 255, 0, 0, 11, 0, 4, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    #     ]
    #     with Image.open('Reduced MNIST Data/Reduced Trainging data/0/5842.jpg') as image:
    #         pixels = image_to_pixels(image)
    #         np.testing.assert_almost_equal(
    #             pixels,np.ndarray(shape=(28,28), dtype=int, buffer=np.array(answer)))
    # def test_get_one_dataset(self):
    #     d = get_one_dataset('Reduced MNIST Data/Reduced Trainging data/0')
    #     print(d)
    def test_get_one_pathlist(self):
        self.assertEqual(len(get_one_pathlist('Reduced MNIST Data/Reduced Trainging data/0')), 1000)
    # def test_path_to_pixels(self):
    #     pixels = path_to_pixels('Reduced MNIST Data/Reduced Trainging data/0/5842.jpg')
    #     self.assertEqual(pixels.shape, (28,28))
    # def test_hidden_layer(self):
    #     A0 = np.zeros(shape=(784,5))

    #     A0[0][0] = 255
    #     A0[0][1] = 255
    #     A0[0][2] = 255
    #     A0[0][3] = 0
    #     A0[0][4] = 255
        
    #     W1 = generate_initial_weights()
    #     W2 = generate_initial_weights()
    #     b1 = np.ones(10)
    #     b2 = np.ones(10)
    #     Z1, G, A1 = hidden_layer(A0, W1, b1)
    #     print(A0, Z1, G, A1)
    # def test_asdf(self):
    #     x = []
    #     x.append(np.random.rand(784))
        
    #     a0 = np.transpose(x)
    #     w1 = np.random.rand(10, 784)
    #     d = np.dot(w1, a0)
    #     print(d, d.shape, d.dtype)

        
    # def test_forward_propagation(self):
    #     # pretend scenario where 5 is the target value, 
    #     # which is represnted by a corner in the top left (0,0) being
    #     slice = np.zeros(shape=(784,1))
    #     slice[0] = 255
    #     A0 = np.zeros(shape=(784,1))
       
    #     A0[0] = 255
        
    #     W1 = np.random.rand(10, 784)
    #     W2 = np.random.rand(10, 10)
    #     print(W1[0][0],W1[9][9])
    #     print(W2[0][0],W2[9][9])
    #     # print(W1, W2)
    #     b1 = np.ones(shape=(10,1))
    #     b2 = np.ones(shape=(10,1))
        
    #     # print(W1,W2)
    #     for x in range(8000):
    #         # print(A0)
    #         Z1, Z2, A1, A2 = forward_propagation(A0, W1, W2, b1, b2)
    #         Y = fill_at(5)
    #         m = A0.shape[1] # current sample size
    #         W1, W2, b1, b2 = backward_propagation(
    #             A0, W1, W2, Z1, A1, A2, b1, b2, Y, m, .001
    #         )
    #         A0 = np.append(A0, slice, axis=1)
    #     print('---------------------------------')
    #     print(W1[0][0],W1[9][9])
    #     print(W2[0][0],W2[9][9])
if __name__ == '__main__':
    unittest.main()