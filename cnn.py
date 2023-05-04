import numpy as np
import pickle
import matplotlib.pyplot as plt
from math import sqrt, ceil
import cv2
from timeit import default_timer as timer

def flatten(x):
    """
    Flattens the input tensor x to a 2D tensor, where the first dimension is the batch size
    and the second dimension is the flattened feature vector of each example in the batch.
    """
    return np.reshape(x, (x.shape[0], -1))

class CNN:
    def __init__(self, num_layers):
        self.num_layers = num_layers
        self.conv_layers = []
        self.fc_layers = []
        for i in range(num_layers):
            if i == 0:
                in_channels = 3 # for RGB images
                out_channels = 16
            else:
                in_channels = out_channels
                out_channels = 32
            self.conv_layers.append(ConvLayer(in_channels, out_channels))
        self.fc_layers.append(FCLayer(7 * 7 * 32, 128))
        self.fc_layers.append(FCLayer(128, 10)) # for 10-class classification
        
    def forward(self, x):
        for conv_layer in self.conv_layers:
            x = conv_layer.forward(x)
            x = relu(x)
            x = max_pool(x)
        x = flatten(x)
        for fc_layer in self.fc_layers:
            x = fc_layer.forward(x)
            x = relu(x)
        return x
    
class ConvLayer:
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size)
        self.bias = np.zeros((out_channels, 1))
        self.stride = stride
        self.padding = padding
        
    def forward(self, x):
        n, c, h, w = x.shape
        f, c, kh, kw = self.weights.shape
        out_h = int((h + 2 * self.padding - kh) / self.stride) + 1
        out_w = int((w + 2 * self.padding - kw) / self.stride) + 1
        x_pad = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        out = np.zeros((n, f, out_h, out_w))
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.stride
                h_end = h_start + kh
                w_start = j * self.stride
                w_end = w_start + kw
                x_slice = x_pad[:, :, h_start:h_end, w_start:w_end]
                out[:, :, i, j] = np.sum(x_slice * self.weights, axis=(2, 3, 4)) + self.bias
        return out
    
class FCLayer:
    def __init__(self, in_size, out_size):
        self.weights = np.random.randn(out_size, in_size) / np.sqrt(in_size)
        self.bias = np.zeros((out_size, 1))
        
    def forward(self, x):
        out = np.dot(self.weights, x) + self.bias
        return out
    
    def backward(self, x, grad_output, learning_rate):
        grad_input = np.dot(self.weights.T, grad_output)
        grad_weights = np.dot(grad_output, x.T)
        grad_bias = np.sum(grad_output, axis=1, keepdims=True)
        self.weights -= learning_rate * grad_weights
        self.bias -= learning_rate * grad_bias
        return grad_input
    
def relu(x):
    return np.maximum(0, x)

def max_pool(x, size=2, stride=2):
    n, c, h, w = x.shape
    out_h = int(np.ceil(float(h - size) / stride)) + 1
    out_w = int(np.ceil(float(w - size) / stride)) + 1
    out = np.zeros((n, c, out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            h_start = i * stride
            h_end = min(h_start + size, h)
            w_start = j * stride
            w_end = min(w_start + size, w)
            x_slice = x[:, :, h_start:h_end, w_start:w_end]
            out[:, :, i, j] = np.max(x_slice, axis=(2, 3))
    return out
