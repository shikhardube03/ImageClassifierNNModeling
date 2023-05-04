import numpy as np
import pickle
import matplotlib.pyplot as plt
from math import sqrt, ceil
import cv2
import time
from scipy.special import cross_entropy

def flatten(x):
    """
    Flattens the input tensor x to a 2D tensor, where the first dimension is the batch size
    and the second dimension is the flattened feature vector of each example in the batch.
    """
    return np.reshape(x, (x.shape[0], -1))

def relu_grad(x):
    grad = np.zeros(x.shape)
    grad[x > 0] = 1
    return grad

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

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

class Train:
    def __init__(self, cnn):
        self.cnn = cnn
        self.loss = None

    def train(self, x_train, y_train, learning_rate=0.001, num_epochs=10, batch_size=32):
        num_examples = x_train.shape[0]
        num_batches = int(num_examples / batch_size)

        for epoch in range(num_epochs):
            epoch_loss = 0
            start_time = time.time()

            for batch in range(num_batches):
                batch_start = batch * batch_size
                batch_end = (batch + 1) * batch_size
                x = x_train[batch_start:batch_end]
                y = y_train[batch_start:batch_end]

                # Forward pass
                y_pred = self.cnn.forward(x)

                # Compute loss
                self.loss = cross_entropy(y_pred, y)
                epoch_loss += self.loss

                # Backward pass
                grad = y_pred - y
                for layer in reversed(self.cnn.fc_layers):
                    grad = grad * relu_grad(layer.last_input)
                    grad = layer.backward(layer.last_input, grad, learning_rate)
                grad = flatten(grad)
                for layer in reversed(self.cnn.conv_layers):
                    grad = grad.reshape(layer.last_input_shape)
                    grad = grad * relu_grad(layer.last_input)
                    grad = layer.backward(layer.last_input, grad, learning_rate)

            epoch_loss /= num_batches

            print("Epoch: {:3d} | Loss: {:.6f} | Time: {:.2f}s".format(epoch+1, epoch_loss, time.time()-start_time))

class LoadData:
    def __init__(self, data_files):
        self.data = []
        self.labels = []
        self.len_samples = 0
        for file in data_files:
            unp = unpickle(file)
            self.data += unp[b'data']
            self.labels += unp[b'labels']
            self.len_samples += len(unp[b'data'])
            
        self.labels = np.array(self.labels)
        self.data = np.array(self.data)

    def __getitem__(self, idx):
        temp = self.data[idx]
        label = self.labels[idx]

        image = []
        for i in range(32): #row
            row = []
            for j in range(32):
                #first red, second green, third blue
                color = (temp[(i*32)+j], temp[(i*32)+j+1024], temp[(i*32)+j+1024+1024])
                row.append(color)
            image.append(row)

        image = np.array(image)
        return image, label


#Testing Implementation below: 

traindata = ["cifar10_batches/data_batch_1", "cifar10_batches/data_batch_2", "cifar10_batches/data_batch_3", "cifar10_batches/data_batch_4", "cifar10_batches/data_batch_5"]
xtrain = []
ytrain = []
dataset = LoadData(traindata)
for image, label in iter(traindata):
    xtrain.append(image)
    ytrain.append(label)

testcnn = CNN(3)
trainedCNN = Train(testcnn)
trainedCNN.train(xtrain, ytrain)


