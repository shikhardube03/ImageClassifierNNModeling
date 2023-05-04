import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms


from models import resnet18


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


"""
1.  Define and build a PyTorch Dataset
"""
class CIFAR10(Dataset):
    def __init__(self, data_files, transform=None, target_transform=None):
        """
        Initialize your dataset here. Note that transform and target_transform
        correspond to your data transformations for train and test respectively.
        """
     
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

        self.transform = transform
        self.target_transform = target_transform
        

    def __len__(self):
        """
        Return the length of your dataset here.
        """
        return self.len_samples

    def __getitem__(self, idx):
        """
        Obtain a sample from your dataset. 

        Parameters:
            x:      an integer, used to index into your data.

        Outputs:
            y:      a tuple (image, label), although this is arbitrary so you can use whatever you would like.
        """
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
        
        if self.target_transform:
            label = self.target_transform(label)
        if self.transform:
            image = self.transform(image)
        
        return image, label
                
    

def build_dataset(data_files, transform=None):
    """
    Parameters:
        data_files:      a list of strings e.g. "cifar10_batches/data_batch_1" corresponding to the CIFAR10 files to load data
    Outputs:
        dataset:      a PyTorch dataset object to be used in training/testing
    """
    return CIFAR10(data_files, transform)


"""
2.  Build a PyTorch DataLoader
"""
def build_dataloader(dataset, loader_params):
    """
    Parameters:
        dataset:         a PyTorch dataset to load data
        loader_params:   a dict containing all the parameters for the loader. 
        
    Please ensure that loader_params contains the keys "batch_size" and "shuffle" corresponding to those 
    respective parameters in the PyTorch DataLoader class. 

    Outputs:
        dataloader:      a PyTorch dataloader object to be used in training/testing
    """
    #dataset is a CIFAR10 object
    if "batch_size" not in loader_params or "shuffle" not in loader_params: 
        return None
    
    return DataLoader(dataset, loader_params["batch_size"], loader_params["shuffle"])
    #raise NotImplementedError("You need to write this part!")


"""
3. (a) Build a neural network class.
"""
class FinetuneNet(torch.nn.Module):
    def __init__(self):
        """
        Initialize your neural network here. Remember that you will be performing finetuning
        in this network so follow these steps:
        
        1. Initialize convolutional backbone with pretrained model parameters.
        2. Freeze convolutional backbone.
        3. Initialize linear layer(s). 
        """
        super().__init__()
        ################# Your Code Starts Here #################

        #backbone = resnet18(pretrained=True)
        backbone = resnet18()
        backbone.load_state_dict(torch.load('resnet18.pt'))
        for param in backbone.parameters():
            param.requires_grad = False

        num_classes = 8
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, num_classes)

        self.backbone = backbone
        
        ################## Your Code Ends here ##################

    def forward(self, x):
        """
        Perform a forward pass through your neural net.

        Parameters:
            x:      an (N, input_size) tensor, where N is arbitrary.

        Outputs:
            y:      an (N, output_size) tensor of output from the network
        """
        ################# Your Code Starts Here #################
        return self.backbone(x)
        ################## Your Code Ends here ##################


"""
3. (b)  Build a model
"""
def build_model(trained=False):
    """
    Parameters:
        trained:         a bool value specifying whether to use a model checkpoint

    Outputs:
        model:           the model to be used for training/testing
    """
    net = FinetuneNet()
    if trained:
        pass
    return net


"""
4.  Build a PyTorch optimizer
"""
def build_optimizer(optim_type, model_params, hparams):
    """
    Parameters:
        optim_type:      the optimizer type e.g. "Adam" or "SGD"
        model_params:    the model parameters to be optimized
        hparams:         the hyperparameters (dict type) for usage with learning rate 

    Outputs:
        optimizer:       a PyTorch optimizer object to be used in training
    """
    if optim_type == "SGD":
        return torch.optim.SGD(model_params, lr=0.0001)
    if optim_type == "Adam":
        return torch.optim.Adam(model_params, lr=0.0001)
    

"""
5. Training loop for model
"""
def train(train_dataloader, model, loss_fn, optimizer):
    """
    Train your neural network.

    Iterate over all the batches in dataloader:
        1.  The model makes a prediction.
        2.  Calculate the error in the prediction (loss).
        3.  Zero the gradients of the optimizer.
        4.  Perform backpropagation on the loss.
        5.  Step the optimizer.

    Parameters:
        train_dataloader:   a dataloader for the training set and labels
        model:              the model to be trained
        loss_fn:            loss function
        optimizer:          optimizer
    """

    ################# Your Code Starts Here #################
    for image, label in iter(train_dataloader):
        pred = model(image)
        l = loss_fn(pred, label.long())
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    ################## Your Code Ends here ##################


"""
6. Testing loop for model
"""
def test(test_dataloader, model):
    """
    This part is optional.

    You can write this part to monitor your model training process.

    Test your neural network.
        1.  Make sure gradient tracking is off, since testing set should only
            reflect the accuracy of your model and should not update your model.
        2.  The model makes a prediction.
        3.  Calculate the error in the prediction (loss).
        4.  Print the loss.

    Parameters:
        test_dataloader:    a dataloader for the testing set and labels
        model:              the model that you will use to make predictions


    Outputs:
        test_acc:           the output test accuracy (0.0 <= acc <= 1.0)
    """

    # test_loss = something
    # print("Test loss:", test_loss)
    raise NotImplementedError("You need to write this part!")

"""
7. Full model training and testing
"""
def run_model():
    """
    The autograder will call this function and measure the accuracy of the returned model.
    Make sure you understand what this function does.
    Do not modify the signature of this function (names and parameters).

    Please run your full model training and testing within this function.

    Outputs:
        model:              trained model
    """
    dataset = build_dataset(["cifar10_batches/data_batch_1", "cifar10_batches/data_batch_2", "cifar10_batches/data_batch_3",
                             "cifar10_batches/data_batch_4", "cifar10_batches/data_batch_5"],
                             transform=transforms.ToTensor())
    params = {"batch_size": 128, "shuffle": True}
    dataloader = build_dataloader(dataset, params)
    m = build_model()
    optimizer = build_optimizer(optim_type="Adam", model_params=m.parameters(), hparams={"lr": 0.0001})
    train(dataloader, m, torch.nn.CrossEntropyLoss(), optimizer)
    return m