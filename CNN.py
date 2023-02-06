import torch
import torch.nn as nn
import torchvision.transforms as T
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import Subset, SubsetRandomSampler
from torchvision.datasets import ImageFolder
from evaluation import *
from preProcessing import *

model_no_ = 3


# Our proposed first model
class CNN1(nn.Module):
    def __init__(self):
        super().__init__()
        # Define the convolutional layers:
        self.convolutional_layers = nn.Sequential(
            # 1st layer
            # input channels 1=> Grayscale
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),  # for normalization, and to accelerate the process
            nn.LeakyReLU(inplace=True),  # activation function
            # Tensor shape (32, 64 , 64, 64)

            # 2st layer
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),  # for normalization, and to accelerate the process
            nn.LeakyReLU(inplace=True),  # activation function

            # 3rd layer
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),  # for normalization, and to accelerate the process
            nn.LeakyReLU(inplace=True),  # activation function

            # 4th layer
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),  # for normalization, and to accelerate the process
            nn.LeakyReLU(inplace=True),  # activation function

            # 5th layer
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 6th layer
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

        )

        # Define the Fully connected/ linear layers:
        # No_of_feature = (W-F + 2*P)/S  + 1
        self.linear_layers = nn.Sequential(
            # Randomly zeros out some elements of input tensor
            nn.Dropout(p=0.1),
            nn.Linear(16 * 16 * 128, 1000),
            nn.ReLU(inplace=True),

            nn.Linear(1000, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),

            # The output layer)
            nn.Linear(512, 5)
        )

    def forward(self, tensor):
        # calling the convolutional layers defined in our class
        tensor = self.convolutional_layers(tensor)
        # flattening the tensor using view function
        # tensor.size(0) returns the batch size (32) which should
        # remain constant, -1 tells to calc the other dimensions
        # print(tensor.size(0))
        tensor = tensor.view(tensor.size(0), -1)
        # calling the linear layers defined in our class
        tensor = self.linear_layers(tensor)
        return tensor


# Our proposed 2nd model or Main/Shallow model
class CNN2(nn.Module):
    def __init__(self):
        super().__init__()
        # Define the convolutional layers:
        self.convolutional_layers = nn.Sequential(
            # 1st layer
            # input channels 1=> Grayscale
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),  # for normalization, and to accelerate the process
            nn.LeakyReLU(inplace=True),  # activation function
            # Tensor shape (32, 64 , 64, 64)

            # 2nd layer
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Tensor shape (32, 64 , 32, 32)

            # 3rd layer
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Tensor shape (32, 128 , 16, 16)

        )

        # Define the Fully connected/ linear layers:
        # No_of_feature = (W-F + 2*P)/S  + 1
        self.linear_layers = nn.Sequential(
            # Randomly zeros out some elements of input tensor
            nn.Dropout(p=0.1),
            nn.Linear(16 * 16 * 128, 1000),
            nn.ReLU(inplace=True),

            nn.Linear(1000, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),

            # The output layer)
            nn.Linear(512, 5)
        )

    def forward(self, tensor):
        # calling the convolutional layers defined in our class
        tensor = self.convolutional_layers(tensor)
        # flattening the tensor using view function
        # tensor.size(0) returns the batch size (32) which should
        # remain constant, -1 tells to calc the other dimensions
        # print(tensor.size(0))
        tensor = tensor.view(tensor.size(0), -1)
        # calling the linear layers defined in our class
        tensor = self.linear_layers(tensor)
        return tensor


# Our proposed 3rd model
class CNN3(nn.Module):
    def __init__(self):
        super().__init__()
        # Define the convolutional layers:
        self.convolutional_layers = nn.Sequential(
            # 1st layer
            # input channels 1=> Grayscale
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),  # for normalization, and to accelerate the process
            nn.LeakyReLU(inplace=True),  # activation function
            # Tensor shape (32, 64 , 64, 64)

            # 2st layer
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),  # for normalization, and to accelerate the process
            nn.LeakyReLU(inplace=True),  # activation function

            # 3rd layer
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),  # for normalization, and to accelerate the process
            nn.LeakyReLU(inplace=True),  # activation function
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 4th layer
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),  # for normalization, and to accelerate the process
            nn.LeakyReLU(inplace=True),  # activation function
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 5th layer
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 6th layer
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

        )

        # Define the Fully connected/ linear layers:
        # No_of_feature = (W-F + 2*P)/S  + 1
        self.linear_layers = nn.Sequential(
            # Randomly zeros out some elements of input tensor
            nn.Dropout(p=0.1),
            nn.Linear(4 * 4 * 128, 1000),
            nn.ReLU(inplace=True),

            nn.Linear(1000, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),

            # The output layer)
            nn.Linear(512, 5)
        )

    def forward(self, tensor):
        # calling the convolutional layers defined in our class
        tensor = self.convolutional_layers(tensor)
        # flattening the tensor using view function
        # tensor.size(0) returns the batch size (32) which should
        # remain constant, -1 tells to calc the other dimensions
        # print(tensor.size(0))
        tensor = tensor.view(tensor.size(0), -1)
        # calling the linear layers defined in our class
        tensor = self.linear_layers(tensor)
        return tensor


'''
A method to run the training process 
'''


def run(model, train_loader, learning_rate, epochs):
    # create an instance of our convolutional network class
    cnn = model()
    lossCriterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
    total_step = len(train_loader)
    loss_list = []
    accuracy_list = []
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):
            output = cnn(images)
            loss = lossCriterion(output, labels)
            loss_list.append((epoch, loss.item()))

            # Applying backpropagation and optimisation
            optimizer.zero_grad()  # no need for gradient in pur classification
            loss.backward()
            optimizer.step()

            # Calculating the accuracy of the cnn during the training process
            totalImages = labels.size(0)
            # max_indexes represents the highest predicted class
            max_elements, max_indexes = torch.max(output.data, 1)
            imagesClassifiedCorrectly = (max_indexes == labels).sum().item()
            accuracy_list.append((epoch, (imagesClassifiedCorrectly / totalImages)))

            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, epochs, i + 1, total_step, loss.item(),
                          (imagesClassifiedCorrectly / totalImages) * 100))
    return cnn


def CNN_project_01(model_no_):
    dataset, my_categories = pre_processing()

    cnn_model = kfold_train_and_evaluate(dataset, my_categories)

    torch.save(cnn_model, top_path + '/trained_model_k_fold.pth')


def kfold_train_and_evaluate(dataset, my_categories):
    torch.manual_seed(42)
    batch_size = 128
    k = 10
    # random_state is like a random seed
    splits = KFold(n_splits=k, shuffle=True, random_state=42)
    # learning_rate = .001
    # np.arrange returns evenly spaced values within a range
    # so gives us an array with evenly spaced values between 0 and dataset.length
    # so [0, 1, 2 ... dataset.length-1]
    # we do this because we are only interested in the indexes of the the dataset
    # we use the split method to split the indexes into train and test indexes
    # fold is like the fold number for example fold 1.
    initial_test_acc = 0
    for fold, (train_idx, test_idx) in enumerate(splits.split(np.arange(len(dataset)))):
        print('Fold Number: {}'.format(fold + 1))
        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(test_idx)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)
        model = choose_network_and_train_model(train_loader)
        test_acc = evaluate(model, test_loader, my_categories)
        if test_acc > initial_test_acc:
            initial_test_acc = test_acc
            saved_model = model

    return saved_model


def choose_network_and_train_model(train_loader):
    if model_no_ == 1:
        cnn_model = run(CNN1, train_loader, learning_rate=0.0001, epochs=8)
    elif model_no_ == 2:
        cnn_model = run(CNN2, train_loader, learning_rate=0.0001, epochs=8)
    else:
        cnn_model = run(CNN3, train_loader, learning_rate=0.0001, epochs=8)

    return cnn_model


if __name__ == '__main__':
    CNN_project_01(model_no_)
