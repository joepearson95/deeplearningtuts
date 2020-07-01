import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import numpy as np
import matplotlib.pyplot as plt

class CNN(nn.Module):
    # straightforward initialisation func, sets everything up and hardcodes num_classes due to the dataset being used
    def __init__(self, lr, epochs, batch_size, num_classes=10):
        super(CNN, self).__init__()
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.loss_history = []
        self.acc_history = []
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu') # if using GPU
        self.conv1 = nn.Conv2d(1, 32, 3) # 1 channel, 32 conv filters, 3*3
        self.bn1 = nn.BatchNorm2d(32) # helps enable smoother training through normalisation
        self.conv2 = nn.Conv2d(32,32,3)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32,32,3)
        self.bn3 = nn.BatchNorm2d(32)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv4 = nn.Conv2d(32, 64, 3)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64,64, 3)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64,64,3)
        self.bn6 = nn.BatchNorm2d(64)
        self.maxpool2 = nn.MaxPool2d(2)

        input_dims = self.calc_input_dims() # automated dimension calculation saves having to do the maths

        self.fc1 = nn.Linear(input_dims, self.num_classes)
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.loss = nn.CrossEntropyLoss()
        self.to(self.device)
        self.get_data()

    def calc_input_dims(self):
        batch_data = T.zeros((1, 1, 28, 28)) #instead of hardwiring batch size, it is 1
        batch_data = self.conv1(batch_data)
        #batch_data = self.bn1(batch_data)
        batch_data = self.conv2(batch_data)
        #batch_data = self.bn2(batch_data)
        batch_data = self.conv3(batch_data)
        batch_data = self.maxpool1(batch_data)
        batch_data = self.conv4(batch_data)
        batch_data = self.conv5(batch_data)
        batch_data = self.conv6(batch_data)
        batch_data = self.maxpool2(batch_data)

        return int(np.prod(batch_data.size()))

    def forward(self, batch_data):
        batch_data = T.tensor(batch_data).to(self.device)

        #feed forwards
        batch_data = self.conv1(batch_data)
        batch_data = self.bn1(batch_data) # before or after relu can effect the results when more complex
        batch_data = F.relu(batch_data)

        batch_data = self.conv2(batch_data)
        batch_data = self.bn2(batch_data)
        batch_data = F.relu(batch_data)

        batch_data = self.conv3(batch_data)
        batch_data = self.bn3(batch_data)
        batch_data = F.relu(batch_data)

        batch_data = self.maxpool1(batch_data)

        batch_data = self.conv4(batch_data)
        batch_data = self.bn4(batch_data)
        batch_data = F.relu(batch_data)

        batch_data = self.conv5(batch_data)
        batch_data = self.bn5(batch_data)
        batch_data = F.relu(batch_data)

        batch_data = self.conv6(batch_data)
        batch_data = self.bn6(batch_data)
        batch_data = F.relu(batch_data)

        batch_data = self.maxpool2(batch_data)
        batch_data = batch_data.view(batch_data.size()[0], -1) # flatten on 0th element

        classes = self.fc1(batch_data)
        return classes

    def get_data(self):
        mnist_train_data = MNIST('mnist', train=True, download=True, transform=ToTensor())
        self.train_data_loader = T.utils.data.DataLoader(mnist_train_data,
                                 batch_size=self.batch_size, shuffle=True,
                                 num_workers=8) # num_workers = threads
        mnist_test_data = MNIST('mnist', train=False, download=True, transform=ToTensor())
        self.train_data_loader = T.utils.data.DataLoader(mnist_test_data,
                                 batch_size=self.batch_size, shuffle=True,
                                 num_workers=8)

    def _train(self):
        self.train()
        for i in range(self.epochs):
            ep_loss = 0
            ep_acc = []
            # enumerate data
            for j, (input, label) in enumerate(self.train_data_loader):
                # accumulates from training step. This can cause issues. Thus, zero gradient at each train step
                self.optimizer.zero_grad()
                label = label.to(self.device)
                prediction = self.forward(input) # feedforward pass for this iterations input
                loss = self.loss(prediction, label) # loss for deep NN
                # to observe network
                prediction = F.softmax(prediction, dim=1) # actual prediction of the class.
                classes = T.argmax(prediction, dim=1)
                # where wrongly classified, make it 1. else, 0
                wrong = T.where(classes != label,
                                T.tensor([1.]).to(self.device),
                                T.tensor([0.]).to(self.device))
                acc = 1 - T.sum(wrong) / self.batch_size
                ep_acc.append(acc.item()) # .item gets the value, not the obj tensor
                self.acc_history.append(acc.item())
                ep_loss += loss.item()
                # this is required, otherwise no learning is done
                loss.backward()
                self.optimizer.step()
            print('Finish epoch ', i, 'total loss %.3f' % ep_loss, 'accuracy %.3f' % np.mean(ep_acc))
            self.loss_history.append(ep_loss)

    def _test(self):
        self.test()
        ep_loss = 0
        ep_acc = []
        # enumerate data
        for j, (input, label) in enumerate(self.test_data_loader):
            label = label.to(self.device)
            prediction = self.forward(input) # feedforward pass for this iterations input
            loss = self.loss(prediction, label) # loss for deep NN
            # to observe network
            prediction = F.softmax(prediction, dim=1) # actual prediction of the class.
            classes = T.argmax(prediciton, dim=1)
            # where wrongly classified, make it 1. else, 0
            wrong = T.where(classes != label,
                            T.tensor([1.]).to(self.device),
                            T.tensor([0.]).to(self.device))
            acc = 1 - T.sum(wrong) / self.batch_size
            ep_acc.append(acc.item()) # .item gets the value, not the obj tensor
            ep_loss += loss.item()
        print('Finish epoch ', i, 'total loss %.3f' % ep_loss, 'accuracy %.3f' % np.mean(ep_acc))

if __name__ == '__main__':
    net = CNN(lr=0.001, batch_size=128, epochs=25)
    net._train()

    plt.plot(net.loss_history)
    plt.show()
    plt.plot(net.acc_history)
    plt.show()
