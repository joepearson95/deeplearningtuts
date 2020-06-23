import torch
import torchvision
from torchvision import transforms, datasets

train = datasets.MNIST("", train=True, download=True,
                    transform = transforms.Compose([transforms.ToTensor()]))

test = datasets.MNIST("", train=False, download=True,
                    transform = transforms.Compose([transforms.ToTensor()]))

# batch = how many passed at a time to the model. Shuffling for no bias - similar to cards
trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)

import torch.nn as nn
import torch.nn.functional as F

# neural network class, inherriting the nn module.
# init is comprised of a self initialising part from nn.Module and creating a
# fully conected 1 (N) linear nn. Taking the images (input = 28*28 image) and hidden layers (3 layer 64 node), giving out 10 classes/neurons (numbers)
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)

    # feed forward creation using relu (Rectified Linear Unit) instead of any other activation function right now
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)  # softmax using 'x' and specified dimension - similar to axis

net = Net()
X = torch.rand((28,28))
X = X.view(1, 28*28)
print(net(X))
