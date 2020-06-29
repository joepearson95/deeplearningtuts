import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

# State if data is being built each time - used for bigger datasets
REBUILD_DATA = False

class DogsVSCats():
    IMG_SIZE = 50 # making sure all images are the same size
    CATS = "PetImages/Cat" # these two strings are filepaths
    DOGS = "PetImages/Dog"
    LABELS = {CATS: 0, DOGS: 1} # similar to ohe
    training_data = []
    catcount = 0
    dogcount = 0

    def make_training_data(self):
        # Loop the folders
        for label in self.LABELS:
            print(label)
            # loop the images with progress bar, convert to greyscale and resize for simplicity
            for f in tqdm(os.listdir(label)):
                try:
                    path = os.path.join(label, f)
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                    # training data of the images in OHE
                    self.training_data.append([np.array(img), np.eye(2)[self.LABELS[label]]])

                    # if cat, add 1. otherwise, add 1 to dog
                    if label == self.CATS:
                        self.catcount += 1
                    elif label == self.DOGS:
                        self.dogcount += 1
                    else:
                        self.catcount += 1
                except Exception as e:
                    # some images aren't good, will cause error.
                    #print(str(e))
                    pass
        np.random.shuffle(self.training_data)
        np.save("training_data.npy", self.training_data)
        print("Cats: ", self.catcount)
        print("Dogs: ", self.dogcount)

# If data is being rebuilt, make said training data
if REBUILD_DATA:
    dogsvcats = DogsVSCats()
    dogsvcats.make_training_data()

# Load the newly creataed training_data file and allow (some reason it caused an error)
training_data =  np.load("training_data.npy", allow_pickle=True)
#print(len(training_data))
#print(training_data[0])
# after printing some data, show the image in grey scale
#plt.imshow(training_data[0][0], cmap="gray")
#plt.show()
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # input of 1, 32 conv features, kernel size of 5 (5*5 window/kernel)
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)

        # Has to move to a linear layer. The below three lines are for doing so
        # Done by 'passing fake data through it'. In order to figure out the shape
        x = torch.randn(50,50).view(-1,1,50,50)
        self._to_linear = None
        self.convs(x) # serves as part of the 'forward' method

        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512,2)

    # Similar to forward method, but only performed on certain layers
    def convs(self, x):
        # max pooling essentially gets the max value from each section
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2,2))

        print(x[0].shape)
        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x) # pass all conv layers
        x = x.view(-1, self._to_linear) #flatten image
        x = F.relu(self.fc1(x)) # pass through first fully connected layer
        x = self.fc2(x) # pass through final fully connected layer before
        return F.softmax(x, dim = 1) # x is batch of "x's". dim 1 would be the distribution

net = Net()

import torch.optim as optim

optimizer = optim.Adam(net.parameters(), lr = 0.001)
loss_func = nn.MSELoss()

X = torch.Tensor([i[0] for  i in training_data]).view(-1, 50, 50)
X = X / 255.0
y = torch.Tensor([i[1] for i in training_data])

VAL_PCT = 0.1
val_size = int(len(X) * VAL_PCT)

train_X  = X[:-val_size]
train_y  = y[:-val_size]

test_X  = X[-val_size:]
test_y  = y[-val_size:]
print(len(test_X))
print(len(train_X))

BATCH_SIZE = 100
EPOCHS = 1

for epoch in range(EPOCHS):
    # start at 0, go for length of training and step the size of batch size
    for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
        batch_X = train_X[i:i+BATCH_SIZE].view(-1, 1, 50, 50)
        batch_y = train_y[i:i+BATCH_SIZE]

        net.zero_grad()
        outputs = net(batch_X)
        loss = loss_func(outputs, batch_y)
        loss.backward()
        optimizer.step()

print(loss)
correct = 0
total = 0
with torch.no_grad():
    for i in tqdm(range(len(test_X))):
        real_class = torch.argmax(test_y[i])
        net_out = net(test_X[i].view(-1,1,50,50))[0]
        predicted_class = torch.argmax(net_out)
        if predicted_class == real_class:
            correct += 1
        total += 1
print("Acc %:", round(correct/total, 3))
