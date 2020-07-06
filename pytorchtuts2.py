import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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

        #print(x[0].shape)
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

# If data is being rebuilt, make said training data
if REBUILD_DATA:
    dogsvcats = DogsVSCats()
    dogsvcats.make_training_data()

# Load the newly creataed training_data file and allow (some reason it caused an error)
training_data =  np.load("training_data.npy", allow_pickle=True)
print(len(training_data))
#print(training_data[0])
# after printing some data, show the image in grey scale
#plt.imshow(training_data[0][0], cmap="gray")
#plt.show()

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

# third param used to minimise errors
def fwd_pass(X, y, train=False):
    if train:
        net.zero_grad()

    # gets outputs based on first parameter and then obtains the matches from it.
    # then it computes the accuracy and loss
    outputs = net(X)
    matches = [torch.argmax(i) == torch.argmax(j) for i, j in zip(outputs, y)]
    acc = matches.count(True)/len(matches)
    loss = loss_func(outputs, y)

    if train:
        loss.backward()
        optimizer.step()
    return acc, loss

def test(size=32):
    X, y = test_X[:size], test_y[:size]
    with torch.no_grad():
        val_acc, val_loss = fwd_pass(X.view(-1, 1, 50, 50), y)
    return val_acc, val_loss

val_acc, val_loss = test(size=32)
print(val_acc, val_loss)
import time

MODEL_NAME = f"model-{int(time.time())}"
net = Net()
optimizer = optim.Adam(net.parameters(), lr = 0.001)
loss_func = nn.MSELoss()
print(MODEL_NAME)
def train():
    BATCH_SIZE = 100
    epochs = 8
    with open("model.log", "a") as f:
        for epoch in range(EPOCHS):
            for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
                batch_X = train_X[i:i+BATCH_SIZE].view(-1,1,50,50)
                batch_y = train_y[i:i+BATCH_SIZE]

                batch_X, batch_y = batch_X, batch_y

                acc, loss = fwd_pass(batch_X, batch_y, train=True)

                #print(f"Acc: {round(float(acc),2)}  Loss: {round(float(loss),4)}")
                #f.write(f"{MODEL_NAME},{round(time.time(),3)},train,{round(float(acc),2)},{round(float(loss),4)}\n")
                # just to show the above working, and then get out:
                if i % 50 == 0:
                    val_acc, val_loss = test(size=100)
                    f.write(f"{MODEL_NAME},{round(time.time(),3)},{round(float(acc),2)},{round(float(loss), 4)},{round(float(val_acc),2)},{round(float(val_loss),4)},{epoch}\n")


import matplotlib.pyplot as plt
from matplotlib import style

style.use("ggplot")

model_name = "model-1593688481" # grab whichever model name you want here. We could also just reference the MODEL_NAME if you're in a notebook still.


def create_acc_loss_graph(model_name):
    contents = open("model.log", "r").read().split("\n")

    times = []
    accuracies = []
    losses = []

    val_accs = []
    val_losses = []

    for c in contents:
        if model_name in c:
            name, timestamp, acc, loss, val_acc, val_loss, epoch = c.split(",")

            times.append(float(timestamp))
            accuracies.append(float(acc))
            losses.append(float(loss))

            val_accs.append(float(val_acc))
            val_losses.append(float(val_loss))


    fig = plt.figure()

    ax1 = plt.subplot2grid((2,1), (0,0))
    ax2 = plt.subplot2grid((2,1), (1,0), sharex=ax1)


    ax1.plot(times, accuracies, label="acc")
    ax1.plot(times, val_accs, label="val_acc")
    ax1.legend(loc=2)
    ax2.plot(times,losses, label="loss")
    ax2.plot(times,val_losses, label="val_loss")
    ax2.legend(loc=2)
    plt.show()

create_acc_loss_graph(model_name)
