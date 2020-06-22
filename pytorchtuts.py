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

#print data
for data in trainset:
    print(data)
    break

#print images       tensors
x, y = data[0][0], data[1][0]
#print(y)
import matplotlib.pyplot as plt
plt.imshow(data[0][0].view(28,28))
plt.show()

# count the amounts and present as percentage
total = 0
counter_dict =  {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}

for data in trainset:
    Xs, Ys = data
    for y in Ys:
        counter_dict[int(y)] += 1
        total += 1

for i in counter_dict:
    print(f"{i}: {counter_dict[i]/total*100}")